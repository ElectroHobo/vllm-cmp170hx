# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recovery of damaged DeepSeek V4.1 tool calls.

V4.1 sometimes writes a tool call without its proper DSML markup, mostly in
long multi-turn tool use and near the context limit. Each form below used to
lose the call or leak its markup into the reply:

- a near-miss wrapper spelling (``<｜DSML｜calls>``, ``<｜DSML｜_calls>``): the
  wrapper leaked into the reply as text;
- a call in the generic XML idiom, ``<invoke name="TOOL">`` or the invoke
  fused into its first parameter as ``<parameter name="TOOL">``, with plain
  ``<parameter name="...">`` children and DSML or plain closers: the whole
  call leaked into the reply;
- a headless call, only the parameter tags before the invoke closer: the tool
  is the single declared tool whose parameters cover every parameter written.

The damaged forms are held as provisional candidates (vllm.sm80.
parser_recovery): a candidate becomes a tool call only once its invoke
closer arrives; an XML-idiom candidate also needs a declared tool name, and a
headless one exactly one covering declared tool. Anything else, including an
ambiguous headless block, is restored as text. Well-formed DSML takes the
exact upstream path, and text inside the thinking block stays reasoning, as
upstream keeps it.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from typing import Any

import regex as re

from vllm.parser.deepseek_v4 import DeepSeekV4Parser, _unwrap_wrapper_args
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine_config import ParserEngineConfig, ParserState
from vllm.sm80.parser_recovery import RecoveringParserEngine, RecoveryTransition
from vllm.tool_parsers.utils import iter_response_function_tool_info

TOOL_START_NEAR_MISS = ("<｜DSML｜calls>", "<｜DSML｜_calls>")
TOOL_END_NEAR_MISS = ("</｜DSML｜calls>", "</｜DSML｜_calls>")
XML_INVOKE_PREFIX = '<invoke name="'
XML_PARAM_PREFIX = '<parameter name="'
XML_INVOKE_END = "</invoke>"
XML_TOOL_END = ("</calls>", "</function_calls>")

# Parameter markup that stays inside a held candidate's arguments (the
# parameter tags close with the invoke's '">').
_ARGS_TERMINALS = (
    "PARAM_START",
    "PARAM_CLOSE",
    "INVOKE_NAME_END",
    "INVOKE_PREFIX_PARAM",
)

# Parameter tags of a damaged call: DSML, plain or mixed, ``string`` optional.
_PARAM_OPEN = re.compile(
    r'<(?:｜DSML｜ )?parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>'
)
_PARAM_CLOSE = re.compile(r"</(?:｜DSML｜ )?parameter>")
_CALL_CLOSE = re.compile(r"</(?:｜DSML｜ )?(?:invoke|calls|function_calls)>")
# A plain (non-DSML) parameter or invoke opener marks the XML idiom.
_PLAIN_OPEN = re.compile(r'<(?!｜DSML｜ )(?:parameter|invoke)\s+name="')


@functools.cache
def _recovery_config(
    base_config: Callable[..., ParserEngineConfig], thinking: bool
) -> ParserEngineConfig:
    config = base_config(thinking=thinking)
    terminals = dict(config.terminals)
    terminals["TOOL_START"] = (
        *_spellings(terminals["TOOL_START"]),
        *TOOL_START_NEAR_MISS,
    )
    terminals["TOOL_END"] = (
        *_spellings(terminals["TOOL_END"]),
        *TOOL_END_NEAR_MISS,
        *XML_TOOL_END,
    )
    terminals["INVOKE_PREFIX_XML"] = XML_INVOKE_PREFIX
    terminals["INVOKE_PREFIX_PARAM"] = XML_PARAM_PREFIX
    terminals["INVOKE_END_XML"] = XML_INVOKE_END

    transitions = dict(config.transitions)
    for state in (
        ParserState.CONTENT,
        ParserState.TOOL_PREAMBLE,
        ParserState.TOOL_BETWEEN,
    ):
        for terminal, headless in (
            ("INVOKE_PREFIX_XML", False),
            ("INVOKE_PREFIX_PARAM", True),
        ):
            transitions[(state, terminal)] = RecoveryTransition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
                provisional=True,
                headless=headless,
            )
    end = config.transitions[(ParserState.TOOL_ARGS, "INVOKE_END")]
    commit = RecoveryTransition(
        end.next_state, end.events, end.skip_in_token_id_mode, commit=True
    )
    transitions[(ParserState.TOOL_ARGS, "INVOKE_END")] = commit
    transitions[(ParserState.TOOL_ARGS, "INVOKE_END_XML")] = commit
    return replace(config, terminals=terminals, transitions=transitions)


def _spellings(value: str | tuple[str, ...]) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else value


def _tool_schemas(tools: Sequence[Any]) -> Iterator[tuple[str, dict, list]]:
    """Yield ``(name, properties, required)`` of every declared function."""
    for tool in tools:
        if isinstance(tool, dict):
            fn = tool.get("function") or tool
            infos = [(fn.get("name"), fn.get("parameters"))]
        elif getattr(tool, "function", None) is not None:
            infos = [(tool.function.name, tool.function.parameters)]
        else:
            infos = iter_response_function_tool_info(tool)
        for name, params in infos:
            if name:
                params = params if isinstance(params, dict) else {}
                yield (
                    name,
                    params.get("properties") or {},
                    list(params.get("required") or []),
                )


def _needs_heal(raw_args: str) -> bool:
    """Whether arguments carry the XML idiom; well-formed DSML never does."""
    if _PLAIN_OPEN.search(raw_args):
        return True
    # A value before the first parameter opener is a fused invoke's value.
    first_close = _PARAM_CLOSE.search(raw_args)
    if first_close is None:
        return False
    first_open = _PARAM_OPEN.search(raw_args)
    return (
        first_open is None or first_close.start() < first_open.start()
    ) and bool(raw_args[: first_close.start()].strip())


def _primary_param(props: dict, required: list) -> str | None:
    """The parameter a fused invoke's leading value belongs to."""
    for name in required:
        prop = props.get(name)
        if isinstance(prop, dict) and prop.get("type") in (None, "string"):
            return name
    for name in required:
        if name in props:
            return name
    for name, prop in props.items():
        if isinstance(prop, dict) and prop.get("type") == "string":
            return name
    if props:
        return next(iter(props))
    return required[0] if required else None


def _healed_value(name: str, is_str: str | None, value: str, props: dict) -> Any:
    if is_str == "true":
        return value
    prop = props.get(name)
    if is_str is None and isinstance(prop, dict) and prop.get("type") == "string":
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _heal_params(raw_args: str, props: dict, required: list) -> str:
    """Arguments of a DSML/plain/mixed parameter blob as a JSON object."""
    params: dict[str, Any] = {}
    opens = list(_PARAM_OPEN.finditer(raw_args))
    open_starts = [m.start() for m in opens]
    close_starts = [m.start() for m in _PARAM_CLOSE.finditer(raw_args)]

    first_open = open_starts[0] if open_starts else len(raw_args)
    lead_close = next((s for s in close_starts if s < first_open), None)
    if lead_close is not None and (lead := raw_args[:lead_close].strip()):
        primary = _primary_param(props, required)
        if primary:
            params[primary] = _healed_value(primary, None, lead, props)

    for m in opens:
        start = m.end()
        ends = [s for s in close_starts if s >= start]
        ends += [s for s in open_starts if s > m.start()]
        if call_close := _CALL_CLOSE.search(raw_args, start):
            ends.append(call_close.start())
        end = min(ends) if ends else len(raw_args)
        params[m.group(1)] = _healed_value(
            m.group(1), m.group(2), raw_args[start:end], props
        )
    return json.dumps(params, ensure_ascii=False)


def recovering_parser(parser_cls: type[DeepSeekV4Parser]) -> type[DeepSeekV4Parser]:
    """The V4.1 DSML parser class with damaged-call recovery."""
    base_config = parser_cls.parser_config

    class RecoveringParser(parser_cls):  # type: ignore[valid-type, misc]
        parser_config = staticmethod(functools.partial(_recovery_config, base_config))

        def __init__(self, tokenizer, tools=None, **kwargs) -> None:
            super().__init__(tokenizer, tools, **kwargs)
            self._declared_tools: list = list(tools or [])
            RecoveringParserEngine.adopt(
                self._engine,
                name_validator=self._accept_recovered_name,
                headless_candidates=self._covering_tools,
                args_terminals=_ARGS_TERMINALS,
            )

        def _check_skip_tool_parsing(self, request) -> None:
            super()._check_skip_tool_parsing(request)
            # The tools of this request only: a request without tools never
            # recovers a call by an earlier request's tool names.
            self._declared_tools = list(getattr(request, "tools", None) or [])

        def _accept_recovered_name(self, name: str) -> bool:
            return any(
                tool_name == name
                for tool_name, _, _ in _tool_schemas(self._declared_tools)
            )

        def _covering_tools(self, raw_args: str) -> list[str]:
            names = {m.group(1) for m in _PARAM_OPEN.finditer(raw_args)}
            if not names:
                return []
            return [
                tool_name
                for tool_name, props, _ in _tool_schemas(self._declared_tools)
                if props and names <= props.keys()
            ]

        def _headless_tool(self, raw_args: str) -> str | None:
            covering = self._covering_tools(raw_args)
            return covering[0] if len(covering) == 1 else None

        def _try_extract_name(self, idx: int) -> str | None:
            return super()._try_extract_name(idx) or self._headless_tool(
                self._tool_slots[idx].args
            )

        def _convert_args(self, raw_args: str, partial: bool) -> str:
            if not _needs_heal(raw_args):
                return super()._convert_args(raw_args, partial)
            name = next(
                (s.name for s in self._tool_slots if s.args == raw_args and s.name),
                None,
            ) or self._headless_tool(raw_args)
            props: dict = {}
            required: list = []
            for tool_name, tool_props, tool_required in _tool_schemas(
                self._declared_tools
            ):
                if tool_name == name:
                    props, required = tool_props, tool_required
                    break
            return _unwrap_wrapper_args(
                _heal_params(raw_args, props, required), self._tools, name
            )

    RecoveringParser.__name__ = parser_cls.__name__
    RecoveringParser.__qualname__ = parser_cls.__qualname__
    return RecoveringParser
