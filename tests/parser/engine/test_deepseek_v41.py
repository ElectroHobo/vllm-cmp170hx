# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.replay_harness import (
    DUMMY_TOOLS,
    MockTokenizer,
    _test_request,
    collect_output,
    replay_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.deepseek_v41 import DeepSeekV41Parser, deepseek_v41_config
from vllm.parser.parser_manager import ParserManager

CALLS = (
    '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="city" string="true">杭州</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="count" string="false">42</｜DSML｜ parameter>\n'
    '</｜DSML｜ invoke>\n<｜DSML｜ invoke name="add">\n'
    '<｜DSML｜ parameter name="x" string="false">1.5</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="y" string="false">2.25</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)


def tokenizer_for(text, special_calls):
    vocab = {"<think>": 50, "</think>": 51}
    if special_calls:
        vocab |= {"<｜DSML｜ calls>": 52, "</｜DSML｜ calls>": 53}
    tokens: list[tuple[int, str]] = []
    while text:
        special = next((marker for marker in vocab if text.startswith(marker)), None)
        piece = special or text[0]
        tokens.append((vocab[piece] if special else 100 + len(tokens), piece))
        text = text[len(piece) :]
    return MockTokenizer(vocab, tokens), tokens


def parser_for(tokenizer, controls):
    cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v41",
        reasoning_parser_name="deepseek_v41",
        enable_auto_tools=True,
    )
    return cls(tokenizer, chat_template_kwargs=controls)


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("special_calls", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_registered_adapters_parse_parallel_calls_across_chunks(
    thinking,
    special_calls,
    chunk_size,
):
    text = ("Plan.</think>" if thinking else "") + "Checking." + CALLS
    tokenizer, tokens = tokenizer_for(text, special_calls)
    parser = parser_for(tokenizer, {"thinking": thinking})
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[50 if thinking else 51],
        )
    )
    assert output.reasoning == ("Plan." if thinking else "")
    assert output.content.strip() == "Checking."
    assert [call["name"] for call in output.tool_calls] == ["get_weather", "add"]
    assert [json.loads(call["arguments"]) for call in output.tool_calls] == [
        {"city": "杭州", "count": 42},
        {"x": 1.5, "y": 2.25},
    ]


@pytest.mark.parametrize("thinking", [False, True])
def test_registered_adapters_parse_complete_output(thinking):
    text = ("Plan.</think>" if thinking else "") + CALLS
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": thinking})
    reasoning, content, calls = parser.parse(
        text,
        _test_request(tools=DUMMY_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert (reasoning or "") == ("Plan." if thinking else "")
    assert not (content or "").strip()
    assert [call.name for call in calls] == ["get_weather", "add"]
    assert json.loads(calls[0].arguments) == {"city": "杭州", "count": 42}


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("wrapper", ["<｜DSML｜calls>", "<｜DSML｜_calls>"])
def test_lenient_wrapper_spelling_parses_like_the_exact_one(thinking, wrapper):
    """Near the context limit the model drops the space in the outer wrapper
    (or writes an underscore). The calls must parse exactly as with the
    correct spelling and the near-miss closing tag must not leak as text."""
    end_wrapper = "</" + wrapper[1:]
    text = ("Plan.</think>" if thinking else "") + CALLS.replace(
        "<｜DSML｜ calls>", wrapper
    ).replace("</｜DSML｜ calls>", end_wrapper)
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": thinking})
    reasoning, content, calls = parser.parse(
        text,
        _test_request(tools=DUMMY_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert (reasoning or "") == ("Plan." if thinking else "")
    assert not (content or "").strip()
    assert [call.name for call in calls] == ["get_weather", "add"]
    assert json.loads(calls[0].arguments) == {"city": "杭州", "count": 42}
    assert json.loads(calls[1].arguments) == {"x": 1.5, "y": 2.25}


@pytest.mark.parametrize(
    ("controls", "text", "expected_reasoning", "reasoning_tokens"),
    [
        ({"thinking": False}, "12", "", 0),
        ({"enable_thinking": False}, "12", "", 0),
        ({"thinking": True, "reasoning_effort": "none"}, "12", "", 0),
        ({}, "</think>12", "", 0),
        ({}, "Plan.</think>12", "Plan.", 5),
    ],
)
def test_reasoning_adapter_controls_and_usage(
    controls,
    text,
    expected_reasoning,
    reasoning_tokens,
):
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, controls)
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=1,
            finished_on_last=True,
        )
    )
    assert output.reasoning == expected_reasoning
    assert output.content == "12"
    assert (
        parser.reasoning_parser.count_reasoning_tokens([tid for tid, _ in tokens])
        == reasoning_tokens
    )
    assert parser.is_reasoning_end([50, 100, 51])
    assert not parser.is_reasoning_end([51, 100, 50])


def test_python_argument_conversion_and_partial_values():
    converter = deepseek_v41_config().arg_converter
    raw = (
        '<｜DSML｜ parameter name="object" string="false">'
        '{"a": [true, null]}</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="bad" string="false">[broken</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="bare">7</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="text" string="true">  a<b'
    )
    assert json.loads(converter(raw, True)) == {
        "object": {"a": [True, None]},
        "bad": "[broken",
        "bare": 7,
        "text": "  a<b",
    }


# Hybrid Anthropic-XML / DSML tool-call idiom. In long multi-turn tool use the
# model opens a call in plain XML (``<parameter name="TOOLNAME">`` or
# ``<invoke name="…">``) and converges to DSML by the close. The opening is
# not DSML, so before the healing path the whole call, close tags and all,
# leaked into the reply as content with ``finish_reason: stop``. These
# fixtures are trimmed from real captured leaks.

_BASH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "required": ["command", "i"],
                "properties": {
                    "command": {"type": "string"},
                    "i": {"type": "string"},
                    "timeout": {"type": "number"},
                },
            },
        },
    }
]

# Variant A: the invoke is a plain ``<parameter name="bash">`` with a real
# ``<parameter name="command">`` child (Anthropic-faithful).
_HYBRID_A = (
    'Now fixing your device:\n\n'
    '<parameter name="bash">\n'
    '<parameter name="command">cd /srv && grep -rn wifi.conf . | head</parameter>\n'
    '<parameter name="i">Finding wifi.conf claims</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="timeout" string="false">60</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)
# Variant B: the invoke and its command are fused into
# ``<parameter name="bash">COMMAND</…parameter>``; the command survives as a
# bare leading value mapped to the tool's primary required parameter.
_HYBRID_B = (
    'Starting the app:\n\n'
    '<parameter name="bash">systemctl restart pdj-app; echo done</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="i" string="true">Restarting the app</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="timeout" string="false">200</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)


@pytest.mark.parametrize(
    ("text", "preamble", "expected"),
    [
        (
            _HYBRID_A,
            "Now fixing your device:",
            {
                "command": "cd /srv && grep -rn wifi.conf . | head",
                "i": "Finding wifi.conf claims",
                "timeout": 60,
            },
        ),
        (
            _HYBRID_B,
            "Starting the app:",
            {
                "command": "systemctl restart pdj-app; echo done",
                "i": "Restarting the app",
                "timeout": 200,
            },
        ),
    ],
)
def test_hybrid_xml_tool_call_is_healed(text, preamble, expected):
    """A tool call written in the plain/hybrid XML idiom is recovered rather
    than leaked into content, with its markup kept out of the reply."""
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": False})
    reasoning, content, calls = parser.parse(
        text,
        _test_request(tools=_BASH_TOOL),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert [call.name for call in calls] == ["bash"]
    assert json.loads(calls[0].arguments) == expected
    # The plan sentence stays as content; none of the markup leaks.
    assert (content or "").strip() == preamble
    for marker in ("DSML", "<parameter", "<invoke", "</invoke>", "</calls>"):
        assert marker not in (content or "")


@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_hybrid_xml_tool_call_is_healed_streaming(chunk_size):
    tokenizer, tokens = tokenizer_for(_HYBRID_A, False)
    parser = parser_for(tokenizer, {"thinking": False})
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=_BASH_TOOL,
            prompt_token_ids=[51],
        )
    )
    assert [call["name"] for call in output.tool_calls] == ["bash"]
    assert json.loads(output.tool_calls[0]["arguments"]) == {
        "command": "cd /srv && grep -rn wifi.conf . | head",
        "i": "Finding wifi.conf claims",
        "timeout": 60,
    }
    for marker in ("DSML", "<parameter", "<invoke"):
        assert marker not in output.content


def test_plain_parameter_markup_in_prose_is_not_a_tool_call():
    """A ``<parameter name="X">`` whose name is not a declared tool must stay
    as ordinary text; the recovery path is gated by the tool-name validator."""
    text = 'Use it like <parameter name="foo">bar</parameter> in your prompt.'
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": False})
    reasoning, content, calls = parser.parse(
        text,
        _test_request(tools=_BASH_TOOL),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert not calls
    assert content == text


# Headless tool calls: the model drops the invoke opener AND the tool name,
# emitting only parameter tags. The tool is inferred from the parameter names
# (a tool always covers its own parameters, so a unique cover is the right
# tool; anything ambiguous is left as text). Fixtures mirror real captured
# leaks (a `bash` run and a process-manager `proc` call).

_HEADLESS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "required": ["command", "i"],
                "properties": {
                    "command": {"type": "string"},
                    "i": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "proc",
            "parameters": {
                "type": "object",
                "required": ["op", "i"],
                "properties": {
                    "i": {"type": "string"},
                    "op": {"type": "string"},
                    "name": {"type": "string"},
                    "application": {"type": "string"},
                    "args": {"type": "array"},
                },
            },
        },
    },
]

# The command is the fused first parameter (plain open, DSML close); `command`
# belongs only to `bash`.
_HEADLESS_BASH = (
    "Now fixing your device:\n\n"
    '<parameter name="command" string="true">cd /srv && ls</parameter>\n'
    '<parameter name="i">Listing the tree</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)
# The first parameter is the shared `i`; the tool only resolves once `op` and
# `application` arrive, which name `proc` uniquely.
_HEADLESS_PROC = (
    "Starting the build:\n\n"
    '<parameter name="i">Starting the image build</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="op" string="true">start</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="name" string="true">build-x</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="application" string="true">./b.sh</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="args" string="false">["x"]</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)


@pytest.mark.parametrize(
    ("text", "preamble", "tool", "expected"),
    [
        (
            _HEADLESS_BASH,
            "Now fixing your device:",
            "bash",
            {"command": "cd /srv && ls", "i": "Listing the tree"},
        ),
        (
            _HEADLESS_PROC,
            "Starting the build:",
            "proc",
            {
                "i": "Starting the image build",
                "op": "start",
                "name": "build-x",
                "application": "./b.sh",
                "args": ["x"],
            },
        ),
    ],
)
def test_headless_tool_call_is_recovered_by_inference(text, preamble, tool, expected):
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": False})
    reasoning, content, calls = parser.parse(
        text,
        _test_request(tools=_HEADLESS_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert [call.name for call in calls] == [tool]
    assert json.loads(calls[0].arguments) == expected
    assert (content or "").strip() == preamble
    for marker in ("DSML", "<parameter", "</invoke>", "</calls>"):
        assert marker not in (content or "")


@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_headless_tool_call_is_recovered_streaming(chunk_size):
    tokenizer, tokens = tokenizer_for(_HEADLESS_PROC, False)
    parser = parser_for(tokenizer, {"thinking": False})
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=_HEADLESS_TOOLS,
            prompt_token_ids=[51],
        )
    )
    assert [c["name"] for c in output.tool_calls] == ["proc"]
    assert json.loads(output.tool_calls[0]["arguments"])["op"] == "start"
    for marker in ("DSML", "<parameter"):
        assert marker not in output.content


def test_unidentifiable_headless_block_is_restored_as_text_not_dropped():
    """A headless block whose parameters name no tool must be left as content,
    never committed as a nameless call (which would silently drop the text)."""
    text = (
        "Doing it:\n\n"
        '<parameter name="zzz" string="true">v</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="qqq" string="true">w</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": False})
    reasoning, content, calls = parser.parse(
        text,
        _test_request(tools=_HEADLESS_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert not calls
    assert "zzz" in (content or "") and "qqq" in (content or "")


def test_ambiguous_headless_block_stays_text():
    """Parameters every declared tool covers name no tool: the block is text."""
    text = (
        "Doing it:\n\n"
        '<parameter name="i">Listing</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": False})
    reasoning, content, calls = parser.parse(
        text,
        _test_request(tools=_HEADLESS_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert not calls
    assert content == text


# Text inside the thinking block stays reasoning, a DSML invoke without its
# ``calls`` wrapper or an XML-idiom call included, as upstream keeps it.

_WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
_INVOKE = (
    '<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="city" string="true">Seoul</｜DSML｜ parameter>\n'
)


def _parse(text, controls, tools=_WEATHER_TOOLS, **request):
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, controls)
    return parser.parse(
        text,
        ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            tools=tools,
            **request,
        ),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )


@pytest.mark.parametrize(
    "call",
    [
        _INVOKE + "</｜DSML｜ invoke>\n</｜DSML｜ calls>",  # complete
        _INVOKE,  # truncated before the invoke closer
        _INVOKE + "</｜DSML｜ calls>",  # wrapper closer without the invoke closer
        '<invoke name="get_weather">\n<parameter name="city">Seoul</parameter>\n'
        "</invoke>",  # XML idiom
    ],
)
@pytest.mark.parametrize("answer", ["", "</think>Answer."])
def test_call_inside_thinking_stays_reasoning(call, answer):
    reasoning, content, calls = _parse("Plan." + call + answer, {"thinking": True})
    assert not calls
    # Reasoning keeps no trailing whitespace.
    assert reasoning == ("Plan." + call).rstrip()
    assert (content or "") == answer.removeprefix("</think>")


@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_invoke_inside_thinking_stays_reasoning_streaming(chunk_size):
    text = "Plan." + _INVOKE + "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": True})
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=_WEATHER_TOOLS,
            prompt_token_ids=[50],
        )
    )
    assert output.reasoning == text
    assert output.content == ""
    assert not output.tool_calls


@pytest.mark.parametrize(
    "text",
    [
        # an undeclared tool
        '<invoke name="search">\n<parameter name="q">x</parameter>\n</invoke>',
        # a declared tool without its invoke closer
        '<invoke name="get_weather">\n<parameter name="city">Seoul</parameter>\n'
        "</｜DSML｜ calls>",
        # truncated
        '<invoke name="get_weather">\n<parameter name="city">Seoul',
    ],
)
def test_xml_idiom_candidate_that_does_not_validate_is_text(text):
    reasoning, content, calls = _parse("See " + text, {"thinking": False})
    assert not calls
    assert content == "See " + text


def test_xml_idiom_call_is_recovered():
    text = (
        'Checking.\n<invoke name="get_weather">\n'
        '<parameter name="city">Seoul</parameter>\n</invoke>'
    )
    reasoning, content, calls = _parse(text, {"thinking": False})
    assert [call.name for call in calls] == ["get_weather"]
    assert json.loads(calls[0].arguments) == {"city": "Seoul"}
    assert content.strip() == "Checking."


def test_tool_choice_none_recovers_no_call():
    text = (
        'Checking.\n<invoke name="get_weather">\n'
        '<parameter name="city">Seoul</parameter>\n</invoke>'
    )
    reasoning, content, calls = _parse(text, {"thinking": False}, tool_choice="none")
    assert not calls


def test_special_token_never_enters_a_held_candidate():
    eos = "<｜end▁of▁sentence｜>"
    tokenizer = make_mock_tokenizer({"<think>": 50, "</think>": 51, eos: 1})
    tools = _test_request(tools=_WEATHER_TOOLS).tools
    parser = DeepSeekV41Parser(
        tokenizer, tools=tools, chat_template_kwargs={"thinking": False}
    )
    text = '<invoke name="get_weather">\n<parameter name="city">Seoul'
    assert parser._engine.feed(text, []) == []
    events = parser._engine.feed(eos, [1])
    events.extend(parser._engine.finish())
    assert "".join(event.value for event in events if event.value) == text
