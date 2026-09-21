import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tradingswarm.fleet import (
    _ORCHESTRATION_TOOLS,
    TradingSwarmSession,
    _evidence,
    _trading_tool,
)

pytest.importorskip("copilot")


def make_session():
    handlers = []
    session = Mock()
    session.on.side_effect = lambda fn: handlers.append(fn) or (lambda: handlers.remove(fn))
    session.abort = AsyncMock()
    session.disconnect = AsyncMock()

    async def respond(*args, **kwargs):
        for handler in handlers:
            handler(SimpleNamespace(
                type="assistant.message", agent_id=None,
                data=SimpleNamespace(content="Combined research"),
            ))
            handler(SimpleNamespace(
                type="assistant.message", agent_id="child",
                data=SimpleNamespace(content="Worker report"),
            ))
        return SimpleNamespace(started=True)

    session.send_and_wait = AsyncMock(side_effect=respond)
    session.rpc.fleet.start = AsyncMock(side_effect=respond)
    client = SimpleNamespace(stop=AsyncMock())
    return TradingSwarmSession(client, session), handlers


def test_native_fleet_rpc_and_parent_response():
    wrapper, handlers = make_session()
    assert asyncio.run(wrapper.prompt("/fleet Analyze AAPL on 2026-01-02")) == "Combined research"
    request = wrapper.session.rpc.fleet.start.call_args.args[0]
    assert request.prompt == "Analyze AAPL on 2026-01-02"
    assert request.wait is True
    wrapper.session.send_and_wait.assert_not_called()
    assert handlers == []


def test_normal_prompt_uses_persistent_session():
    async def run():
        wrapper, _ = make_session()
        await wrapper.prompt("AAPL on 2026-01-02")
        await wrapper.prompt("Explain the uncertainty")
        assert wrapper.session.send_and_wait.call_count == 2
        wrapper.session.rpc.fleet.start.assert_not_called()
    asyncio.run(run())


@pytest.mark.parametrize("failure", [TimeoutError(), RuntimeError("RPC failed")])
def test_failure_aborts_and_unsubscribes(failure):
    wrapper, handlers = make_session()
    wrapper.session.rpc.fleet.start.side_effect = failure
    with pytest.raises(type(failure)):
        asyncio.run(wrapper.prompt("AAPL", fleet=True))
    wrapper.session.abort.assert_awaited_once()
    assert not handlers


def test_declined_start_is_not_success():
    wrapper, _ = make_session()
    wrapper.session.rpc.fleet.start.side_effect = None
    wrapper.session.rpc.fleet.start.return_value = SimpleNamespace(started=False)
    with pytest.raises(RuntimeError, match="did not start"):
        asyncio.run(wrapper.prompt("AAPL", fleet=True))


def test_cancel_aborts_runtime():
    async def run():
        wrapper, handlers = make_session()
        started = asyncio.Event()

        async def wait(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        wrapper.session.rpc.fleet.start.side_effect = wait
        task = asyncio.create_task(wrapper.prompt("AAPL", fleet=True))
        await started.wait()
        with pytest.raises(RuntimeError, match="already running"):
            await wrapper.prompt("second")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        wrapper.session.abort.assert_awaited_once()
        assert not handlers
    asyncio.run(run())


def test_close_stops_client_even_if_disconnect_fails():
    wrapper, _ = make_session()
    wrapper.session.disconnect.side_effect = RuntimeError("disconnected")
    with pytest.raises(RuntimeError):
        asyncio.run(wrapper.close())
    wrapper.client.stop.assert_awaited_once()
    asyncio.run(wrapper.close())
    wrapper.client.stop.assert_awaited_once()


def test_stop_failure_forces_runtime_shutdown():
    wrapper, _ = make_session()
    wrapper.client.stop.side_effect = RuntimeError("transport closed")
    wrapper.client.force_stop = AsyncMock()
    asyncio.run(wrapper.close())
    wrapper.client.force_stop.assert_awaited_once()


@pytest.mark.parametrize("prompt", ["", " ", "/fleet", "/fleet\n"])
def test_empty_prompts_do_not_start_work(prompt):
    wrapper, _ = make_session()
    with pytest.raises(ValueError):
        asyncio.run(wrapper.prompt(prompt))
    wrapper.session.send_and_wait.assert_not_called()
    wrapper.session.rpc.fleet.start.assert_not_called()


def test_multiline_fleet_command():
    wrapper, _ = make_session()
    asyncio.run(wrapper.prompt("/fleet\nAnalyze NVDA"))
    assert wrapper.session.rpc.fleet.start.call_args.args[0].prompt == "Analyze NVDA"


def test_create_restricts_tools_and_configuration(tmp_path):
    async def run():
        client = SimpleNamespace(create_session=AsyncMock(), stop=AsyncMock())
        with patch("copilot.CopilotClient", return_value=client):
            await TradingSwarmSession.create(cwd=str(tmp_path))
        kwargs = client.create_session.call_args.kwargs
        assert kwargs["available_tools"].to_list() == [
            *[f"builtin:{name}" for name in _ORCHESTRATION_TOOLS], "custom:trading_evidence",
        ]
        assert not any(tool.startswith("mcp:") for tool in kwargs["available_tools"])
        assert kwargs["enable_config_discovery"] is False
        assert kwargs["enable_file_hooks"] is False
        assert "general-purpose" in kwargs["excluded_builtin_agents"]
        assert "provider" not in kwargs
        assert len(kwargs["custom_agents"]) == 8
        assert kwargs["on_permission_request"](None, {}).kind == "reject"
    asyncio.run(run())


def test_create_failure_stops_client(tmp_path):
    client = SimpleNamespace(create_session=AsyncMock(side_effect=RuntimeError()), stop=AsyncMock())
    with patch("copilot.CopilotClient", return_value=client), pytest.raises(RuntimeError):
        asyncio.run(TradingSwarmSession.create(cwd=str(tmp_path)))
    client.stop.assert_awaited_once()


def test_reject_relative_cwd_and_invalid_mcp():
    with pytest.raises(ValueError, match="absolute"):
        asyncio.run(TradingSwarmSession.create(cwd="relative"))
    with pytest.raises(ValueError, match="MCP stdio command"):
        asyncio.run(TradingSwarmSession.create(mcp_servers={"unknown": {}}))


def test_explicit_stdio_mcp_configuration(tmp_path):
    async def run():
        client = SimpleNamespace(create_session=AsyncMock(), stop=AsyncMock())
        server = {"command": "/opt/market-mcp", "args": ["--stdio"], "env": {"MODE": "research"}}
        with patch("copilot.CopilotClient", return_value=client):
            await TradingSwarmSession.create(cwd=str(tmp_path), mcp_servers={"market": server})
        kwargs = client.create_session.call_args.kwargs
        assert "mcp:*" in kwargs["available_tools"].to_list()
        assert kwargs["mcp_servers"]["market"] == {
            **server, "type": "stdio", "tools": ["*"], "working_directory": str(tmp_path),
        }
    asyncio.run(run())


def test_dated_evidence_preserves_ticker_and_cutoff():
    with patch("tradingagents.agents.utils.agent_utils.get_news") as tool:
        _evidence("news", "SHOP.TO", "2026-01-02")
    args = tool.invoke.call_args.args[0]
    assert args["ticker"] == "SHOP.TO"
    assert args["trade_date"] == args["end_date"] == "2026-01-02"


def test_tool_failure_is_explicit():
    invocation = SimpleNamespace(arguments={"topic": "news", "ticker": "AAPL", "as_of": "invalid"})
    result = asyncio.run(_trading_tool({"as_of": "invalid"}).handler(invocation))
    assert result.result_type == "failure"
    assert "Evidence unavailable" in result.text_result_for_llm


def test_user_cutoff_is_retained_for_followups():
    async def run():
        wrapper, _ = make_session()
        await wrapper.prompt("Analyze AAPL as of 2026-01-02 using news since 2025-12-01")
        await wrapper.prompt("Explain the risks")
        assert wrapper._scope["as_of"] == "2026-01-02"
    asyncio.run(run())


@pytest.mark.parametrize("scope", [{}, {"as_of": "2026-01-02"}])
def test_model_cannot_choose_an_unapproved_cutoff(scope):
    invocation = SimpleNamespace(
        arguments={"topic": "news", "ticker": "AAPL", "as_of": "2026-02-01"},
    )
    with patch("tradingswarm.fleet._evidence") as evidence:
        result = asyncio.run(_trading_tool(scope).handler(invocation))
    assert result.result_type == "failure"
    evidence.assert_not_called()


def test_ambiguous_dates_require_explicit_cutoff():
    wrapper, _ = make_session()
    with pytest.raises(ValueError, match="one analysis cutoff"):
        asyncio.run(wrapper.prompt("Compare 2025-01-01 and 2026-01-01"))
