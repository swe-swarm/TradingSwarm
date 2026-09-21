"""ACP adapter contract and real NDJSON transport tests (no network service)."""

import asyncio
import json
import socket
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from acp import PROTOCOL_VERSION, RequestError, run_agent, schema as s
from copilot.generated.rpc import PermissionDecisionApproveOnce, PermissionDecisionReject
from copilot.generated.session_events import PermissionRequestRead, SessionEvent

from tradingswarm import acp


def text(value):
    return [s.TextContentBlock(type="text", text=value)]


def event(kind, **data):
    return SessionEvent.from_dict({
        "id": "00000000-0000-0000-0000-000000000001", "timestamp": "2026-01-01T00:00:00Z",
        "parentId": None, "type": kind, "data": data,
    })


@pytest.fixture
def setup(monkeypatch):
    runtime = SimpleNamespace(
        prompt=AsyncMock(return_value="answer"), abort=AsyncMock(), close=AsyncMock(),
    )
    create = AsyncMock(return_value=runtime)
    monkeypatch.setattr(acp.TradingSwarmSession, "create", create)
    auth = SimpleNamespace(
        start=AsyncMock(), stop=AsyncMock(),
        get_auth_status=AsyncMock(return_value=SimpleNamespace(isAuthenticated=True)),
    )
    monkeypatch.setattr(acp, "CopilotClient", MagicMock(return_value=auth))
    client = SimpleNamespace(session_update=AsyncMock(), request_permission=AsyncMock())
    agent = acp.TradingSwarmAgent()
    agent.on_connect(client)
    return SimpleNamespace(agent=agent, client=client, runtime=runtime, create=create, auth=auth)


async def new_session(setup):
    await setup.agent.initialize(PROTOCOL_VERSION)
    return (await setup.agent.new_session(cwd=str(acp.Path.cwd()), mcp_servers=[])).session_id


def test_initialize_truthful_capabilities(setup):
    async def check():
        response = await setup.agent.initialize(999, s.ClientCapabilities())
        assert response.protocol_version == PROTOCOL_VERSION
        assert not response.agent_capabilities.load_session
        assert not response.agent_capabilities.prompt_capabilities.image
        assert not response.agent_capabilities.prompt_capabilities.audio
        assert not response.agent_capabilities.prompt_capabilities.embedded_context
        assert not response.agent_capabilities.mcp_capabilities.http
        assert response.agent_capabilities.session_capabilities.list is None
        assert [method.id for method in response.auth_methods] == ["copilot-cli"]
        setup.auth.start.assert_not_awaited()
    asyncio.run(check())


def test_session_persists_context_mode_and_fleet_command(setup):
    async def check():
        sid = await new_session(setup)
        await setup.agent.prompt(sid, text("first"))
        await setup.agent.prompt(sid, text("continue"))
        await setup.agent.set_session_mode(sid, "fleet")
        await setup.agent.prompt(sid, text("parallel"))
        await setup.agent.set_session_mode(sid, "analysis")
        await setup.agent.prompt(sid, text("/fleet parallel once"))
        assert [call.args[0] for call in setup.runtime.prompt.await_args_list] == [
            "first", "continue", "parallel", "parallel once",
        ]
        assert [call.kwargs["fleet"] for call in setup.runtime.prompt.await_args_list] == [
            False, False, True, True,
        ]
        setup.create.assert_awaited_once()
        assert setup.create.await_args.kwargs["cwd"] == str(acp.Path.cwd())
        assert setup.agent.sessions[sid].mode == "analysis"
        await setup.agent.shutdown()
        setup.runtime.close.assert_awaited_once()
    asyncio.run(check())


@pytest.mark.parametrize("kwargs", [
    {"cwd": "."},
    {"cwd": "/nonexistent-tradingswarm-directory"},
    {"mcp_servers": [{"type": "http", "name": "unsafe", "url": "http://localhost"}]},
    {"mcp_servers": [{"name": "exec", "command": "sh", "args": [], "env": []}]},
    {"additional_directories": ["/"]},
])
def test_rejects_unsupported_session_configuration(setup, kwargs):
    async def check():
        await setup.agent.initialize(1)
        with pytest.raises(RequestError) as error:
            await setup.agent.new_session(**{"cwd": str(acp.Path.cwd()), **kwargs})
        assert error.value.code == -32602
        setup.create.assert_not_awaited()
    asyncio.run(check())


def test_authentication_uses_existing_cli_login(setup):
    async def check():
        await setup.agent.initialize(1)
        setup.auth.get_auth_status.return_value.isAuthenticated = False
        with pytest.raises(RequestError) as error:
            await setup.agent.new_session(str(acp.Path.cwd()))
        assert error.value.code == -32000
        assert "copilot login" in error.value.data["message"]
        setup.create.assert_not_awaited()
        setup.auth.stop.assert_awaited_once()
        acp.CopilotClient.assert_called_once_with(use_logged_in_user=True)
        with pytest.raises(RequestError) as error:
            await setup.agent.authenticate("token")
        assert error.value.code == -32602
    asyncio.run(check())


def test_auth_failure_cleanup_does_not_hide_auth_required(setup):
    async def check():
        await setup.agent.initialize(1)
        setup.auth.start.side_effect = RuntimeError("unavailable")
        setup.auth.stop.side_effect = RuntimeError("not started")
        with pytest.raises(RequestError) as error:
            await setup.agent.authenticate("copilot-cli")
        assert error.value.code == -32000
    asyncio.run(check())


@pytest.mark.parametrize("blocks", [
    [], text(" "), text("/fleet"), text("/unknown task"),
    [s.ImageContentBlock(type="image", data="aGVsbG8=", mime_type="image/png")],
    [*text("read"), s.ResourceContentBlock(type="resource_link", uri="file:///private", name="file")],
])
def test_rejects_unsupported_prompt_content(setup, blocks):
    async def check():
        sid = await new_session(setup)
        with pytest.raises(RequestError) as error:
            await setup.agent.prompt(sid, blocks)
        assert error.value.code == -32602
        setup.runtime.prompt.assert_not_awaited()
    asyncio.run(check())


def test_invalid_state_modes_and_unimplemented_methods(setup):
    async def check():
        with pytest.raises(RequestError):
            await setup.agent.new_session(str(acp.Path.cwd()))
        sid = await new_session(setup)
        with pytest.raises(RequestError) as error:
            await setup.agent.prompt("unknown", text("hello"))
        assert error.value.code == -32002
        with pytest.raises(RequestError) as error:
            await setup.agent.set_session_mode(sid, "unsafe")
        assert error.value.code == -32602
        for method in ("load_session", "list_sessions", "fork_session", "resume_session",
                       "close_session", "set_config_option"):
            with pytest.raises(RequestError) as error:
                await getattr(setup.agent, method)()
            assert error.value.code == -32601
    asyncio.run(check())


def test_sdk_events_stream_tools_and_subagents(setup):
    async def check():
        sid = await new_session(setup)
        setup.client.session_update.reset_mock()

        async def streamed_prompt(value, *, fleet, on_event):
            for update in [
                event("assistant.message_delta", messageId="m", deltaContent="Hello "),
                event("assistant.message_delta", messageId="m", deltaContent="world"),
                event("assistant.message", messageId="m", content="Hello world"),
                event("assistant.reasoning_delta", reasoningId="r", deltaContent="Considering"),
                event("tool.execution_start", toolCallId="t", toolName="task", arguments={"x": 1}),
                event("subagent.started", toolCallId="t", agentName="research",
                      agentDisplayName="Research", agentDescription="Research task"),
                event("tool.execution_progress", toolCallId="t", progressMessage="Working"),
                event("tool.execution_partial_result", toolCallId="t", partialOutput="Evidence"),
                event("subagent.failed", toolCallId="t", agentName="research",
                      agentDisplayName="Research", error="failed"),
                event("tool.execution_complete", toolCallId="t", success=False),
            ]:
                on_event(update)
            await asyncio.sleep(0)
            return "Hello world"

        setup.runtime.prompt.side_effect = streamed_prompt
        assert (await setup.agent.prompt(sid, text("hello"))).stop_reason == "end_turn"
        updates = [call.kwargs["update"] for call in setup.client.session_update.await_args_list]
        assert [update.session_update for update in updates] == [
            "agent_message_chunk", "agent_message_chunk", "agent_thought_chunk", "tool_call",
            "tool_call_update", "tool_call_update", "tool_call_update", "tool_call_update",
            "tool_call_update",
        ]
        assert updates[3].raw_input == {"x": 1}
        assert updates[-1].status == "failed"
        assert updates[4].title == "Research"
        assert updates[5].content[0].content.text == "Working"
        for update in updates:
            json.dumps(update.model_dump(mode="json", by_alias=True))
    asyncio.run(check())


def test_native_agent_id_identifies_child_events_not_event_chain_parent_id(setup):
    async def check():
        sid = await new_session(setup)
        setup.client.session_update.reset_mock()

        async def child_only(value, *, fleet, on_event):
            child = event("assistant.message_delta", messageId="child", deltaContent="Child evidence")
            child.agent_id = "research-agent"
            on_event(child)
            return "Parent synthesis"

        setup.runtime.prompt.side_effect = child_only
        await setup.agent.prompt(sid, text("/fleet research"))
        updates = [call.kwargs["update"] for call in setup.client.session_update.await_args_list]
        assert [update.content.text for update in updates] == ["Child evidence", "Parent synthesis"]
        assert updates[0].field_meta["agentId"] == "research-agent"

        setup.client.session_update.reset_mock()

        async def parent_message(value, *, fleet, on_event):
            parent = event("assistant.message", messageId="parent", content="Parent answer")
            parent.parent_id = acp.uuid4()
            on_event(parent)
            return "Parent answer"

        setup.runtime.prompt.side_effect = parent_message
        await setup.agent.prompt(sid, text("continue"))
        updates = [call.kwargs["update"] for call in setup.client.session_update.await_args_list]
        assert [update.content.text for update in updates] == ["Parent answer"]

    asyncio.run(check())


@pytest.mark.parametrize("outcome,approved", [
    (s.AllowedOutcome(outcome="selected", option_id="allow-once"), True),
    (s.AllowedOutcome(outcome="selected", option_id="reject-once"), False),
    (s.AllowedOutcome(outcome="selected", option_id="invented"), False),
    (s.DeniedOutcome(outcome="cancelled"), False),
    ({"outcome": {"outcome": "selected", "optionId": "allow-once"}}, False),
    (RuntimeError("disconnected"), False),
])
def test_permissions_require_explicit_valid_approval(setup, outcome, approved):
    async def check():
        sid = await new_session(setup)
        request = PermissionRequestRead(intention="Read evidence", path="/evidence", tool_call_id="t")
        if isinstance(outcome, Exception):
            setup.client.request_permission.side_effect = outcome
        else:
            setup.client.request_permission.return_value = (
                s.RequestPermissionResponse(outcome=outcome)
                if not isinstance(outcome, dict) else outcome
            )
        assert isinstance(await setup.agent._permission(sid, request, {}), PermissionDecisionReject)
        async with setup.agent.sessions[sid].lock:
            result = await setup.agent._permission(sid, request, {})
        assert isinstance(result, PermissionDecisionApproveOnce if approved else PermissionDecisionReject)
        options = setup.client.request_permission.await_args.kwargs["options"]
        assert [option.kind for option in options] == ["allow_once", "reject_once"]
        assert setup.client.request_permission.await_args.kwargs["tool_call"].raw_input["path"] == "/evidence"
    asyncio.run(check())


def test_cancel_inflight_prompt_and_pending_permission(setup):
    async def check():
        sid = await new_session(setup)
        running = asyncio.Event()
        permission_started = asyncio.Event()
        permission_result = []

        async def ask_permission(**kwargs):
            permission_started.set()
            await asyncio.Event().wait()

        async def long_prompt(value, *, fleet, on_event):
            running.set()
            permission_result.append(await setup.create.await_args.kwargs["on_permission_request"](
                PermissionRequestRead(intention="Read", path="/evidence"), {},
            ))
            await asyncio.Event().wait()

        setup.client.request_permission.side_effect = ask_permission
        setup.runtime.prompt.side_effect = long_prompt
        turn = asyncio.create_task(setup.agent.prompt(sid, text("long task")))
        await running.wait()
        await permission_started.wait()
        with pytest.raises(RequestError):
            await setup.agent.prompt(sid, text("concurrent"))
        with pytest.raises(RequestError):
            await setup.agent.set_session_mode(sid, "fleet")
        await setup.agent.cancel(sid)
        assert (await asyncio.wait_for(turn, 1)).stop_reason == "cancelled"
        setup.runtime.abort.assert_awaited_once()
        assert not setup.agent.sessions[sid].lock.locked()
        setup.runtime.prompt.side_effect = None
        assert (await setup.agent.prompt(sid, text("continue"))).stop_reason == "end_turn"
    asyncio.run(check())


def test_errors_release_lock_and_shutdown_closes_all_sessions(setup):
    async def check():
        sid = await new_session(setup)
        setup.runtime.prompt.side_effect = RuntimeError("private runtime detail")
        with pytest.raises(RequestError) as error:
            await setup.agent.prompt(sid, text("hello"))
        assert error.value.code == -32603
        assert "private" not in str(error.value.data)
        assert not setup.agent.sessions[sid].lock.locked()
        await setup.agent.shutdown()
        await setup.agent.shutdown()
        setup.runtime.close.assert_awaited_once()
        assert not setup.agent.sessions
    asyncio.run(check())


def test_shutdown_cancels_inflight_prompt(setup):
    async def check():
        sid = await new_session(setup)
        running = asyncio.Event()

        async def long_prompt(*args, **kwargs):
            running.set()
            await asyncio.Event().wait()

        setup.runtime.prompt.side_effect = long_prompt
        turn = asyncio.create_task(setup.agent.prompt(sid, text("research")))
        await running.wait()
        await setup.agent.shutdown()
        assert (await asyncio.wait_for(turn, 1)).stop_reason == "cancelled"
        setup.runtime.close.assert_awaited_once()
    asyncio.run(check())


def test_stream_delivery_error_cancels_runtime_and_releases_lock(setup):
    async def check():
        sid = await new_session(setup)
        stopped = asyncio.Event()

        async def long_prompt(value, *, fleet, on_event):
            on_event(event("assistant.message_delta", messageId="m", deltaContent="hello"))
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        setup.runtime.prompt.side_effect = long_prompt
        setup.client.session_update.side_effect = RuntimeError("disconnected")
        with pytest.raises(RequestError):
            await asyncio.wait_for(setup.agent.prompt(sid, text("research")), 1)
        assert stopped.is_set()
        assert not setup.agent.sessions[sid].lock.locked()
    asyncio.run(check())


def test_pending_permission_is_denied_on_cancellation(setup):
    async def check():
        sid = await new_session(setup)
        pending = asyncio.Event()

        async def permission(**kwargs):
            pending.set()
            await asyncio.Event().wait()

        setup.client.request_permission.side_effect = permission
        state = setup.agent.sessions[sid]
        async with state.lock:
            task = asyncio.create_task(setup.agent._permission(
                sid, PermissionRequestRead(intention="Read", path="/evidence"), {},
            ))
            await pending.wait()
            state.cancelled.set()
            assert isinstance(await asyncio.wait_for(task, 1), PermissionDecisionReject)
    asyncio.run(check())


@asynccontextmanager
async def transport(agent):
    left, right = socket.socketpair()
    reader, writer = await asyncio.open_connection(sock=left)
    agent_reader, agent_writer = await asyncio.open_connection(sock=right)
    task = asyncio.create_task(run_agent(agent, agent_writer, agent_reader))

    class Wire:
        async def send(self, message):
            writer.write((json.dumps({"jsonrpc": "2.0", **message}) + "\n").encode())
            await writer.drain()

        async def read(self):
            return json.loads(await asyncio.wait_for(reader.readline(), 2))

        async def response(self, request_id):
            while True:
                message = await self.read()
                if message.get("id") == request_id:
                    return message

    try:
        yield Wire()
    finally:
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(task, 2)
        agent_writer.close()
        await agent_writer.wait_closed()
        await agent.shutdown()


def test_official_transport_initialize_new_mode_and_invalid_request(setup):
    async def check():
        async with transport(setup.agent) as wire:
            await wire.send({"id": 1, "method": "initialize", "params": {"protocolVersion": 1}})
            response = await wire.response(1)
            assert response["result"]["protocolVersion"] == 1
            await wire.send({
                "id": 2, "method": "session/new",
                "params": {"cwd": str(acp.Path.cwd()), "mcpServers": []},
            })
            sid = (await wire.response(2))["result"]["sessionId"]
            await wire.send({
                "id": 3, "method": "session/set_mode",
                "params": {"sessionId": sid, "modeId": "fleet"},
            })
            assert "result" in await wire.response(3)
            await wire.send({
                "id": 4, "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "research"}]},
            })
            update = await wire.read()
            assert update["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
            assert (await wire.response(4))["result"]["stopReason"] == "end_turn"
            assert setup.runtime.prompt.await_args.kwargs["fleet"] is True
            await wire.send({"id": 5, "method": "session/prompt", "params": {}})
            assert (await wire.response(5))["error"]["code"] == -32602
            await wire.send({"id": 6, "method": "not/a/method", "params": {}})
            assert (await wire.response(6))["error"]["code"] == -32601
    asyncio.run(check())


def test_official_transport_cancel_during_inflight_turn(setup):
    async def check():
        running = asyncio.Event()

        async def long_prompt(*args, **kwargs):
            running.set()
            await asyncio.Event().wait()

        setup.runtime.prompt.side_effect = long_prompt
        async with transport(setup.agent) as wire:
            await wire.send({"id": 1, "method": "initialize", "params": {"protocolVersion": 1}})
            await wire.response(1)
            await wire.send({
                "id": 2, "method": "session/new",
                "params": {"cwd": str(acp.Path.cwd()), "mcpServers": []},
            })
            sid = (await wire.response(2))["result"]["sessionId"]
            await wire.send({
                "id": 3, "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "long task"}]},
            })
            await asyncio.wait_for(running.wait(), 1)
            await wire.send({"method": "session/cancel", "params": {"sessionId": sid}})
            assert (await wire.response(3))["result"]["stopReason"] == "cancelled"
            setup.runtime.abort.assert_awaited_once()
    asyncio.run(check())


def test_official_transport_permission_request_and_approval(setup):
    async def check():
        async def request_permission(value, *, fleet, on_event):
            result = await setup.create.await_args.kwargs["on_permission_request"](
                PermissionRequestRead(intention="Read evidence", path="/evidence", tool_call_id="t"),
                {},
            )
            assert isinstance(result, PermissionDecisionApproveOnce)
            return "Approved evidence"

        setup.runtime.prompt.side_effect = request_permission
        async with transport(setup.agent) as wire:
            await wire.send({"id": 1, "method": "initialize", "params": {"protocolVersion": 1}})
            await wire.response(1)
            await wire.send({
                "id": 2, "method": "session/new",
                "params": {"cwd": str(acp.Path.cwd()), "mcpServers": []},
            })
            sid = (await wire.response(2))["result"]["sessionId"]
            await wire.send({
                "id": 3, "method": "session/prompt",
                "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "research"}]},
            })
            permission = await wire.read()
            assert permission["method"] == "session/request_permission"
            assert permission["params"]["toolCall"]["toolCallId"] == "t"
            await wire.send({
                "id": permission["id"],
                "result": {"outcome": {"outcome": "selected", "optionId": "allow-once"}},
            })
            assert (await wire.response(3))["result"]["stopReason"] == "end_turn"
    asyncio.run(check())


def test_stdio_routes_runtime_prints_to_stderr_and_closes_on_failure(monkeypatch, capsys):
    async def check():
        reader, writer = object(), object()
        monkeypatch.setattr(acp, "stdio_streams", AsyncMock(return_value=(reader, writer)))
        agent = SimpleNamespace(shutdown=AsyncMock())
        monkeypatch.setattr(acp, "TradingSwarmAgent", MagicMock(return_value=agent))

        async def transport_runner(actual_agent, actual_writer, actual_reader):
            assert (actual_agent, actual_writer, actual_reader) == (agent, writer, reader)
            print("vendor diagnostic")
            raise RuntimeError("transport closed")

        monkeypatch.setattr(acp, "run_agent", transport_runner)
        with pytest.raises(RuntimeError, match="transport closed"):
            await acp.serve()
        agent.shutdown.assert_awaited_once()

    asyncio.run(check())
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "vendor diagnostic\n"
