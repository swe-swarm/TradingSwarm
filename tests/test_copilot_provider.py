"""Native SDK tool handoff preserves LangGraph's trusted date injection."""

import asyncio
import inspect
import json
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.copilot_client import ChatCopilot
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.model_catalog import get_model_options
from tradingagents.llm_clients.validators import validate_model

pytestmark = pytest.mark.unit


@pytest.fixture
def sdk():
    return pytest.importorskip("copilot")


def event(kind, data):
    from copilot.session_events import SessionEvent, SessionEventType

    return SessionEvent(
        data=data, id="event", timestamp=datetime.now(timezone.utc), type=SessionEventType(kind)
    )


def assistant(content="report", *, name=None, args=None):
    from copilot.session_events import AssistantMessageData, AssistantMessageToolRequest

    requests = None
    if name:
        requests = [AssistantMessageToolRequest(name=name, arguments=args, tool_call_id="call-1")]
    return event(
        "assistant.message",
        AssistantMessageData(content=content, message_id="message-1", tool_requests=requests),
    )


@pytest.fixture
def runtime(monkeypatch, sdk):
    from copilot.session_events import SessionIdleData

    client_signature = inspect.signature(sdk.CopilotClient)
    session_signature = inspect.signature(sdk.CopilotClient.create_session)
    send_signature = inspect.signature(sdk.CopilotSession.send)
    clients = []
    controls = {
        "events": [assistant(), event("session.idle", SessionIdleData())],
        "create_error": None,
        "start_error": None,
        "create_hang": False,
        "start_hang": False,
        "send_error": None,
        "hang": False,
    }

    class Session:
        session_id = "session-1"

        def __init__(self):
            self.abort = AsyncMock()
            self.disconnect = AsyncMock()
            self.unsubscribed = False

        def on(self, callback):
            self.callback = callback

            def unsubscribe():
                self.unsubscribed = True
            return unsubscribe

        async def send(self, *args, **kwargs):
            send_signature.bind(self, *args, **kwargs)
            self.prompt = args[0]
            if controls["send_error"]:
                raise controls["send_error"]
            for item in controls["events"]:
                self.callback(item)
            if controls["hang"]:
                await asyncio.Event().wait()
            return "message-1"

    class Client:
        def __init__(self, **kwargs):
            client_signature.bind(**kwargs)
            self.options = kwargs
            self.session = Session()
            async def start():
                if controls["start_error"]:
                    raise controls["start_error"]
                if controls["start_hang"]:
                    await asyncio.Event().wait()
            self.start = AsyncMock(side_effect=start)
            self.stop = AsyncMock()
            self.force_stop = AsyncMock()
            self.delete_session = AsyncMock()
            clients.append(self)

        async def create_session(self, **kwargs):
            session_signature.bind(self, **kwargs)
            self.config = kwargs
            if controls["create_error"]:
                raise controls["create_error"]
            if controls["create_hang"]:
                await asyncio.Event().wait()
            return self.session

    monkeypatch.setattr(sdk, "CopilotClient", Client)
    return controls, clients


@pytest.mark.parametrize("provider", ["copilot", "github", "GitHub"])
def test_factory_catalog_and_github_auth(provider, sdk):
    client = create_llm_client(provider, "account-specific-model")
    assert isinstance(client.get_llm(), ChatCopilot)
    assert client.validate_model()
    assert validate_model(provider, "account-specific-model")
    assert get_api_key_env(provider) is None
    assert get_model_options(provider, "quick") == [("Custom model ID", "custom")]


def test_optional_sdk_error_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "copilot", None)
    client = create_llm_client("copilot", "model")
    with pytest.raises(ImportError, match="pip install.*github-copilot-sdk"):
        client.get_llm()


@pytest.mark.parametrize("kwargs", [
    {"base_url": "https://api.openai.com/v1"},
    {"api_key": "not-a-key"},
    {"temperature": 0.5},
    {"max_tokens": 100},
    {"max_retries": 2},
])
def test_unsupported_provider_options_are_not_silently_ignored(kwargs, sdk):
    with pytest.raises((ValueError, ValidationError)):
        create_llm_client("copilot", "model", **kwargs).get_llm()


def test_sync_lifecycle_and_safe_native_session(runtime, sdk):
    _, clients = runtime
    model = ChatCopilot(model="model")
    assert model.invoke("hello").content == "report"
    assert model.invoke("other request").content == "report"
    assert len(clients) == 2
    client = clients[0]
    assert client.options == {"use_logged_in_user": True}
    config = client.config
    assert config["tools"] == []
    assert config["available_tools"].to_list() == []
    assert config["excluded_tools"].to_list() == ["builtin:*", "mcp:*"]
    assert config["system_message"]["mode"] == "replace"
    assert config["on_permission_request"](None, None).kind == "reject"
    for key in (
        "enable_config_discovery", "enable_file_hooks", "enable_host_git_operations",
        "enable_skills", "enable_session_store", "manage_schedule_enabled",
        "enable_on_demand_instruction_discovery",
    ):
        assert config[key] is False
    assert config["infinite_sessions"] == {"enabled": False}
    client.session.abort.assert_not_awaited()
    client.session.disconnect.assert_awaited_once()
    client.delete_session.assert_awaited_once_with("session-1")
    client.stop.assert_awaited_once()
    assert client.session.unsubscribed


def test_all_message_roles_tool_ids_and_results_are_preserved(runtime):
    _, clients = runtime
    messages = [
        SystemMessage("Analyze as of the supplied date."),
        HumanMessage("AAPL"),
        AIMessage("", tool_calls=[{"name": "prices", "args": {"ticker": "AAPL"}, "id": "c1"}]),
        ToolMessage("actual historical prices", name="prices", tool_call_id="c1"),
        HumanMessage(content=[{"type": "text", "text": "summarize"}]),
    ]
    ChatCopilot(model="model").invoke(messages)
    client = clients[0]
    assert "Analyze as of the supplied date." in client.config["system_message"]["content"]
    history = json.loads(client.session.prompt)
    assert [item["role"] for item in history] == ["user", "assistant", "tool", "user"]
    assert history[1]["tool_calls"][0]["id"] == "c1"
    assert history[2] == {
        "role": "tool", "name": "prices", "content": "actual historical prices",
        "tool_call_id": "c1", "status": "success",
    }
    assert history[3]["content"] == "summarize"


def test_real_sdk_tools_hand_off_to_date_enforced_graph(runtime, sdk, monkeypatch):
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    from tradingagents.agents.utils import fundamental_data_tools

    class State(MessagesState):
        trade_date: str

    controls, clients = runtime
    tool = fundamental_data_tools.get_fundamentals
    controls["events"] = [assistant(
        "", name=tool.name, args={"ticker": "AAPL", "curr_date": "2099-01-01"}
    )]
    routed = Mock(return_value="real historical fundamentals")
    monkeypatch.setattr(fundamental_data_tools, "route_to_vendor", routed)
    result = ChatCopilot(model="model").bind_tools([tool]).invoke("Analyze AAPL")
    routed.assert_not_called()
    declaration = clients[0].config["tools"][0]
    assert isinstance(declaration, sdk.Tool)
    assert declaration.handler is None
    assert "trade_date" not in declaration.parameters["properties"]
    assert clients[0].config["available_tools"].to_list() == [f"custom:{tool.name}"]
    clients[0].session.abort.assert_awaited_once()
    assert result.tool_calls == [{
        "name": tool.name,
        "args": {"ticker": "AAPL", "curr_date": "2099-01-01"},
        "id": "call-1", "type": "tool_call",
    }]
    graph = StateGraph(State)
    graph.add_node("tools", ToolNode([tool]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    state = graph.compile().invoke({"messages": [result], "trade_date": "2025-01-15"})
    routed.assert_called_once_with("get_fundamentals", "AAPL", "2025-01-15")
    assert state["messages"][-1].content == "real historical fundamentals"


class Decision(BaseModel):
    """An investment decision."""

    action: str
    quantity: int


def test_structured_output_native_schema_and_local_validation(runtime, sdk):
    controls, clients = runtime
    controls["events"] = [assistant("", name="Decision", args={"action": "HOLD", "quantity": 0})]
    model = ChatCopilot(model="model").with_structured_output(Decision)
    assert model.invoke("decide") == Decision(action="HOLD", quantity=0)
    assert isinstance(clients[0].config["tools"][0], sdk.Tool)
    assert "must respond by calling" in clients[0].config["system_message"]["content"]
    controls["events"] = [assistant("", name="Decision", args={"action": "HOLD", "quantity": "bad"})]
    with pytest.raises(ValidationError):
        model.invoke("decide")
    raw = ChatCopilot(model="model").with_structured_output(Decision, include_raw=True)
    result = raw.invoke("decide")
    assert isinstance(result["raw"], AIMessage)
    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], ValidationError)


@pytest.mark.parametrize("tool_call", [False, True])
def test_actual_sdk_serialization_and_event_dispatch(monkeypatch, sdk, tool_call):
    """Exercise real SDK constructors/serialization, mocking only JSON-RPC I/O."""
    from copilot.session_events import SessionIdleData

    client = sdk.CopilotClient(use_logged_in_user=True)
    client._state = "connected"
    client.start = AsyncMock()
    payloads = []

    async def request(method, params=None, **kwargs):
        payloads.append((method, params))
        if method == "session.create":
            return {"sessionId": params["sessionId"]}
        if method == "session.send":
            session = client._sessions[params["sessionId"]]
            response = (
                assistant("", name="Decision", args={"action": "HOLD", "quantity": 0})
                if tool_call else assistant()
            )
            session._dispatch_event(response)
            if not tool_call:
                session._dispatch_event(event("session.idle", SessionIdleData()))
            return {"messageId": "message-1"}
        return {"success": True}

    transport = Mock(request=AsyncMock(side_effect=request), stop=AsyncMock())
    client._client = transport
    monkeypatch.setattr(sdk, "CopilotClient", Mock(return_value=client))
    model = ChatCopilot(model="model")
    result = (
        model.with_structured_output(Decision).invoke("decide")
        if tool_call else model.invoke("analyze")
    )
    assert result == Decision(action="HOLD", quantity=0) if tool_call else result.content == "report"
    config = next(params for method, params in payloads if method == "session.create")
    assert config["excludedTools"] == ["builtin:*", "mcp:*"]
    assert config["toolFilterPrecedence"] == "excluded"
    assert config["enableConfigDiscovery"] is False
    if tool_call:
        assert config["tools"][0]["name"] == "Decision"
        assert config["tools"][0]["skipPermission"] is True
        assert config["availableTools"] == ["custom:Decision"]
        assert any(method == "session.abort" for method, _ in payloads)
    assert not any("handlePendingToolCall" in method for method, _ in payloads)
    assert any(method == "session.detach" for method, _ in payloads)
    transport.stop.assert_awaited_once()


def test_async_and_sync_invocation_inside_running_loop(runtime):
    async def run():
        model = ChatCopilot(model="model")
        assert (await model.ainvoke("async")).content == "report"
        assert model.invoke("sync inside loop").content == "report"
    asyncio.run(run())


@pytest.mark.parametrize("failure", [
    "create", "send", "session", "empty", "timeout", "start", "start_timeout", "create_timeout",
])
def test_failure_paths_clean_up_without_masking_original(runtime, failure):
    from copilot.session_events import SessionErrorData, SessionIdleData

    controls, clients = runtime
    model = ChatCopilot(model="model", timeout=0.02)
    if failure == "create":
        controls["create_error"] = RuntimeError("create failed")
    elif failure == "send":
        controls["send_error"] = RuntimeError("send failed")
    elif failure == "session":
        controls["events"] = [event("session.error", SessionErrorData("test", "session failed"))]
    elif failure == "empty":
        controls["events"] = [event("session.idle", SessionIdleData())]
    elif failure == "timeout":
        controls["events"] = []
        controls["hang"] = True
    elif failure == "start":
        controls["start_error"] = RuntimeError("start failed")
    elif failure == "start_timeout":
        controls["start_hang"] = True
    elif failure == "create_timeout":
        controls["create_hang"] = True
    with pytest.raises((RuntimeError, TimeoutError)):
        model.invoke("analyze")
    client = clients[0]
    if failure not in ("create", "start", "start_timeout", "create_timeout"):
        client.session.abort.assert_awaited_once()
        client.session.disconnect.assert_awaited_once()
    client.stop.assert_awaited_once()


def test_caller_cancellation_aborts_and_closes_session(runtime):
    controls, clients = runtime
    controls["events"] = []

    async def run():
        task = asyncio.create_task(ChatCopilot(model="model").ainvoke("analyze"))
        while not clients or not hasattr(clients[0].session, "prompt"):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    clients[0].session.abort.assert_awaited_once()
    clients[0].session.disconnect.assert_awaited_once()
    clients[0].stop.assert_awaited_once()


@pytest.mark.parametrize("request_name,args", [
    ("bash", {"command": "echo forbidden"}),
    ("Decision", "not-json"),
    ("Decision", []),
])
def test_invalid_or_unbound_tool_requests_fail_closed(runtime, request_name, args):
    controls, clients = runtime
    controls["events"] = [assistant("", name=request_name, args=args)]
    with pytest.raises(ValueError):
        ChatCopilot(model="model").bind_tools([Decision]).invoke("decide")
    clients[0].session.abort.assert_awaited_once()
    clients[0].stop.assert_awaited_once()


def test_stop_failure_uses_force_stop_and_is_not_hidden(runtime):
    controls, clients = runtime

    async def run():
        controls["events"] = []
        task = asyncio.create_task(ChatCopilot(model="model").ainvoke("analyze"))
        while not clients or not hasattr(clients[0].session, "prompt"):
            await asyncio.sleep(0)
        clients[0].stop.side_effect = RuntimeError("stop failed")
        clients[0].session.callback(assistant())
        from copilot.session_events import SessionIdleData
        clients[0].session.callback(event("session.idle", SessionIdleData()))
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await task
    asyncio.run(run())
    clients[0].force_stop.assert_awaited_once()


@pytest.mark.parametrize("stage", ["abort", "disconnect", "delete_session", "stop"])
def test_cleanup_operations_are_bounded_and_continue_after_timeout(runtime, stage):
    controls, clients = runtime
    controls["events"] = []

    async def hang():
        await asyncio.Event().wait()

    async def run():
        model = ChatCopilot(model="model", cleanup_timeout=0.01)
        task = asyncio.create_task(model.bind_tools([Decision]).ainvoke("decide"))
        while not clients or not hasattr(clients[0].session, "prompt"):
            await asyncio.sleep(0)
        target = clients[0].session if stage in ("abort", "disconnect") else clients[0]
        if stage == "delete_session":
            async def hang_delete(_session_id):
                await hang()
            target.delete_session.side_effect = hang_delete
        else:
            getattr(target, stage).side_effect = hang
        clients[0].session.callback(
            assistant("", name="Decision", args={"action": "HOLD", "quantity": 0})
        )
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await asyncio.wait_for(task, timeout=1)
    asyncio.run(run())
    clients[0].session.disconnect.assert_awaited_once()
    clients[0].stop.assert_awaited_once()
    if stage == "stop":
        clients[0].force_stop.assert_awaited_once()


def test_unsupported_invocation_binding_and_structured_options(runtime):
    model = ChatCopilot(model="model")
    with pytest.raises(ValueError, match="stop"):
        model.invoke("hello", stop=["END"])
    with pytest.raises(ValueError, match="invocation"):
        model.invoke("hello", temperature=0)
    with pytest.raises(ValueError, match="tool options"):
        model.bind_tools([Decision], parallel_tool_calls=False)
    with pytest.raises(ValueError, match="structured output"):
        model.with_structured_output(Decision, method="json_schema")
    with pytest.raises(ValueError, match="text messages only"):
        model.invoke([HumanMessage(content=[{"type": "image_url", "image_url": "https://example.com"}])])
    with pytest.raises(ValueError, match="required structured"):
        model.with_structured_output(Decision).invoke("decide")
