"""TradingSwarm's text-only ACP adapter, using the official stdio transport."""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import redirect_stdout, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from acp import PROTOCOL_VERSION, Agent, RequestError, run_agent, schema as s
from acp.stdio import stdio_streams
from copilot import CopilotClient
from copilot.generated.rpc import PermissionDecisionApproveOnce, PermissionDecisionReject

from tradingswarm.fleet import TradingSwarmSession

log = logging.getLogger(__name__)


def _value(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _wire(obj: Any) -> Any:
    return obj.to_dict() if hasattr(obj, "to_dict") else obj


def _invalid(message: str) -> RequestError:
    return RequestError.invalid_params({"message": message})


@dataclass
class _Session:
    runtime: Any
    mode: str = "analysis"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    sender: asyncio.Task | None = None
    cancelling: bool = False


class TradingSwarmAgent(Agent):
    """One persistent Copilot session per ACP session; no client filesystem access."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model
        self.connection: Any = None
        self.sessions: dict[str, _Session] = {}
        self.initialized = False
        self.closing = False

    def on_connect(self, conn: Any) -> None:
        self.connection = conn

    async def initialize(
        self, protocol_version: int, client_capabilities: Any = None,
        client_info: Any = None, **kwargs: Any,
    ) -> s.InitializeResponse:
        self.initialized = True
        return s.InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_info=s.Implementation(name="tradingswarm", version="0.1.0"),
            agent_capabilities=s.AgentCapabilities(),
            auth_methods=[s.AuthMethodAgent(
                id="copilot-cli", name="Existing Copilot CLI login",
                description="Run `copilot login` in a terminal, then retry. No tokens are collected.",
            )],
        )

    def _ready(self) -> None:
        if not self.initialized or self.closing:
            raise RequestError.invalid_request({"message": "Agent is not initialized or is closing"})

    def _session(self, session_id: str) -> _Session:
        self._ready()
        if session_id not in self.sessions:
            raise RequestError.resource_not_found(session_id)
        return self.sessions[session_id]

    async def authenticate(self, method_id: str, **kwargs: Any) -> s.AuthenticateResponse:
        self._ready()
        if method_id != "copilot-cli":
            raise _invalid("Unsupported authentication method")
        client = CopilotClient(use_logged_in_user=True)
        try:
            await client.start()
            status = await client.get_auth_status()
            if not status.isAuthenticated:
                raise RequestError.auth_required({"message": "Run `copilot login`, then retry"})
        except RequestError:
            raise
        except Exception as exc:
            log.warning("Unable to verify Copilot CLI authentication: %s", type(exc).__name__)
            raise RequestError.auth_required(
                {"message": "Copilot CLI login is unavailable; run `copilot login`, then retry"}
            ) from exc
        finally:
            try:
                await client.stop()
            except Exception as exc:
                log.warning("Copilot auth cleanup failed: %s", type(exc).__name__)
        return s.AuthenticateResponse()

    async def new_session(
        self, cwd: str, mcp_servers: list | None = None,
        additional_directories: list[str] | None = None, **kwargs: Any,
    ) -> s.NewSessionResponse:
        self._ready()
        if not Path(cwd).is_absolute() or not Path(cwd).is_dir():
            raise _invalid("cwd must be an existing absolute directory")
        if mcp_servers:
            raise _invalid("Client-supplied MCP servers are unsupported; use the built-in trading tools")
        if additional_directories:
            raise _invalid("Additional directories are unsupported")
        await self.authenticate("copilot-cli")
        session_id = str(uuid4())

        async def permission(request: Any, context: Any) -> Any:
            return await self._permission(session_id, request, context)

        try:
            runtime = await TradingSwarmSession.create(
                model=self.model, cwd=cwd, on_permission_request=permission,
            )
        except Exception as exc:
            log.warning("Copilot session creation failed: %s", type(exc).__name__)
            raise RequestError.internal_error({"message": "Unable to create Copilot session"}) from exc
        if self.closing:
            await runtime.close()
            raise RequestError.invalid_request({"message": "Agent is closing"})
        self.sessions[session_id] = _Session(runtime)
        try:
            await self._update(session_id, s.AvailableCommandsUpdate(
                session_update="available_commands_update",
                available_commands=[s.AvailableCommand(
                    name="fleet", description="Run this prompt using the native Copilot fleet",
                    input=s.AvailableCommandInput(
                        root=s.UnstructuredCommandInput(hint="Task for the fleet"),
                    ),
                )],
            ))
        except BaseException:
            self.sessions.pop(session_id, None)
            await runtime.close()
            raise
        return s.NewSessionResponse(
            session_id=session_id,
            modes=s.SessionModeState(
                current_mode_id="analysis",
                available_modes=[
                    s.SessionMode(id="analysis", name="Analysis"),
                    s.SessionMode(id="fleet", name="Fleet"),
                ],
            ),
        )

    async def set_session_mode(
        self, session_id: str, mode_id: str, **kwargs: Any,
    ) -> s.SetSessionModeResponse:
        state = self._session(session_id)
        if mode_id not in ("analysis", "fleet"):
            raise _invalid("modeId must be analysis or fleet")
        if state.lock.locked() or state.cancelling:
            raise RequestError.invalid_request({"message": "Cannot change mode during a prompt"})
        state.mode = mode_id
        await self._update(session_id, s.CurrentModeUpdate(
            session_update="current_mode_update", current_mode_id=mode_id,
        ))
        return s.SetSessionModeResponse()

    async def _update(self, session_id: str, update: Any) -> None:
        if self.connection is None:
            raise RequestError.internal_error({"message": "ACP client is not connected"})
        await self.connection.session_update(session_id=session_id, update=update)

    async def _permission(self, session_id: str, request: Any, context: Any) -> Any:
        denied = PermissionDecisionReject()
        state = self.sessions.get(session_id)
        if (
            state is None or not state.lock.locked() or state.cancelled.is_set()
            or self.closing or self.connection is None
        ):
            return denied
        turn_cancelled = state.cancelled
        approval = asyncio.create_task(self.connection.request_permission(
            session_id=session_id,
            tool_call=s.ToolCallUpdate(
                tool_call_id=_value(request, "tool_call_id") or str(uuid4()),
                title=_value(request, "intention") or _value(request, "tool_name")
                or f"Copilot {_value(request, 'kind', 'tool')} permission",
                status="pending", raw_input=_wire(request),
            ),
            options=[
                s.PermissionOption(option_id="allow-once", name="Allow once", kind="allow_once"),
                s.PermissionOption(option_id="reject-once", name="Reject", kind="reject_once"),
            ],
        ))
        cancelled = asyncio.create_task(turn_cancelled.wait())
        try:
            done, _ = await asyncio.wait(
                [approval, cancelled], timeout=60, return_when=asyncio.FIRST_COMPLETED,
            )
            if approval in done and not turn_cancelled.is_set() and not self.closing:
                response = approval.result()
                if (
                    isinstance(response, s.RequestPermissionResponse)
                    and isinstance(response.outcome, s.AllowedOutcome)
                    and response.outcome.option_id == "allow-once"
                ):
                    return PermissionDecisionApproveOnce(approved_interactively=True)
        except Exception as exc:
            log.warning("ACP permission request denied: %s", type(exc).__name__)
        finally:
            for task in (approval, cancelled):
                task.cancel()
            await asyncio.gather(approval, cancelled, return_exceptions=True)
        return denied

    async def prompt(
        self, session_id: str, prompt: list, **kwargs: Any,
    ) -> s.PromptResponse:
        state = self._session(session_id)
        if not prompt or any(not isinstance(block, s.TextContentBlock) for block in prompt):
            raise _invalid("Only nonempty text prompts are supported")
        text = "\n".join(block.text for block in prompt).strip()
        fleet = state.mode == "fleet"
        if text.startswith("/"):
            command, *arguments = text.split(maxsplit=1)
            if command != "/fleet":
                raise _invalid("Unsupported slash command; use /fleet")
            fleet = True
            text = arguments[0].strip() if arguments else ""
        if not text:
            raise _invalid("Prompt text must not be empty")
        if state.lock.locked() or state.cancelling:
            raise RequestError.invalid_request({"message": "A prompt is already running"})
        async with state.lock:
            state.cancelled = asyncio.Event()
            queue: asyncio.Queue = asyncio.Queue()
            streamed: set[tuple[str | None, str]] = set()
            started: set[str] = set()
            emitted = False
            accepting = True

            def on_event(event: Any) -> None:
                nonlocal emitted
                if not accepting or state.cancelled.is_set():
                    return
                kind = _value(event, "type")
                kind = _value(kind, "value", kind)
                agent_id = _value(event, "agent_id")
                data = _value(event, "data")
                update = None
                if kind in ("assistant.message_delta", "assistant.message"):
                    message_id = _value(data, "message_id", "")
                    message_key = (agent_id, message_id)
                    if kind.endswith("_delta"):
                        content = _value(data, "delta_content", "")
                        if content:
                            streamed.add(message_key)
                    else:
                        content = "" if message_key in streamed else _value(data, "content", "")
                    if content:
                        if not agent_id:
                            emitted = True
                        update = s.AgentMessageChunk(
                            session_update="agent_message_chunk",
                            content=s.TextContentBlock(type="text", text=content),
                            field_meta={"parentToolCallId": _value(data, "parent_tool_call_id")},
                        )
                elif kind == "assistant.reasoning_delta":
                    update = s.AgentThoughtChunk(
                        session_update="agent_thought_chunk",
                        content=s.TextContentBlock(type="text", text=_value(data, "delta_content", "")),
                    )
                elif kind in ("tool.execution_start", "subagent.started"):
                    tool_id = _value(data, "tool_call_id")
                    title = _value(data, "tool_name") or _value(data, "agent_display_name", "Subagent")
                    if tool_id in started:
                        update = s.ToolCallProgress(
                            session_update="tool_call_update", tool_call_id=tool_id,
                            title=title, status="in_progress",
                        )
                    else:
                        started.add(tool_id)
                        update = s.ToolCallStart(
                            session_update="tool_call", tool_call_id=tool_id, title=title,
                            kind="other", status="in_progress", raw_input=_value(data, "arguments"),
                            field_meta={"parentToolCallId": _value(data, "parent_tool_call_id")},
                        )
                elif kind in ("tool.execution_complete", "subagent.completed", "subagent.failed"):
                    failed = kind == "subagent.failed" or (
                        kind == "tool.execution_complete" and not _value(data, "success", False)
                    )
                    update = s.ToolCallProgress(
                        session_update="tool_call_update", tool_call_id=_value(data, "tool_call_id"),
                        status="failed" if failed else "completed",
                        raw_output=_wire(_value(data, "error") if failed else _value(data, "result")),
                    )
                elif kind in ("tool.execution_progress", "tool.execution_partial_result"):
                    content = _value(data, "progress_message") or _value(data, "partial_output", "")
                    update = s.ToolCallProgress(
                        session_update="tool_call_update", tool_call_id=_value(data, "tool_call_id"),
                        status="in_progress",
                        content=[s.ContentToolCallContent(
                            type="content", content=s.TextContentBlock(type="text", text=content),
                        )],
                    )
                if update is not None:
                    if agent_id:
                        update.field_meta = {**(update.field_meta or {}), "agentId": agent_id}
                    queue.put_nowait(update)

            async def send_updates() -> None:
                while (update := await queue.get()) is not None:
                    if not state.cancelled.is_set():
                        await self._update(session_id, update)

            sender = asyncio.create_task(send_updates())
            state.sender = sender
            state.task = asyncio.create_task(state.runtime.prompt(
                text, fleet=fleet, on_event=on_event,
            ))
            try:
                done, _ = await asyncio.wait(
                    [state.task, sender], return_when=asyncio.FIRST_COMPLETED,
                )
                if sender in done:
                    await sender
                result = await state.task
                accepting = False
                if not emitted and result and not state.cancelled.is_set():
                    queue.put_nowait(s.AgentMessageChunk(
                        session_update="agent_message_chunk",
                        content=s.TextContentBlock(type="text", text=result),
                    ))
                queue.put_nowait(None)
                await sender
                return s.PromptResponse(
                    stop_reason="cancelled" if state.cancelled.is_set() else "end_turn",
                )
            except asyncio.CancelledError:
                if state.cancelled.is_set():
                    return s.PromptResponse(stop_reason="cancelled")
                raise
            except Exception as exc:
                log.warning("Copilot prompt failed: %s", type(exc).__name__)
                raise RequestError.internal_error({"message": "Copilot prompt failed"}) from exc
            finally:
                accepting = False
                state.task.cancel()
                sender.cancel()
                await asyncio.gather(state.task, sender, return_exceptions=True)
                state.task = None
                state.sender = None

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        state = self._session(session_id)
        if not state.lock.locked():
            return
        state.cancelled.set()
        state.cancelling = True
        if state.task is not None:
            state.task.cancel()
        if state.sender is not None:
            state.sender.cancel()
        try:
            await asyncio.wait_for(state.runtime.abort(), timeout=5)
        except Exception as exc:
            log.warning("Copilot abort failed: %s", type(exc).__name__)
        finally:
            state.cancelling = False

    async def shutdown(self) -> None:
        if self.closing:
            return
        self.closing = True
        states = list(self.sessions.values())
        for state in states:
            state.cancelled.set()
            if state.task is not None:
                state.task.cancel()
            if state.sender is not None:
                state.sender.cancel()
        await asyncio.gather(*(state.runtime.close() for state in states), return_exceptions=True)
        self.sessions.clear()

    async def load_session(self, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("session/load")

    async def list_sessions(self, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("session/list")

    async def fork_session(self, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("session/fork")

    async def resume_session(self, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("session/resume")

    async def close_session(self, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("session/close")

    async def set_config_option(self, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("session/set_config_option")

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        log.warning("Unsupported ACP notification: %s", method)


async def serve(model: str | None = None) -> None:
    agent = TradingSwarmAgent(model=model)
    reader, writer = await stdio_streams()
    # Capture the protocol's stdout pipe first; vendor tool prints are diagnostics.
    with redirect_stdout(sys.stderr):
        try:
            await run_agent(agent, writer, reader)
        finally:
            await agent.shutdown()


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    with suppress(KeyboardInterrupt):
        asyncio.run(serve())


if __name__ == "__main__":
    main()
