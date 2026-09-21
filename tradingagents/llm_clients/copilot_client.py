"""GitHub-authenticated Copilot SDK adapter (no BYOK or OpenAI proxy).

The SDK is an agent runtime, not a chat-completions endpoint. Each invocation
owns a fresh client/session and sends a role-labelled JSON conversation, with
system instructions supplied separately. Native declaration-only SDK tools
return genuine model tool requests as LangChain ``AIMessage.tool_calls``.
The session is aborted before those requests execute: the graph's ToolNode,
not the SDK, executes the original tools and injects trusted ``trade_date``
state. The next invocation includes the real ToolMessages. No placeholder tool
results are sent to Copilot, and no dates are inferred from model text.

Built-in tools, MCP, workspace instructions, hooks, skills and host git
operations are disabled. Structured output uses the same native tool-request
mechanism and LangChain's schema parsers; a schema is never executed as a tool.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from .base_client import BaseLLMClient
from .validators import validate_model

logger = logging.getLogger(__name__)


def _sdk():
    try:
        import copilot
        from copilot.rpc import PermissionDecisionReject
    except ImportError as exc:
        raise ImportError(
            "GitHub Copilot requires github-copilot-sdk>=1.0.14. "
            "Install it with: pip install 'github-copilot-sdk>=1.0.14', "
            "then authenticate the Copilot CLI with your GitHub account."
        ) from exc
    return copilot, PermissionDecisionReject


def _text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    parts = []
    for part in message.content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part["text"])
        else:
            raise ValueError("Copilot currently supports text messages only.")
    return "\n".join(parts)


def _conversation(messages: list[BaseMessage]) -> tuple[str, str]:
    systems, history = [], []
    for message in messages:
        text = _text(message)
        if isinstance(message, SystemMessage):
            systems.append(text)
            continue
        roles = {"human": "user", "ai": "assistant", "tool": "tool"}
        if message.type not in roles:
            raise ValueError(f"Unsupported Copilot message type: {message.type}")
        entry: dict[str, Any] = {"role": roles[message.type], "content": text}
        if message.name:
            entry["name"] = message.name
        if isinstance(message, AIMessage) and message.tool_calls:
            entry["tool_calls"] = message.tool_calls
        if isinstance(message, ToolMessage):
            entry["tool_call_id"] = message.tool_call_id
            entry["status"] = message.status
        history.append(entry)
    systems.append(
        "Continue the conversation supplied as a JSON message array. Preserve its "
        "user, assistant and tool roles; tool messages are actual results of earlier "
        "calls. Never invent tool results. Use only the declared tools when needed."
    )
    return "\n\n".join(systems), json.dumps(history, ensure_ascii=False)


class ChatCopilot(BaseChatModel):
    """LangChain chat model backed by a per-invocation Copilot SDK session.

    GitHub authentication uses the SDK's logged-in user/environment credential
    chain. Unsupported sampling/token/retry controls fail explicitly rather than
    silently changing the requested behavior. ``timeout`` includes startup and
    inference; each cleanup operation has a separate ``cleanup_timeout``.
    """

    model_config = ConfigDict(extra="forbid")
    model: str
    timeout: float = Field(default=180.0, gt=0)
    cleanup_timeout: float = Field(default=10.0, gt=0)
    reasoning_effort: str | None = None

    @property
    def _llm_type(self) -> str:
        return "github-copilot"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        # LangChain passes this tracing metadata for with_structured_output.
        trace = kwargs.pop("ls_structured_output_format", None)
        if kwargs:
            raise ValueError(f"Unsupported Copilot tool options: {sorted(kwargs)}")
        specs = [convert_to_openai_tool(tool)["function"] for tool in tools]
        names = [spec["name"] for spec in specs]
        if len(names) != len(set(names)):
            raise ValueError("Copilot tool names must be unique.")
        if tool_choice not in (None, "auto", "none", "any", "required", *names):
            raise ValueError(f"Unsupported Copilot tool_choice: {tool_choice!r}")
        if tool_choice in ("any", "required") and not specs:
            raise ValueError("Required tool choice needs at least one tool.")
        options = {"tools": specs, "tool_choice": tool_choice}
        if trace is not None:
            options["ls_structured_output_format"] = trace
        return self.bind(**options)

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        method = kwargs.pop("method", "function_calling")
        strict = kwargs.pop("strict", None)
        if method != "function_calling" or strict is not None or kwargs:
            raise ValueError(
                "Copilot structured output supports function_calling with local "
                "schema validation only; strict/JSON-mode options are unsupported."
            )
        return super().with_structured_output(schema, include_raw=include_raw)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        def run():
            return asyncio.run(self._agenerate(messages, stop=stop, **kwargs))

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return run()
        # Some synchronous graph nodes are invoked from an async CLI.
        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(context.run, run).result()

    async def _cleanup(self, client, session, *, abort: bool) -> list[Exception]:
        errors = []

        async def attempt(operation):
            try:
                await asyncio.wait_for(operation(), self.cleanup_timeout)
                return True
            except Exception as exc:
                errors.append(exc)
                return False

        if session is not None:
            if abort:
                await attempt(session.abort)
            await attempt(session.disconnect)
            await attempt(lambda: client.delete_session(session.session_id))
        if not await attempt(client.stop):
            await attempt(client.force_stop)
        return errors

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if stop:
            raise ValueError("Copilot SDK does not support stop sequences.")
        tools = kwargs.pop("tools", [])
        choice = kwargs.pop("tool_choice", None)
        kwargs.pop("ls_structured_output_format", None)
        if kwargs:
            raise ValueError(f"Unsupported Copilot invocation options: {sorted(kwargs)}")
        if self.reasoning_effort not in (None, "low", "medium", "high", "xhigh"):
            raise ValueError("Unsupported Copilot reasoning_effort.")

        sdk, reject_permission = _sdk()
        system, prompt = _conversation(messages)
        if choice == "none":
            tools = []
        required = choice not in (None, "auto", "none")
        if required:
            target = "one of the declared tools" if choice in ("any", "required") else choice
            system += f"\nYou must respond by calling {target}, not with plain text."
        names = {tool["name"] for tool in tools}
        allowed = sdk.ToolSet()
        for name in names:
            allowed.add_custom(name)
        declarations = [
            sdk.Tool(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=tool.get("parameters", {"type": "object", "properties": {}}),
                skip_permission=True,
            )
            for tool in tools
        ]
        loop = asyncio.get_running_loop()
        completed = loop.create_future()
        last_message = None
        session = None
        unsubscribe = None
        abort = True
        failed = False

        def tool_message(requests, content=""):
            calls = []
            for request in requests:
                name = request.name
                if name not in names:
                    raise ValueError(f"Copilot requested an unbound tool: {name}")
                if required and choice not in ("any", "required", name):
                    raise ValueError(f"Copilot did not select required tool {choice!r}.")
                args = request.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                if not isinstance(args, dict) or not request.tool_call_id:
                    raise ValueError(f"Invalid Copilot tool request for {name}.")
                calls.append({"name": name, "args": args, "id": request.tool_call_id})
            return AIMessage(content=content, tool_calls=calls)

        def on_event(event):
            nonlocal last_message
            if completed.done():
                return
            kind = getattr(event.type, "value", event.type)
            data = event.data
            # Sub-agent messages must never become the graph's assistant response.
            if getattr(data, "parent_tool_call_id", None):
                return
            try:
                if kind == "assistant.message":
                    requests = data.tool_requests or []
                    last_message = (
                        tool_message(requests, data.content)
                        if requests else AIMessage(content=data.content)
                    )
                    if requests:
                        completed.set_result(last_message)
                elif kind == "session.error":
                    completed.set_exception(RuntimeError(f"Copilot session error: {data.message}"))
                elif kind == "session.idle":
                    if last_message is None:
                        raise RuntimeError("Copilot completed without an assistant response.")
                    if required and not last_message.tool_calls:
                        raise ValueError("Copilot did not return the required structured/tool response.")
                    completed.set_result(last_message)
            except Exception as exc:
                completed.set_exception(exc)

        client = sdk.CopilotClient(use_logged_in_user=True)
        try:
            async with asyncio.timeout(self.timeout):
                await client.start()
                session = await client.create_session(
                    model=self.model,
                    reasoning_effort=self.reasoning_effort,
                    system_message={"mode": "replace", "content": system},
                    tools=declarations,
                    available_tools=allowed,
                    excluded_tools=sdk.ToolSet().add_builtin("*").add_mcp("*"),
                    on_permission_request=lambda *_: reject_permission(
                        feedback="Only graph-executed financial tools are permitted."
                    ),
                    streaming=False,
                    enable_config_discovery=False,
                    skip_custom_instructions=True,
                    enable_on_demand_instruction_discovery=False,
                    enable_file_hooks=False,
                    enable_host_git_operations=False,
                    enable_skills=False,
                    enable_session_store=False,
                    enable_session_telemetry=False,
                    enable_experimental_mode=False,
                    skip_embedding_retrieval=True,
                    memory={"enabled": False},
                    manage_schedule_enabled=False,
                    coauthor_enabled=False,
                    custom_agents_local_only=True,
                    custom_agents=[],
                    plugin_directories=[],
                    included_builtin_skills=[],
                    mcp_servers={},
                    infinite_sessions={"enabled": False},
                )
                unsubscribe = session.on(on_event)
                await session.send(prompt)
                message = await completed
                abort = bool(message.tool_calls)
                return ChatResult(generations=[ChatGeneration(message=message)])
        except BaseException:
            failed = True
            raise
        finally:
            if unsubscribe:
                unsubscribe()
            # Shield teardown so caller cancellation cannot orphan the SDK process.
            cleanup = asyncio.create_task(self._cleanup(client, session, abort=abort))
            try:
                errors = await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            finally:
                if not completed.done():
                    completed.cancel()
                elif not completed.cancelled():
                    completed.exception()
            if errors:
                if failed:
                    logger.warning("Copilot cleanup encountered %d error(s).", len(errors))
                else:
                    raise RuntimeError("Copilot session cleanup failed.") from errors[0]


class CopilotClient(BaseLLMClient):
    """Factory adapter using Copilot's GitHub account, never provider API keys."""

    def get_llm(self) -> ChatCopilot:
        if self.base_url:
            raise ValueError("Copilot uses GitHub authentication; base_url/BYOK is unsupported.")
        _sdk()
        return ChatCopilot(model=self.model, **self.kwargs)

    def validate_model(self) -> bool:
        return validate_model("copilot", self.model)
