import json
from collections import deque

import litellm
from arclet.entari import MessageCreatedEvent, Session, filter_
from arclet.entari.config import config_model_validate
from arclet.entari.event.config import ConfigReload
from arclet.entari.event.send import SendResponse
from arclet.letoderea import BLOCK, ExitState, on
from arclet.letoderea.typing import Contexts, generate_contexts

from ..config import Config, _conf
from ..events.tools import LLMToolEvent, available_functions, tools
from ..log import logger
from ..service import llm

RECORD = deque(maxlen=16)


@on(SendResponse)
async def _record(event: SendResponse):
    if event.result and event.session:
        RECORD.append(event.session.event.sn)


@on(MessageCreatedEvent, priority=1000).if_(filter_.to_me)
async def run_conversation(session: Session, ctx: Contexts):
    if session.event.sn in RECORD:
        return BLOCK
    msg = session.elements.extract_plain_text()
    messages: list = [{"role": "user", "content": msg, "name": session.user.name}]
    final_answer = ""
    for step in range(8):
        response = await llm.generate(
            messages,
            stream=False,
            tools=tools,
            tool_choice="auto",
        )

        response_message = response["choices"][0]["message"]
        tool_calls = response_message.tool_calls
        messages.append(
            {
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
            }
        )

        if tool_calls:
            calls = [tc for tc in tool_calls if isinstance(tc, litellm.ChatCompletionMessageToolCall)]
            for tool_call in calls:
                function_name = tool_call.function.name

                if function_name is None:
                    continue

                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                ctx1 = await generate_contexts(LLMToolEvent(), inherit_ctx=ctx)
                logger.info(f"Calling tool: {function_name} with args: {function_args}")
                try:
                    resp = await function_to_call.handle(ctx1 | function_args, inner=True)
                    if isinstance(resp, ExitState):
                        if resp.args[0] is not None:
                            result = {"ok": True, "data": resp.args[0]}
                        else:
                            result = {"ok": False, "error": "No response"}
                    else:
                        result = {"ok": True, "data": resp}
                except Exception as e:
                    result = {"ok": False, "error": repr(e)}
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            continue
        final_answer = response_message.content or ""
        break
    if final_answer:
        await session.send(final_answer)
    else:
        await session.send("对话失败，请稍后再试")
    return BLOCK


@on(ConfigReload)
async def reload_config(event: ConfigReload):
    if event.scope != "plugin":
        return
    if event.key not in ("entari_plugin_llm", "llm"):
        return
    new_conf = config_model_validate(Config, event.value)
    _conf.models = new_conf.models
    _conf.prompt = new_conf.prompt
