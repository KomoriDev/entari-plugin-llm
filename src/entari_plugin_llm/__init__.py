import inspect
import json
from collections import deque
from typing import Annotated, Any, TypeAlias, get_args

import openai
from arclet.entari import (
    BasicConfModel,
    Entari,
    MessageCreatedEvent,
    Session,
    declare_static,
    filter_,
    metadata,
    plugin_config,
)
from arclet.entari.config import config_model_validate
from arclet.entari.event.config import ConfigReload
from arclet.entari.event.send import SendResponse
from arclet.entari.logger import log
from arclet.letoderea import BLOCK, ExitState, Subscriber, define, on
from arclet.letoderea.provider import get_providers
from arclet.letoderea.typing import Contexts, Result, generate_contexts
from docstring_parser import parse
from openai import omit
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
from tarina import Empty
from tarina.generic import get_origin, origin_is_union
from typing_extensions import Doc


class Config(BasicConfModel):
    api_key: str
    """API key for authentication with the OpenAI API"""
    base_url: str = "https://api.openai.com/v1"
    """Base URL for the OpenAI API"""
    model: str = "gpt-4"
    """Model to use for the OpenAI API"""
    prompt: str = ""
    """Default prompt template"""
    context_length: int = 50
    """Number of messages to keep in context"""


metadata(
    name="entari-plugin-llm",
    author=[{"name": "RF-Tar-Railt", "email": "rf_tar_railt@qq.com"}],
    version="0.1.0",
    description="An Entari Plugin for LLM Chat with Function Call",
    config=Config,
)
declare_static()


_conf = plugin_config(Config)

JSON_VALUE: TypeAlias = str | int | float | bool | None
JSON_TYPE: TypeAlias = dict[str, "JSON_TYPE"] | list["JSON_TYPE"] | JSON_VALUE


class LLMToolEvent:
    __publisher__ = "tools_pub"

    def check_result(self, value: Any) -> Result[JSON_TYPE] | None:
        if isinstance(value, (str, int, float, bool, type(None), list, dict)):
            return Result(value)


tools_pub = define(LLMToolEvent, name="tools_pub")
tools_pub.bind(*get_providers(MessageCreatedEvent))


tools = []
available_functions: dict[str, Subscriber[JSON_TYPE]] = {}
mapping = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    set: "array",
    tuple: "array",
    dict: "object",
}
logger = log.wrapper("[llm]")


@tools_pub.check
def _register_tool(_, sub: Subscriber):
    properties = {}
    required = []
    doc = inspect.cleandoc(sub.__doc__ or "")

    parsed = parse(doc)
    param_docs = {p.arg_name: p.description or "" for p in parsed.params}

    for param in sub.params:
        if param.providers:  # skip provided parameters
            continue
        if param.default is Empty:
            required.append(param.name)
        anno = param.annotation
        orig = get_origin(anno)
        if origin_is_union(orig) and type(None) in get_args(anno):  # pragma: no cover
            t = get_args(anno)[0]
        else:
            t = anno
        documentation = param_docs.get(param.name, "")
        if get_origin(t) is Annotated:  # pragma: no cover
            t, *meta = get_args(t)
            if doc := next((i for i in meta if isinstance(i, Doc)), None):
                documentation = doc.documentation
        properties[param.name] = {
            "title": param.name.title(),
            "type": mapping.get(get_origin(t), "object"),
            "description": documentation,
        }

    tools.append(
        {
            "type": "function",
            "function": {
                "name": sub.__name__,
                "description": parsed.description or doc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
    )
    available_functions[sub.__name__] = sub
    sub._attach_disposes(lambda s: available_functions.pop(s.__name__, None))  # type: ignore
    return True


client = openai.AsyncClient(api_key=_conf.api_key, base_url=_conf.base_url)

RECORD = deque(maxlen=16)


@on(SendResponse)
async def _record(event: SendResponse):
    if event.result and event.session:
        RECORD.append(event.session.event.sn)


@on(MessageCreatedEvent, priority=3)
async def record_message(session: Session, app: Entari):
    key = f"llm_message_record:{session.account.platform}_{session.channel.id}"
    if await app.cache.has(key):
        messages: deque[dict] = await app.cache.get(key)
    else:
        messages = deque(maxlen=_conf.context_length)
        await app.cache.set(key, messages, None)
    msg = session.content
    messages.append(
        {
            "role": "user",
            "content": f"[{session.user.name}@{session.user.id}] {msg}",
            "name": f"{session.user.name}@{session.user.id}",
        }
    )


@on(MessageCreatedEvent, priority=1000).if_(filter_.to_me)
async def run_conversation(session: Session, ctx: Contexts, app: Entari):
    if session.event.sn in RECORD:
        return BLOCK
    key = f"llm_message_record:{session.account.platform}_{session.channel.id}"
    messages: deque = await app.cache.get(key)
    if _conf.prompt:
        messages.insert(0, {"role": "system", "content": _conf.prompt})
    final_answer = ""
    for step in range(8):
        response = await client.chat.completions.create(
            model=_conf.model,
            messages=list(messages),
            tools=tools if tools else omit,
            tool_choice="auto",
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        messages.append(
            {
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
            }
        )

        if tool_calls:
            calls = [tc for tc in tool_calls if isinstance(tc, ChatCompletionMessageFunctionToolCall)]
            for tool_call in calls:
                function_name = tool_call.function.name
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
        if tool_calls:
            messages.append({"role": "assistant", "content": final_answer})
        break
    if final_answer:
        await session.send(final_answer)
    else:
        await session.send("对话失败，请稍后再试")
    messages.popleft()
    return BLOCK


@on(ConfigReload)
async def reload_config(event: ConfigReload):
    global client

    if event.scope != "plugin":
        return
    if event.key not in ("entari_plugin_llm", "llm"):
        return
    new_conf = config_model_validate(Config, event.value)
    _conf.model = new_conf.model
    _conf.prompt = new_conf.prompt
    _conf.context_length = new_conf.context_length
    await client.close()
    client = openai.AsyncClient(api_key=new_conf.api_key, base_url=new_conf.base_url)
