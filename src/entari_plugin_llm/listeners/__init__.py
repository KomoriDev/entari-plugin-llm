from collections import deque

from arclet.entari import MessageCreatedEvent, Session, filter_
from arclet.entari.config import config_model_validate
from arclet.entari.event.config import ConfigReload
from arclet.entari.event.lifespan import Ready
from arclet.entari.event.send import SendResponse
from arclet.letoderea import BLOCK, on
from arclet.letoderea.typing import Contexts

from .._handler import LLMSessionManager
from .._jsondata import get_default_model, set_default_model
from ..config import Config, _conf
from ..log import logger

RECORD = deque(maxlen=16)


@on(Ready)
async def _():
    if not _conf.models:
        set_default_model(None)
        logger.warning("未配置任何模型，已清空本地默认模型配置")
        return

    first_model = _conf.models[0].name
    default_model = get_default_model()
    if not default_model:
        set_default_model(first_model)
        logger.info(f"未检测到本地默认模型，已设置为首个模型: {first_model}")
        return

    matched = next(
        (
            m
            for m in _conf.models
            if m.name == default_model or m.alias == default_model
        ),
        None,
    )
    if matched is None:
        set_default_model(first_model)
        logger.warning(
            f"本地默认模型不存在于当前配置: {default_model}，已重置为: {first_model}",
        )
        return

    if matched.name != default_model:
        set_default_model(matched.name)
        logger.info(f"已将本地默认模型标准化为模型名: {matched.name}")


@on(SendResponse)
async def _record(event: SendResponse):
    if event.result and event.session:
        RECORD.append(event.session.event.sn)


@on(MessageCreatedEvent, priority=1000).if_(filter_.to_me)
async def run_conversation(session: Session, ctx: Contexts):
    if session.event.sn in RECORD:
        return BLOCK

    msg = session.elements.extract_plain_text()
    answer = await LLMSessionManager.chat(user_input=msg, ctx=ctx, session=session)
    await session.send(answer)
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
