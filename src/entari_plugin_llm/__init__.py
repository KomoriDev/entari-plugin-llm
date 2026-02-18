from arclet.entari import declare_static, metadata

from .config import Config
from .events import LLMToolEvent as LLMToolEvent
from .log import _suppress_litellm_logging

metadata(
    name="entari-plugin-llm",
    author=[{"name": "RF-Tar-Railt", "email": "rf_tar_railt@qq.com"}],
    version="0.1.0",
    description="An Entari Plugin for LLM Chat with Function Call",
    config=Config,
)
declare_static()
_suppress_litellm_logging()

from . import listeners as listeners
from .service import llm as llm
