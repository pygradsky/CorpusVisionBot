from src.handlers.cmds.start_cmd import router as start_cmd_router
from src.handlers.text.text_handler import router as text_handler_router
from src.handlers.media.media_handler import router as media_handler_router

__all_routers__ = [
    start_cmd_router,
    text_handler_router,
    media_handler_router,
]
