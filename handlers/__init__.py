from .start import register_start_handler
from .text import register_text_handler


def register_handlers(dp):
    register_start_handler(dp)
    register_text_handler(dp)
