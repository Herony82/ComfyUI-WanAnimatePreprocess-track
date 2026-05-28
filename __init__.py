import os

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

# Conferma caricamento nei log di ComfyUI
print(
    f"[ComfyUI-WanAnimatePreprocess-track] Caricato. "
    f"WEB_DIRECTORY: {os.path.join(os.path.dirname(__file__), 'web')}"
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
