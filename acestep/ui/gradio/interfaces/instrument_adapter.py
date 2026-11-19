"""Select named adapters without collisions between different final directories."""

import gradio as gr


def activate_adapter(handler, path, name):
    """Load once under a unique name, then explicitly select and enable it."""
    registry = getattr(handler, "_active_loras", {}) or {}
    if name not in registry:
        message = handler.add_lora(str(path), adapter_name=name)
        if not message.startswith("✅"):
            raise gr.Error(f"模型載入失敗：{message}")
    message = handler.set_active_lora_adapter(name)
    if not message.startswith("✅") or not handler.lora_loaded:
        raise gr.Error(f"模型選擇失敗：{message}")
    handler.set_use_lora(True)
    handler.set_lora_scale(1.0)
    if not handler.use_lora:
        raise gr.Error("微調模型未啟用，請重新套用。")
    handler.quick_adapter_path = str(path.resolve())
    handler.quick_adapter_name = name
