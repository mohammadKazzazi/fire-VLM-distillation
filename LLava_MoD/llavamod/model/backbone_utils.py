"""Path helpers for backbone selection in train entry points."""


def _lower(path: str) -> str:
    return (path or "").lower()


def is_qwen3_path(path: str) -> bool:
    p = _lower(path)
    return "qwen3" in p or "qwen-3" in p


def is_qwen2_path(path: str) -> bool:
    p = _lower(path)
    return ("qwen2" in p or "qwen-2" in p) and not is_qwen3_path(path)


def is_llama3_path(path: str) -> bool:
    p = _lower(path)
    return "llama-3" in p or "llama3" in p


def is_gemma2_path(path: str) -> bool:
    p = _lower(path)
    return "gemma-2" in p or "gemma2" in p
