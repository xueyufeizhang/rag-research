"""Explicit model completion metadata shared by generation and extraction."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    text: str
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    truncated: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("LLM response text must be a string")
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise TypeError("LLM finish_reason must be a string or None")
        if self.usage is not None and not isinstance(self.usage, dict):
            raise TypeError("LLM usage must be an object or None")
        if self.truncated is not None and not isinstance(self.truncated, bool):
            raise TypeError("LLM truncated must be a boolean or None")


def normalize_llm_response(value: str | LLMResponse) -> LLMResponse:
    """Legacy strings have unknown completion status, even when valid JSON."""
    if isinstance(value, LLMResponse):
        return value
    if isinstance(value, str):
        return LLMResponse(text=value)
    raise TypeError("LLM callable must return str or LLMResponse")


def truncation_from_finish_reason(reason: str | None) -> bool | None:
    """Use only explicit provider signals; other reasons remain unknown."""
    if reason == "length":
        return True
    if reason == "stop":
        return False
    return None
