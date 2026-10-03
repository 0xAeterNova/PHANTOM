"""Bounded, persistence-free contracts for conversational model providers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ProviderMode(str, Enum):
    REAL = "real"
    DEMO = "demo"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        clean = self.content.strip()
        if not clean:
            raise ValueError("chat message content cannot be empty")
        if len(clean) > 20_000:
            raise ValueError("individual chat messages cannot exceed 20000 characters")
        object.__setattr__(self, "content", clean)

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    name: str
    mode: ProviderMode
    model: str
    revision: str | None = None
    device: str | None = None
    sends_user_data_off_device: bool = False
    synthetic: bool = False

    def __post_init__(self) -> None:
        if self.mode is ProviderMode.REAL and self.synthetic:
            raise ValueError("a real provider cannot be marked synthetic")
        if self.mode is ProviderMode.DEMO and not self.synthetic:
            raise ValueError("a demo provider must be marked synthetic")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    provider: ProviderInfo
    truncated: bool = False

    def __post_init__(self) -> None:
        clean = self.text.strip()
        if not clean:
            raise ValueError("generated text cannot be empty")
        object.__setattr__(self, "text", clean)


class LLMError(RuntimeError):
    """Base error whose message must never include a prompt, transcript, or key."""


class ProviderUnavailableError(LLMError):
    """The configured provider or one of its required dependencies is unavailable."""


class ProviderRequestError(LLMError):
    """A bounded generation request failed."""


class ProviderResponseError(LLMError):
    """The provider returned a malformed or empty response."""


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def info(self) -> ProviderInfo: ...

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        timeout_seconds: float | None = None,
    ) -> GenerationResult: ...


def validate_messages(
    messages: Sequence[ChatMessage],
    *,
    max_messages: int = 32,
    max_characters: int = 24_000,
) -> tuple[ChatMessage, ...]:
    """Validate a bounded prompt without retaining it beyond the caller's request."""

    if not messages:
        raise ValueError("at least one chat message is required")
    if len(messages) > max_messages:
        raise ValueError(f"chat prompt exceeds the {max_messages}-message limit")
    checked = tuple(messages)
    if any(not isinstance(message, ChatMessage) for message in checked):
        raise TypeError("all prompt entries must be ChatMessage instances")
    if sum(len(message.content) for message in checked) > max_characters:
        raise ValueError(f"chat prompt exceeds the {max_characters}-character limit")
    if not any(message.role is ChatRole.USER for message in checked):
        raise ValueError("chat prompt must contain a user message")
    return checked


def bound_generated_text(text: str, max_characters: int) -> tuple[str, bool]:
    clean = text.strip()
    if not clean:
        raise ProviderResponseError("the model provider returned an empty response")
    if len(clean) <= max_characters:
        return clean, False
    marker = "\n[response truncated]"
    keep = max(1, max_characters - len(marker))
    return clean[:keep].rstrip() + marker, True
