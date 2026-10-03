"""Thread-safe, bounded, in-memory conversation history."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from phantom.llm.contracts import ChatMessage, ChatRole

_TRUNCATION_MARKER = "…[truncated]"


def _truncate(value: str, limit: int) -> str:
    clean = value.strip()
    if not clean:
        raise ValueError("conversation messages cannot be empty")
    if len(clean) <= limit:
        return clean
    if limit <= len(_TRUNCATION_MARKER):
        return clean[:limit]
    return clean[: limit - len(_TRUNCATION_MARKER)].rstrip() + _TRUNCATION_MARKER


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    user: str
    assistant: str

    @property
    def character_count(self) -> int:
        return len(self.user) + len(self.assistant)

    def as_messages(self) -> tuple[ChatMessage, ChatMessage]:
        return (
            ChatMessage(ChatRole.USER, self.user),
            ChatMessage(ChatRole.ASSISTANT, self.assistant),
        )


class ConversationMemory:
    """Keep only recent dialogue in process memory; no serialization API is provided."""

    def __init__(
        self,
        *,
        max_turns: int = 8,
        max_characters: int = 12_000,
        max_message_characters: int = 4_000,
    ) -> None:
        if not 1 <= max_turns <= 100:
            raise ValueError("max_turns must be between 1 and 100")
        if not 128 <= max_characters <= 200_000:
            raise ValueError("max_characters must be between 128 and 200000")
        if not 32 <= max_message_characters <= max_characters:
            raise ValueError("max_message_characters must be between 32 and max_characters")
        self.max_turns = max_turns
        self.max_characters = max_characters
        self.max_message_characters = max_message_characters
        self._turns: deque[ConversationTurn] = deque()
        self._character_count = 0
        self._lock = threading.RLock()

    def _fit_new_turn(self, user: str, assistant: str) -> ConversationTurn:
        user_text = _truncate(user, self.max_message_characters)
        assistant_text = _truncate(assistant, self.max_message_characters)
        if len(user_text) + len(assistant_text) <= self.max_characters:
            return ConversationTurn(user_text, assistant_text)

        user_budget = max(32, self.max_characters // 2)
        assistant_budget = max(32, self.max_characters - user_budget)
        user_text = _truncate(user_text, user_budget)
        assistant_text = _truncate(assistant_text, assistant_budget)
        return ConversationTurn(user_text, assistant_text)

    def append_turn(self, user: str, assistant: str) -> ConversationTurn:
        turn = self._fit_new_turn(user, assistant)
        with self._lock:
            self._turns.append(turn)
            self._character_count += turn.character_count
            while len(self._turns) > self.max_turns or (
                self._character_count > self.max_characters and len(self._turns) > 1
            ):
                removed = self._turns.popleft()
                self._character_count -= removed.character_count
        return turn

    def messages(self) -> tuple[ChatMessage, ...]:
        with self._lock:
            return tuple(message for turn in self._turns for message in turn.as_messages())

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
            self._character_count = 0

    @property
    def turn_count(self) -> int:
        with self._lock:
            return len(self._turns)

    @property
    def character_count(self) -> int:
        with self._lock:
            return self._character_count
