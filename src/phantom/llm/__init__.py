"""Local-first conversational-model contracts and providers."""

from phantom.llm.context import (
    AgreementLevel,
    ApproximateAgeEstimate,
    EmotionEstimate,
    MultimodalPromptContext,
    PromptBuilder,
    PromptBundle,
    SafetyDirective,
)
from phantom.llm.contracts import (
    ChatMessage,
    ChatRole,
    GenerationResult,
    LLMError,
    LLMProvider,
    ProviderInfo,
    ProviderMode,
    ProviderRequestError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from phantom.llm.memory import ConversationMemory, ConversationTurn
from phantom.llm.providers import (
    DeterministicDemoProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    QwenTransformersProvider,
)

__all__ = [
    "AgreementLevel",
    "ApproximateAgeEstimate",
    "ChatMessage",
    "ChatRole",
    "ConversationMemory",
    "ConversationTurn",
    "DeterministicDemoProvider",
    "EmotionEstimate",
    "GenerationResult",
    "LLMError",
    "LLMProvider",
    "MultimodalPromptContext",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "PromptBuilder",
    "PromptBundle",
    "ProviderInfo",
    "ProviderMode",
    "ProviderRequestError",
    "ProviderResponseError",
    "ProviderUnavailableError",
    "QwenTransformersProvider",
    "SafetyDirective",
]
