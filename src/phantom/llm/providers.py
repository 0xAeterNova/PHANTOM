"""Explicit real and demo conversational-model providers.

No class in this module automatically substitutes the deterministic demo for a
failed real provider. Selection and any same-mode fallback remain an explicit
orchestrator responsibility.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from phantom.llm.contracts import (
    ChatMessage,
    ChatRole,
    GenerationResult,
    ProviderInfo,
    ProviderMode,
    ProviderRequestError,
    ProviderResponseError,
    ProviderUnavailableError,
    bound_generated_text,
    validate_messages,
)

HttpTransport = Callable[[str, Mapping[str, str], bytes, float, int], bytes]

_LIST_OR_HEADING = re.compile(r"(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)")
_WHITESPACE = re.compile(r"\s+")
_TERMINAL_PUNCTUATION = ".!?\u061f\u3002"


def _complete_local_response(text: str, *, hit_token_limit: bool) -> str:
    """Return speech-friendly model text and discard a token-capped fragment."""

    cleaned = _LIST_OR_HEADING.sub("", text)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    if not cleaned:
        raise ProviderResponseError("the local model returned no usable text")
    if hit_token_limit and cleaned[-1] not in _TERMINAL_PUNCTUATION:
        last_terminal = max(cleaned.rfind(mark) for mark in _TERMINAL_PUNCTUATION)
        cleaned = cleaned[: last_terminal + 1].rstrip() if last_terminal >= 0 else f"{cleaned}."
    return cleaned


def _validate_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("provider endpoint must be an absolute HTTP or HTTPS URL")
    if parsed.username or parsed.password:
        raise ValueError("provider credentials must not be embedded in the endpoint URL")
    return endpoint.rstrip("/")


def _validate_provider_bounds(
    max_input_characters: int,
    max_output_characters: int,
    max_response_bytes: int | None = None,
) -> None:
    if not 128 <= max_input_characters <= 200_000:
        raise ValueError("max_input_characters must be between 128 and 200000")
    if not 64 <= max_output_characters <= 50_000:
        raise ValueError("max_output_characters must be between 64 and 50000")
    if max_response_bytes is not None and not 1_024 <= max_response_bytes <= 20 * 1024 * 1024:
        raise ValueError("max_response_bytes must be between 1 KiB and 20 MiB")


def _effective_timeout(value: float | None, default: float) -> float:
    selected = default if value is None else float(value)
    if not 0.1 <= selected <= 600.0:
        raise ValueError("request timeout must be between 0.1 and 600 seconds")
    return selected


def _validate_transport_bytes(data: bytes, max_response_bytes: int) -> bytes:
    if not isinstance(data, bytes):
        raise ProviderResponseError("the LLM HTTP transport returned non-byte data")
    if len(data) > max_response_bytes:
        raise ProviderResponseError("the LLM HTTP response exceeded its configured size limit")
    return data


def _default_http_transport(
    endpoint: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout_seconds: float,
    max_response_bytes: int,
) -> bytes:
    request = Request(_validate_endpoint(endpoint), data=body, headers=dict(headers), method="POST")
    try:
        # The endpoint validator permits HTTP(S) only and rejects embedded credentials.
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
            data = bytes(response.read(max_response_bytes + 1))
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ProviderRequestError("the configured LLM HTTP provider request failed") from exc
    if len(data) > max_response_bytes:
        raise ProviderResponseError("the LLM HTTP response exceeded its configured size limit")
    return data


def _decode_json(data: bytes) -> Mapping[str, Any]:
    try:
        decoded = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderResponseError("the LLM provider returned invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ProviderResponseError("the LLM provider returned an unexpected response shape")
    return decoded


def _content_from_openai_response(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ProviderResponseError("the OpenAI-compatible provider returned no choices")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ProviderResponseError("the OpenAI-compatible provider returned no message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping) and item.get("type") in {None, "text"}
        ]
        if any(part.strip() for part in text_parts):
            return "".join(text_parts)
    raise ProviderResponseError("the OpenAI-compatible provider returned no text content")


class DeterministicDemoProvider:
    """A visibly synthetic, stateless provider for tests and demonstrations only."""

    def __init__(self) -> None:
        self._info = ProviderInfo(
            name="deterministic-demo",
            mode=ProviderMode.DEMO,
            model="fixed-supportive-template",
            synthetic=True,
        )

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        timeout_seconds: float | None = None,
    ) -> GenerationResult:
        del timeout_seconds
        checked = validate_messages(messages)
        if checked[-1].role is not ChatRole.USER:
            raise ValueError("the final prompt message must use the user role")
        return GenerationResult(
            text=(
                "[DEMO RESPONSE] I hear what you are saying. This response is generated by a "
                "deterministic demonstration template, not a real language model. Would you like "
                "to tell me more?"
            ),
            provider=self.info,
        )


class OllamaProvider:
    """Bounded client for Ollama's local ``/api/chat`` interface."""

    def __init__(
        self,
        model: str,
        *,
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        timeout_seconds: float = 60.0,
        max_tokens: int = 512,
        temperature: float = 0.3,
        max_input_characters: int = 24_000,
        max_output_characters: int = 8_000,
        max_response_bytes: int = 2 * 1024 * 1024,
        transport: HttpTransport | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Ollama model cannot be empty")
        if not 0.1 <= timeout_seconds <= 600.0:
            raise ValueError("timeout_seconds must be between 0.1 and 600")
        if not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be between 1 and 4096")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        _validate_provider_bounds(
            max_input_characters,
            max_output_characters,
            max_response_bytes,
        )
        self.model = model.strip()
        self.endpoint = _validate_endpoint(endpoint)
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_input_characters = max_input_characters
        self.max_output_characters = max_output_characters
        self.max_response_bytes = max_response_bytes
        self._transport = transport or _default_http_transport
        self._info = ProviderInfo(
            name="ollama",
            mode=ProviderMode.REAL,
            model=self.model,
            device="ollama-runtime",
            sends_user_data_off_device=urlsplit(self.endpoint).hostname
            not in {"127.0.0.1", "localhost", "::1"},
        )

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        timeout_seconds: float | None = None,
    ) -> GenerationResult:
        checked = validate_messages(messages, max_characters=self.max_input_characters)
        payload = {
            "model": self.model,
            "messages": [message.as_dict() for message in checked],
            "stream": False,
            "options": {"num_predict": self.max_tokens, "temperature": self.temperature},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        data = _validate_transport_bytes(
            self._transport(
                self.endpoint,
                {"Content-Type": "application/json", "Accept": "application/json"},
                body,
                _effective_timeout(timeout_seconds, self.timeout_seconds),
                self.max_response_bytes,
            ),
            self.max_response_bytes,
        )
        decoded = _decode_json(data)
        message = decoded.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise ProviderResponseError("the Ollama provider returned no text message")
        text, truncated = bound_generated_text(str(message["content"]), self.max_output_characters)
        return GenerationResult(text=text, provider=self.info, truncated=truncated)


class OpenAICompatibleProvider:
    """Generic bounded client for an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        model: str,
        *,
        endpoint: str,
        api_key: str | None = None,
        api_key_environment_variable: str | None = "PHANTOM_LLM_API_KEY",
        timeout_seconds: float = 60.0,
        max_tokens: int = 512,
        temperature: float = 0.3,
        max_input_characters: int = 24_000,
        max_output_characters: int = 8_000,
        max_response_bytes: int = 2 * 1024 * 1024,
        transport: HttpTransport | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("OpenAI-compatible model cannot be empty")
        if not 0.1 <= timeout_seconds <= 600.0:
            raise ValueError("timeout_seconds must be between 0.1 and 600")
        if not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be between 1 and 4096")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        _validate_provider_bounds(
            max_input_characters,
            max_output_characters,
            max_response_bytes,
        )
        if api_key_environment_variable is not None and not re_full_env_name(
            api_key_environment_variable
        ):
            raise ValueError("api key environment variable name is invalid")
        self.model = model.strip()
        self.endpoint = _validate_endpoint(endpoint)
        self._api_key = api_key
        self._api_key_environment_variable = api_key_environment_variable
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_input_characters = max_input_characters
        self.max_output_characters = max_output_characters
        self.max_response_bytes = max_response_bytes
        self._transport = transport or _default_http_transport
        self._info = ProviderInfo(
            name="openai-compatible",
            mode=ProviderMode.REAL,
            model=self.model,
            sends_user_data_off_device=urlsplit(self.endpoint).hostname
            not in {"127.0.0.1", "localhost", "::1"},
        )

    def __repr__(self) -> str:
        return f"OpenAICompatibleProvider(model={self.model!r}, endpoint={self.endpoint!r})"

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def _api_key_value(self) -> str | None:
        if self._api_key:
            return self._api_key
        if self._api_key_environment_variable:
            return os.getenv(self._api_key_environment_variable) or None
        return None

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        timeout_seconds: float | None = None,
    ) -> GenerationResult:
        checked = validate_messages(messages, max_characters=self.max_input_characters)
        payload = {
            "model": self.model,
            "messages": [message.as_dict() for message in checked],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if key := self._api_key_value():
            headers["Authorization"] = f"Bearer {key}"
        data = _validate_transport_bytes(
            self._transport(
                self.endpoint,
                headers,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                _effective_timeout(timeout_seconds, self.timeout_seconds),
                self.max_response_bytes,
            ),
            self.max_response_bytes,
        )
        text, truncated = bound_generated_text(
            _content_from_openai_response(_decode_json(data)),
            self.max_output_characters,
        )
        return GenerationResult(text=text, provider=self.info, truncated=truncated)


def re_full_env_name(value: str) -> bool:
    return bool(value) and all(
        character.isupper() or character.isdigit() or character == "_" for character in value
    )


class QwenTransformersProvider:
    """Lazy local Qwen provider using Transformers without ``trust_remote_code``."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        quantization: str = "none",
        local_files_only: bool = True,
        max_input_tokens: int = 4096,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        max_input_characters: int = 24_000,
        max_output_characters: int = 8_000,
    ) -> None:
        if not model_name_or_path.strip():
            raise ValueError("Qwen model name or path cannot be empty")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if quantization not in {"none", "dynamic-int8"}:
            raise ValueError("quantization must be none or dynamic-int8")
        if not 128 <= max_input_tokens <= 131_072:
            raise ValueError("max_input_tokens must be between 128 and 131072")
        if not 1 <= max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be between 1 and 4096")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        _validate_provider_bounds(max_input_characters, max_output_characters)
        self.model_name_or_path = model_name_or_path.strip()
        self.revision = revision
        self.requested_device = device
        self.quantization = quantization
        self.local_files_only = local_files_only
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_input_characters = max_input_characters
        self.max_output_characters = max_output_characters
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._resolved_device: str | None = None
        self._quantization_backend: str | None = None
        self._lock = threading.RLock()

    @property
    def info(self) -> ProviderInfo:
        device = self._resolved_device or self.requested_device
        if self.quantization != "none":
            device = f"{device};quantization={self.quantization}"
            if self._quantization_backend:
                device = f"{device};engine={self._quantization_backend}"
        return ProviderInfo(
            name="transformers-qwen",
            mode=ProviderMode.REAL,
            model=self.model_name_or_path,
            revision=self.revision,
            device=device,
        )

    @property
    def quantization_backend(self) -> str | None:
        """Return the verified quantizer implementation after lazy model loading."""

        return self._quantization_backend

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> str:
        if requested == "auto":
            if bool(torch.cuda.is_available()):
                return "cuda"
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is not None and bool(mps.is_available()):
                return "mps"
            return "cpu"
        if requested == "cuda" and not bool(torch.cuda.is_available()):
            raise ProviderUnavailableError("CUDA was selected but is not available")
        if requested == "mps":
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is None or not bool(mps.is_available()):
                raise ProviderUnavailableError("MPS was selected but is not available")
        return requested

    @staticmethod
    def _apply_dynamic_int8(model: Any, torch: Any) -> tuple[Any, str]:
        """Quantize CPU linear layers and fail if the request was not honored."""

        linear_type = getattr(getattr(torch, "nn", None), "Linear", None)
        if linear_type is None or not hasattr(model, "modules"):
            raise ProviderUnavailableError(
                "dynamic-int8 quantization requires a compatible PyTorch model"
            )
        linear_count = sum(1 for module in model.modules() if isinstance(module, linear_type))
        if linear_count == 0:
            raise ProviderUnavailableError(
                "dynamic-int8 quantization found no supported linear layers"
            )

        try:
            torchao = import_module("torchao.quantization")
        except (ImportError, ModuleNotFoundError):
            torchao = None

        if torchao is not None:
            quantize = getattr(torchao, "quantize_", None)
            config_type = getattr(torchao, "Int8DynamicActivationInt8WeightConfig", None)
            if not callable(quantize) or not callable(config_type):
                raise ProviderUnavailableError(
                    "dynamic-int8 quantization requires compatible torchao quantization APIs"
                )
            try:
                quantize(model, config_type(), device="cpu")
            except Exception as exc:
                raise ProviderUnavailableError(
                    "the requested torchao dynamic-int8 quantization could not be applied"
                ) from exc
            return model, "torchao"

        quantization = getattr(getattr(torch, "ao", None), "quantization", None)
        quantize_dynamic = getattr(quantization, "quantize_dynamic", None)
        qint8 = getattr(torch, "qint8", None)
        if not callable(quantize_dynamic) or qint8 is None:
            raise ProviderUnavailableError(
                "dynamic-int8 quantization requires torchao or PyTorch eager quantization"
            )
        try:
            quantized_model = quantize_dynamic(
                model,
                {linear_type},
                dtype=qint8,
                inplace=True,
            )
        except Exception as exc:
            raise ProviderUnavailableError(
                "the requested PyTorch dynamic-int8 quantization could not be applied"
            ) from exc
        if quantized_model is None:
            quantized_model = model
        remaining_linear_count = sum(
            1 for module in quantized_model.modules() if isinstance(module, linear_type)
        )
        if remaining_linear_count >= linear_count:
            raise ProviderUnavailableError(
                "PyTorch returned without applying the requested dynamic-int8 quantization"
            )
        return quantized_model, "torch-ao-eager"

    def _load(self) -> tuple[Any, Any, Any, str]:
        with self._lock:
            if (
                self._tokenizer is not None
                and self._model is not None
                and self._torch is not None
                and self._resolved_device is not None
            ):
                return self._tokenizer, self._model, self._torch, self._resolved_device
            try:
                transformers = import_module("transformers")
                torch = import_module("torch")
                resolved = self._resolve_device(torch, self.requested_device)
                if self.quantization == "dynamic-int8" and resolved != "cpu":
                    raise ProviderUnavailableError(
                        "dynamic-int8 quantization is supported only on the CPU provider"
                    )
                kwargs: dict[str, Any] = {
                    "local_files_only": self.local_files_only,
                    "trust_remote_code": False,
                }
                if self.revision:
                    kwargs["revision"] = self.revision
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self.model_name_or_path, **kwargs
                )
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path, **kwargs
                )
                model.to(resolved)
                model.eval()
                quantization_backend: str | None = None
                if self.quantization == "dynamic-int8":
                    # Qwen's checkpoint may load as bfloat16.  PyTorch's CPU
                    # dynamic-linear kernel requires float32 activations before
                    # it quantizes weights internally.
                    make_float = getattr(model, "float", None)
                    if not callable(make_float):
                        raise ProviderUnavailableError(
                            "dynamic-int8 quantization requires float32 model conversion"
                        )
                    model = make_float()
                    model, quantization_backend = self._apply_dynamic_int8(model, torch)
            except ProviderUnavailableError:
                raise
            except (ImportError, ModuleNotFoundError) as exc:
                raise ProviderUnavailableError(
                    "local Qwen inference requires compatible Transformers and PyTorch installations"
                ) from exc
            except Exception as exc:
                raise ProviderUnavailableError(
                    "the configured local Qwen model could not be loaded"
                ) from exc
            self._tokenizer = tokenizer
            self._model = model
            self._torch = torch
            self._resolved_device = resolved
            self._quantization_backend = quantization_backend
            return tokenizer, model, torch, resolved

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        timeout_seconds: float | None = None,
    ) -> GenerationResult:
        if timeout_seconds is not None:
            _effective_timeout(timeout_seconds, timeout_seconds)
        # Process-level cancellation of local generation belongs to the orchestrator.
        checked = validate_messages(messages, max_characters=self.max_input_characters)
        tokenizer, model, torch, device = self._load()
        message_payload = [message.as_dict() for message in checked]
        try:
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    message_payload,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = (
                    "\n".join(f"<{message.role.value}>\n{message.content}" for message in checked)
                    + "\n<assistant>\n"
                )
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
            )
            encoded = {
                name: tensor.to(device) if hasattr(tensor, "to") else tensor
                for name, tensor in encoded.items()
            }
            input_ids = encoded.get("input_ids")
            if input_ids is None:
                raise ProviderResponseError("the tokenizer produced no input IDs")
            input_length = int(input_ids.shape[-1])
            generation_options: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0.0,
            }
            if self.temperature > 0.0:
                generation_options["temperature"] = self.temperature
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                generation_options["pad_token_id"] = eos_token_id
            with torch.inference_mode():
                generated = model.generate(**encoded, **generation_options)
            continuation = generated[0][input_length:]
            decoded = tokenizer.decode(continuation, skip_special_tokens=True)
        except ProviderResponseError:
            raise
        except Exception as exc:
            raise ProviderRequestError("local Qwen generation failed") from exc
        hit_token_limit = int(continuation.shape[-1]) >= self.max_new_tokens
        completed = _complete_local_response(decoded, hit_token_limit=hit_token_limit)
        text, character_truncated = bound_generated_text(completed, self.max_output_characters)
        return GenerationResult(
            text=text,
            provider=self.info,
            truncated=hit_token_limit or character_truncated,
        )
