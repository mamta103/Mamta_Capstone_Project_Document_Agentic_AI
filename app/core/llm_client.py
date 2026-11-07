import json
from dataclasses import dataclass, field

import groq

from app.core.config import settings

TEMPERATURE = 0.1
TIMEOUT_SECONDS = 30
MAX_RETRIES = 2
# gpt-oss models count hidden reasoning tokens against this limit.
MAX_COMPLETION_TOKENS = 1024


class LLMError(Exception):
    """Raised with a user-safe message when the LLM call fails."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ToolTurn:
    """One LLM turn in a tool-calling conversation: plain text, tool calls, or both."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_message(self) -> dict:
        """The assistant message to send back to the LLM along with the tool results."""
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in self.tool_calls
            ],
        }


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, client=None):
        self.model = model or settings.llm_model
        if client is None:
            key = settings.groq_api_key if api_key is None else api_key
            if not key:
                raise LLMError("GROQ_API_KEY is not set. Add it to your .env file.")
            client = groq.Groq(api_key=key, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
        self._client = client

    def complete(self, system: str, user: str) -> str:
        return self._text(self._send(self._messages(system, user)))

    def complete_json(self, system: str, user: str, model: str | None = None) -> dict:
        text = self._text(self._send(self._messages(system, user), model=model, response_format={"type": "json_object"}))
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError("The LLM returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise LLMError("The LLM returned invalid JSON.")
        return data

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> ToolTurn:
        message = self._send(messages, tools=tools, tool_choice="auto")
        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise LLMError("The LLM returned invalid tool arguments.") from exc
            if not isinstance(arguments, dict):
                raise LLMError("The LLM returned invalid tool arguments.")
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
        return ToolTurn(content=message.content or "", tool_calls=calls)

    @staticmethod
    def _messages(system: str, user: str) -> list[dict]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _send(self, messages: list[dict], model: str | None = None, **options):
        model = model or self.model
        extra = dict(options)
        if model.startswith("openai/gpt-oss"):
            extra.update(reasoning_effort="low", include_reasoning=False)
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                **extra,
            )
        except groq.AuthenticationError as exc:
            raise LLMError("The LLM API key is invalid or expired.") from exc
        except (groq.PermissionDeniedError, groq.NotFoundError) as exc:
            raise LLMError(f"The LLM model '{model}' is not available for this API key.") from exc
        except groq.RateLimitError as exc:
            raise LLMError("LLM rate limit reached. Please wait a moment and try again.") from exc
        except groq.APITimeoutError as exc:
            raise LLMError("The LLM service timed out. Please try again.") from exc
        except groq.APIConnectionError as exc:
            raise LLMError("Could not reach the LLM service. Check your network connection.") from exc
        except groq.APIError as exc:
            raise LLMError("The LLM service returned an error. Please try again.") from exc

        return response.choices[0].message

    @staticmethod
    def _text(message) -> str:
        content = message.content
        if not content or not content.strip():
            raise LLMError("The LLM returned an empty response.")
        return content.strip()
