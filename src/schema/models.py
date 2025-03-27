from enum import StrEnum, auto
from typing import TypeAlias


class Provider(StrEnum):
    OPENAI = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    GROQ = auto()
    AWS = auto()
    FAKE = auto()
    Ollama = auto()

class OllamaModelName(StrEnum):
    "These models work when you already installed on local machine"
    
    LLAMA3_2 = "llama3.2"
    LLAMA3 = "llama3"
    CODESTRAL = "codestral"
    CODEQWEN = "codeqwen"
    MISTRAL = "mistral"
    DEEPSEEK_R1_1_5B = "deepseek-r1:1.5b"
    DEEPSEEK_R1_8B = "deepseek-r1:8b"
    DEEPSEEK_CODER_V2   = "deepseek-coder-v2"

class OpenAIModelName(StrEnum):
    """https://platform.openai.com/docs/models/gpt-4o"""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"


class AnthropicModelName(StrEnum):
    """https://docs.anthropic.com/en/docs/about-claude/models#model-names"""

    HAIKU_3 = "claude-3-haiku"
    HAIKU_35 = "claude-3.5-haiku"
    SONNET_35 = "claude-3.5-sonnet"


class GoogleModelName(StrEnum):
    """https://ai.google.dev/gemini-api/docs/models/gemini"""

    GEMINI_15_FLASH = "gemini-1.5-flash"


class GroqModelName(StrEnum):
    """https://console.groq.com/docs/models"""

    LLAMA_31_8B = "groq-llama-3.1-8b"
    LLAMA_33_70B = "groq-llama-3.3-70b"

    LLAMA_GUARD_3_8B = "groq-llama-guard-3-8b"


class AWSModelName(StrEnum):
    """https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html"""

    BEDROCK_HAIKU = "bedrock-3.5-haiku"


class FakeModelName(StrEnum):
    """Fake model for testing."""

    FAKE = "fake"


AllModelEnum: TypeAlias = (
    OpenAIModelName
    | AnthropicModelName
    | GoogleModelName
    | GroqModelName
    | AWSModelName
    | FakeModelName
    | OllamaModelName
)
