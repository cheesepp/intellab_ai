from functools import cache
import os
from typing import TypeAlias

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_community.chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OpenAIModelName,
    OllamaModelName
)

_MODEL_TABLE = {
    OllamaModelName.LLAMA3_2: "llama3.2",
    OllamaModelName.LLAMA3: "llama3",
    OllamaModelName.CODESTRAL: "codestral",
    OllamaModelName.CODEQWEN: "codeqwen",
    OllamaModelName.MISTRAL: "mistral",
    OllamaModelName.DEEPSEEK_R1_1_5B: "deepseek-r1:1.5b",
    OllamaModelName.DEEPSEEK_R1_8B: "deepseek-r1:8b",
    OllamaModelName.DEEPSEEK_CODER_V2: "deepseek-coder-v2",
    OpenAIModelName.GPT_4O_MINI: "gpt-4o-mini",
    OpenAIModelName.GPT_4O: "gpt-4o",
    AnthropicModelName.HAIKU_3: "claude-3-haiku-20240307",
    AnthropicModelName.HAIKU_35: "claude-3-5-haiku-latest",
    AnthropicModelName.SONNET_35: "claude-3-5-sonnet-latest",
    GoogleModelName.GEMINI_15_FLASH: "gemini-1.5-flash",
    GroqModelName.LLAMA_31_8B: "llama-3.1-8b-instant",
    GroqModelName.LLAMA_33_70B: "llama-3.3-70b-versatile",
    GroqModelName.LLAMA_GUARD_3_8B: "llama-guard-3-8b",
    AWSModelName.BEDROCK_HAIKU: "anthropic.claude-3-5-haiku-20241022-v1:0",
    FakeModelName.FAKE: "fake",
}

ModelT: TypeAlias = ChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI | ChatGroq | ChatBedrock


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OllamaModelName:
        return ChatOllama(model=model_name, streaming=True, base_url=os.getenv("OLLAMA_HOST"))
    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in AnthropicModelName:
        return ChatAnthropic(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GoogleModelName:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GroqModelName:
        if model_name == GroqModelName.LLAMA_GUARD_3_8B:
            return ChatGroq(model=api_model_name, temperature=0.0)
        return ChatGroq(model=api_model_name, temperature=0.5)
    if model_name in AWSModelName:
        return ChatBedrock(model_id=api_model_name, temperature=0.5)
    if model_name in FakeModelName:
        return FakeListChatModel(responses=["This is a test response from the fake model."])
