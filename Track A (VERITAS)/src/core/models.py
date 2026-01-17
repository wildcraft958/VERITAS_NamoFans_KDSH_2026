"""
LLM and Embedding model bindings.

Provides unified interfaces for OpenAI/OpenRouter models used across all layers.
"""

import asyncio
from typing import Callable

from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from kdsh.config import settings


def get_openai_client() -> OpenAI:
    """Get synchronous OpenAI client."""
    if settings.has_openrouter_key:
        return OpenAI(
            api_key=settings.openrouter.api_key,
            base_url=settings.openrouter.base_url
        )
    return OpenAI(api_key=settings.openai.api_key)


def get_async_openai_client() -> AsyncOpenAI:
    """Get async OpenAI client."""
    if settings.has_openrouter_key:
        return AsyncOpenAI(
            api_key=settings.openrouter.api_key,
            base_url=settings.openrouter.base_url
        )
    return AsyncOpenAI(api_key=settings.openai.api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def llm_complete(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096
) -> str:
    """
    Complete a prompt using the configured LLM.
    
    Args:
        prompt: User prompt
        system_prompt: System instructions
        model: Model override (defaults to configured model)
        temperature: Sampling temperature
        max_tokens: Maximum response tokens
        
    Returns:
        Generated text response
    """
    client = get_openai_client()
    model = model or settings.openai.model
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content or ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def async_llm_complete(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096
) -> str:
    """Async version of llm_complete."""
    client = get_async_openai_client()
    model = model or settings.openai.model
    
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content or ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_embedding(text: str, model: str | None = None) -> list[float]:
    """
    Get embedding vector for text.
    
    Args:
        text: Input text to embed
        model: Model override (defaults to configured embedding model)
        
    Returns:
        Embedding vector as list of floats
    """
    client = get_openai_client()
    model = model or settings.openai.embedding_model
    
    response = client.embeddings.create(
        model=model,
        input=text
    )
    
    return response.data[0].embedding


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def async_get_embedding(text: str, model: str | None = None) -> list[float]:
    """Async version of get_embedding."""
    client = get_async_openai_client()
    model = model or settings.openai.embedding_model
    
    response = await client.embeddings.create(
        model=model,
        input=text
    )
    
    return response.data[0].embedding


async def batch_embeddings(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Get embeddings for multiple texts concurrently."""
    tasks = [async_get_embedding(text, model) for text in texts]
    return await asyncio.gather(*tasks)


# Factory functions for LangGraph/external frameworks
def create_llm_function() -> Callable[[str], str]:
    """Create LLM function for frameworks like LightRAG."""
    def llm_fn(prompt: str) -> str:
        return llm_complete(prompt)
    return llm_fn


def create_embedding_function() -> Callable[[str], list[float]]:
    """Create embedding function for frameworks."""
    def embed_fn(text: str) -> list[float]:
        return get_embedding(text)
    return embed_fn
