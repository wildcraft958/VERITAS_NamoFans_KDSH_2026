"""
Configuration management for the Narrative Auditor pipeline.

Loads settings from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    """OpenAI API configuration."""
    
    model_config = SettingsConfigDict(env_prefix="OPENAI_")
    
    api_key: str = Field(default="", description="OpenAI API key")
    model: str = Field(default="gpt-4o", description="Model for reasoning tasks")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")


class OpenRouterSettings(BaseSettings):
    """OpenRouter API configuration (alternative to OpenAI)."""
    
    model_config = SettingsConfigDict(env_prefix="OPENROUTER_")
    
    api_key: str = Field(default="", description="OpenRouter API key")
    base_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter base URL")


class RetrieverSettings(BaseSettings):
    """RAPTOR and HippoRAG configuration."""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    raptor_chunk_size: int = Field(default=512, alias="RAPTOR_CHUNK_SIZE")
    raptor_top_k: int = Field(default=10, alias="RAPTOR_TOP_K")
    hipporag_top_k: int = Field(default=5, alias="HIPPORAG_TOP_K")


class AdjudicationSettings(BaseSettings):
    """Conservative adjudication thresholds."""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    # Rule 1: Hard contradiction threshold
    contradiction_threshold: float = Field(
        default=0.90, 
        alias="CONTRADICTION_THRESHOLD",
        description="Evidence weight above this triggers automatic rejection"
    )
    
    # Rule 3: Persona alignment threshold
    persona_alignment_threshold: float = Field(
        default=0.60,
        alias="PERSONA_ALIGNMENT_THRESHOLD", 
        description="Simulator alignment below this triggers rejection"
    )
    
    # Rule 4: Support count threshold
    support_count_threshold: int = Field(
        default=3,
        alias="SUPPORT_COUNT_THRESHOLD",
        description="Minimum supporting evidence count for acceptance"
    )


class PathwaySettings(BaseSettings):
    """Pathway server configuration."""
    
    model_config = SettingsConfigDict(env_prefix="PATHWAY_")
    
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8080, description="Server port")


class Settings(BaseSettings):
    """Main settings container."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Nested settings
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    retriever: RetrieverSettings = Field(default_factory=RetrieverSettings)
    adjudication: AdjudicationSettings = Field(default_factory=AdjudicationSettings)
    pathway: PathwaySettings = Field(default_factory=PathwaySettings)
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    
    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai.api_key)
    
    @property
    def has_openrouter_key(self) -> bool:
        """Check if OpenRouter API key is configured."""
        return bool(self.openrouter.api_key)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience export
settings = get_settings()
