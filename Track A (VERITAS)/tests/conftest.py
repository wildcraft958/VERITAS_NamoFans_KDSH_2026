"""
Pytest configuration and shared fixtures for KDSH VERITAS tests.
"""

import pytest
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Sample data for testing
SAMPLE_NOVEL_TEXT = """
Chapter 1: The Beginning

Robert was a young boy raised in the mountains of Patagonia. His father, a skilled 
horseman, taught him the ways of the land from an early age. The family lived in 
a small cottage near the base of the Andes, where the wind howled through the valleys.

In the spring of 1862, a great storm destroyed their home. Robert's father was killed 
in the avalanche that followed, leaving the boy orphaned at the age of twelve.

Chapter 5: The Journey

Years later, Robert had become a renowned guide, leading travelers through the 
treacherous mountain passes. His knowledge of the terrain was unmatched, and 
his bravery in the face of danger legendary.

He had learned Spanish from the gauchos who roamed the pampas, and spoke it 
with the accent of the southern plains. His horse, Thaouka, was a brown stallion 
of exceptional speed and endurance.

Chapter 10: The Conflict

The expedition faced many challenges. Earthquakes shook the ground without warning, 
causing panic among the travelers. Robert remained calm, using his knowledge of 
the land to guide them to safety.

When wolves attacked their camp, it was Robert who stood guard through the night, 
his rifle at the ready. The animals retreated at dawn, leaving no casualties.
"""

SAMPLE_BACKSTORY_CONSISTENT = """
Robert was born in a small mountain village in the Andes region. His father was 
a horseman who died when Robert was young. He grew up to become a skilled guide, 
learning the ways of the mountains and the language of the gauchos.
"""

SAMPLE_BACKSTORY_CONTRADICTORY = """
Robert was born in London to a wealthy merchant family. He never learned to ride 
horses and spent his entire life in the city, working as a banker. He never visited 
South America and spoke only English.
"""


@pytest.fixture
def sample_novel():
    """Provide sample novel text for testing."""
    logger.info("Loading sample novel fixture")
    return SAMPLE_NOVEL_TEXT


@pytest.fixture
def sample_backstory_consistent():
    """Provide a backstory that should be consistent with the novel."""
    logger.info("Loading consistent backstory fixture")
    return SAMPLE_BACKSTORY_CONSISTENT


@pytest.fixture
def sample_backstory_contradictory():
    """Provide a backstory that should contradict the novel."""
    logger.info("Loading contradictory backstory fixture")
    return SAMPLE_BACKSTORY_CONTRADICTORY


@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API calls for testing without actual API access."""
    with patch('openai.OpenAI') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock chat completions
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """["Robert was born in the mountains", "Robert's father was a horseman"]"""
        mock_instance.chat.completions.create.return_value = mock_response
        
        # Mock embeddings
        mock_embedding = MagicMock()
        mock_embedding.data = [MagicMock()]
        mock_embedding.data[0].embedding = [0.1] * 1536
        mock_instance.embeddings.create.return_value = mock_embedding
        
        logger.info("OpenAI API mock configured")
        yield mock_instance


@pytest.fixture
def test_output_dir(tmp_path):
    """Create a temporary output directory for test artifacts."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Created test output directory: {output_dir}")
    return output_dir


@pytest.fixture
def mock_llm_complete():
    """Mock the llm_complete function for isolated testing."""
    def _mock_llm(prompt, **kwargs):
        logger.info(f"Mock LLM called with prompt length: {len(prompt)}")
        # Return appropriate mock responses based on prompt content
        if "atomic facts" in prompt.lower():
            return '[{"fact": "Test fact 1", "source": "Test sentence"}]'
        elif "classify" in prompt.lower():
            return '{"status": "SUPPORTING", "confidence": 0.9}'
        else:
            return "Mock LLM response"
    
    with patch('core.models.llm_complete', side_effect=_mock_llm):
        yield _mock_llm


@pytest.fixture
def mock_embedding():
    """Mock embedding function."""
    def _mock_embed(text):
        logger.info(f"Mock embedding called for text length: {len(text)}")
        return [0.1] * 1536
    
    with patch('core.models.get_embedding', side_effect=_mock_embed):
        yield _mock_embed
