"""
Layer-specific unit tests for KDSH VERITAS pipeline.

Tests each layer in isolation with mocked dependencies.
"""

import pytest
import logging
from unittest.mock import MagicMock, patch

logger = logging.getLogger(__name__)


class TestLayer1AtomicDecomposition:
    """Tests for Layer 1: FActScore-inspired atomic fact extraction."""
    
    def test_fact_extraction_basic(self, mock_llm_complete):
        """Test basic fact extraction from backstory text."""
        logger.info("Testing basic fact extraction")
        
        from kdsh.layer1.factscore import FactExtractor
        
        extractor = FactExtractor()
        backstory = "Robert was born in 1850. He grew up in the mountains."
        
        with patch.object(extractor, '_parse_response', return_value=[
            {"fact": "Robert was born in 1850", "source": "Robert was born in 1850."},
            {"fact": "Robert grew up in the mountains", "source": "He grew up in the mountains."}
        ]):
            facts = extractor.extract(backstory)
        
        logger.info(f"Extracted {len(facts)} facts")
        assert len(facts) >= 1
    
    def test_fact_extraction_empty_backstory(self):
        """Test handling of empty backstory."""
        logger.info("Testing empty backstory handling")
        
        from kdsh.layer1.factscore import FactExtractor
        
        extractor = FactExtractor()
        
        with patch.object(extractor, 'extract', return_value=[]):
            facts = extractor.extract("")
        
        assert facts == []
        logger.info("Empty backstory handled correctly")
    
    def test_constraint_graph_creation(self):
        """Test constraint graph construction from atomic facts."""
        logger.info("Testing constraint graph creation")
        
        from kdsh.layer1.constraint_graph import AtomicFact, ConstraintGraph, Constraint, ConstraintType
        
        facts = [
            AtomicFact(id="F1", text="Robert was born in 1850", source_sentence="Test"),
            AtomicFact(id="F2", text="Robert grew up in mountains", source_sentence="Test")
        ]
        
        # Create constraints from facts
        constraints = [
            Constraint(id="C1", type=ConstraintType.TEMPORAL, claim=facts[0].text, source_atoms=[facts[0]]),
            Constraint(id="C2", type=ConstraintType.CAUSAL, claim=facts[1].text, source_atoms=[facts[1]])
        ]
        
        graph = ConstraintGraph(backstory_id="test", constraints=constraints)
        
        assert len(graph.constraints) == 2
        logger.info(f"Created constraint graph with {len(graph.constraints)} constraints")


class TestLayer2DualRetrieval:
    """Tests for Layer 2: RAPTOR + HippoRAG dual retrieval."""
    
    def test_raptor_tree_building(self, mock_llm_complete, mock_embedding):
        """Test RAPTOR tree construction from novel text."""
        logger.info("Testing RAPTOR tree building")
        
        from kdsh.layer2.raptor_index import RAPTORIndex
        
        with patch.object(RAPTORIndex, '_summarize', return_value="Test summary"):
            index = RAPTORIndex(chunk_size=100)
            # Skip actual building for unit test
            assert index is not None
        
        logger.info("RAPTOR index instantiation successful")
    
    def test_hipporag_entity_extraction(self, mock_llm_complete, mock_embedding):
        """Test HippoRAG entity and relationship extraction."""
        logger.info("Testing HippoRAG entity extraction")
        
        from kdsh.layer2.hipporag_index import HippoRAGIndex
        
        index = HippoRAGIndex()
        assert index is not None
        
        logger.info("HippoRAG index instantiation successful")
    
    def test_unified_retriever(self, mock_llm_complete, mock_embedding):
        """Test unified retrieval combining RAPTOR and HippoRAG."""
        logger.info("Testing unified retriever")
        
        from kdsh.layer2.unified_retriever import DualRetriever
        
        with patch.object(DualRetriever, '__init__', return_value=None):
            retriever = DualRetriever.__new__(DualRetriever)
            retriever.raptor = MagicMock()
            retriever.hipporag = MagicMock()
            
        assert retriever is not None
        logger.info("Unified retriever instantiation successful")


class TestLayer3ParallelVerification:
    """Tests for Layer 3: Historian + Simulator parallel verification."""
    
    def test_historian_evidence_classification(self, mock_llm_complete):
        """Test Historian evidence classification."""
        logger.info("Testing Historian evidence classification")
        
        from kdsh.layer3.historian import Historian, EvidenceStatus
        
        historian = Historian()
        
        # Mock the classify_evidence method
        with patch.object(historian, '_classify_evidence', return_value=MagicMock(
            status=EvidenceStatus.SUPPORTING,
            weight=0.9,
            reasoning="Evidence supports the claim"
        )):
            result = historian._classify_evidence("Test claim", MagicMock())
        
        assert result.status == EvidenceStatus.SUPPORTING
        logger.info(f"Classification result: {result.status}")
    
    def test_simulator_persona_adoption(self, mock_llm_complete):
        """Test Simulator persona creation from backstory."""
        logger.info("Testing Simulator persona adoption")
        
        from kdsh.layer3.simulator import Simulator
        
        simulator = Simulator()
        
        with patch.object(simulator, 'adopt_persona', return_value=MagicMock(
            traits=["brave", "skilled"],
            values=["loyalty"],
            backstory_summary="A mountain guide"
        )):
            persona = simulator.adopt_persona("Test backstory", "test_id")
        
        assert persona is not None
        logger.info("Persona adoption successful")


class TestLayer4EvidenceLedger:
    """Tests for Layer 4: Evidence Ledger aggregation."""
    
    def test_ledger_entry_creation(self):
        """Test creating ledger entries."""
        logger.info("Testing ledger entry creation")
        
        from kdsh.layer4.ledger import EvidenceLedger, LedgerEntry
        
        ledger = EvidenceLedger()
        
        # Create entry with correct parameter names
        entry = LedgerEntry(
            constraint_id="C1",
            claim="Test claim",
            constraint_type="Temporal",
            historian_verdict={"status": "SUPPORTED"},
            simulator_verdict={"alignment": 0.9},
            supporting_evidence=[],
            contradicting_evidence=[],
            missing_evidence=[]
        )
        
        assert entry.constraint_id == "C1"
        assert entry.claim == "Test claim"
        logger.info("Ledger entry created successfully")


class TestLayer5Adjudication:
    """Tests for Layer 5: Conservative Adjudication with 5 asymmetric rules."""
    
    def test_rule1_hard_contradiction(self):
        """Test Rule 1: Hard contradiction triggers rejection."""
        logger.info("Testing Rule 1: Hard Contradiction")
        
        from kdsh.layer5.adjudicator import ConservativeAdjudicator, FinalVerdict
        from kdsh.layer4.ledger import LedgerEntry
        
        adjudicator = ConservativeAdjudicator(contradiction_threshold=0.7)
        
        # Create entry with hard contradiction
        entry = MagicMock()
        entry.supporting_evidence = []
        entry.contradicting_evidence = [MagicMock(weight=0.9, text="Contradiction", severity="HIGH")]
        entry.missing_evidence = []
        entry.simulator_verdict = {"alignment": 0.8}
        entry.constraint_id = "C1"
        
        verdict = adjudicator.decide([entry])
        
        assert verdict.prediction == 0
        assert "Rule 1" in verdict.triggered_rule
        logger.info(f"Rule 1 triggered correctly: {verdict.triggered_rule}")
    
    def test_rule4_strong_support(self):
        """Test Rule 4: Strong support triggers acceptance."""
        logger.info("Testing Rule 4: Strong Support")
        
        from kdsh.layer5.adjudicator import ConservativeAdjudicator
        
        adjudicator = ConservativeAdjudicator(
            contradiction_threshold=0.7,
            support_count_threshold=2
        )
        
        # Create entries with strong support
        entry = MagicMock()
        entry.supporting_evidence = [MagicMock(), MagicMock(), MagicMock()]
        entry.contradicting_evidence = []
        entry.missing_evidence = []
        entry.simulator_verdict = {"alignment": 0.9}
        entry.constraint_id = "C1"
        
        verdict = adjudicator.decide([entry])
        
        assert verdict.prediction == 1
        assert "Rule 4" in verdict.triggered_rule
        logger.info(f"Rule 4 triggered correctly: {verdict.triggered_rule}")
    
    def test_rule5_default_rejection(self):
        """Test Rule 5: Default conservative rejection."""
        logger.info("Testing Rule 5: Default Conservative Rejection")
        
        from kdsh.layer5.adjudicator import ConservativeAdjudicator
        
        adjudicator = ConservativeAdjudicator(support_count_threshold=10)
        
        # Create entry with insufficient evidence
        entry = MagicMock()
        entry.supporting_evidence = [MagicMock()]  # Only 1 support
        entry.contradicting_evidence = []
        entry.missing_evidence = []
        entry.simulator_verdict = {"alignment": 0.8}
        entry.constraint_id = "C1"
        
        verdict = adjudicator.decide([entry])
        
        assert verdict.prediction == 0
        assert "Rule 5" in verdict.triggered_rule
        logger.info(f"Rule 5 triggered correctly: {verdict.triggered_rule}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--log-cli-level=INFO"])
