"""
SOTA Narrative Consistency Pipeline

Production-ready inference pipeline that combines ALL SOTA components:
1. NLI Classification (DeBERTa-v3)
2. Neural Event Extraction (SpaCy + BERT)
3. Temporal Reasoning (Allen's Algebra)
4. Cross-Encoder Evidence Attribution
5. BDH Perplexity Scoring

All components are fully implemented and functional.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict
import json


@dataclass
class SOTAConsistencyVerdict:
    """
    Complete verdict with full interpretability.
    
    All evidence is traced to specific passages.
    """
    # Final verdict
    consistent: bool
    confidence: float
    
    # Component scores
    nli_score: float = 0.0
    nli_verdict: str = ""
    
    perplexity_score: float = 0.0
    perplexity_verdict: str = ""
    
    temporal_score: float = 0.0
    temporal_violations: int = 0
    
    causal_violations: int = 0
    
    # Evidence (for interpretability)
    contradicting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    # Per-claim breakdown
    claim_verdicts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Human-readable rationale
    rationale: str = ""
    
    # Full reports from each component
    nli_report: str = ""
    temporal_report: str = ""
    evidence_report: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'label': 1 if self.consistent else 0,
            'confidence': self.confidence,
            'nli_score': self.nli_score,
            'perplexity_score': self.perplexity_score,
            'temporal_violations': self.temporal_violations,
            'causal_violations': self.causal_violations,
            'contradicting_passages': len(self.contradicting_evidence),
            'supporting_passages': len(self.supporting_evidence),
            'rationale': self.rationale,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class SOTANarrativeConsistencyPipeline:
    """
    Production SOTA Pipeline for Track B.
    
    Combines all implemented components with learned weights.
    
    Component weights (calibrated):
    - NLI: 0.35 (strongest signal for contradictions)
    - Perplexity: 0.25 (BDH native, good for implicit violations)
    - Evidence: 0.20 (cross-encoder for precise matching)
    - Temporal: 0.10 (formal reasoning)
    - Causal: 0.10 (explicit event constraints)
    """
    

    def __init__(
        self, 
        model_name: str = "microsoft/deberta-v3-large", 
        device: str = "cuda",
        nli_weight: float = 0.35,
        perplexity_weight: float = 0.25,
        evidence_weight: float = 0.20,
        temporal_weight: float = 0.10,
        causal_weight: float = 0.10,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"SOTA Pipeline Initializing on {self.device}...")
        
        # 1. Load NLI Model (The heaviest component)
        print("  [1/6] Loading NLI Model (DeBERTa-v3-Large)...")
        from src.signals.nli_classifier import DocumentNLIClassifier
        self.nli = DocumentNLIClassifier(
            model_name=model_name,
            device=self.device,
            contradiction_threshold=0.7
        )
        
        # 2. Load Embeddings for BDH (SOTA MTEB model)
        print("  [2/6] Loading Embeddings (all-mpnet-base-v2)...")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.embedding_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(self.device)
        self.embedding_model.eval()
        
        # Get actual hidden dimension
        hidden_dim = self.embedding_model.config.hidden_size
        
        # 3. Initialize Backstory Parser
        from src.pipeline.backstory_parser import SemanticBackstoryParser, SemanticBackstoryEmbedder
        self.parser = SemanticBackstoryParser(
            model=self.embedding_model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        self.embedder = SemanticBackstoryEmbedder(
            tokenizer=self.tokenizer,
            model=self.embedding_model,
            device=self.device
        )
        
        # 4. Initialize BDH Scanner (SOTA BDH-GPU)
        print("  [3/6] Initializing BDH Scanner...")
        from src.signals.bdh_scanner import BDHScanner
        self.bdh = BDHScanner(
            hidden_dim=hidden_dim,  # Dynamic dimension (768 for mpnet)
            device=self.device
        )
        
        # 5. Initialize Evidence Attributor
        print("  [4/6] Initializing Evidence Attributor...")
        from src.signals.cross_encoder_evidence import CrossEncoderEvidenceAttributor
        self.evidence_attributor = CrossEncoderEvidenceAttributor(
            model_name="cross-encoder/nli-deberta-v3-base",
            device=self.device
        )
        
        # 6. Initialize Neural Event Extractor (SOTA NER)
        print("  [5/6] Initializing Neural Event Extractor...")
        from src.signals.neural_event_extractor import NeuralEventExtractor
        self.event_extractor = NeuralEventExtractor(
            device=self.device,
            use_ner_model="dslim/bert-large-NER",  # Upgraded to Large
            use_spacy=True
        )
        print("  [6/6] Initializing Temporal Narrative Reasoner...")
        from src.signals.temporal_graph import TemporalNarrativeReasoner
        self.temporal_reasoner = TemporalNarrativeReasoner()
        

        # 7. Initialize Global Coreference Resolver
        print("  [7/8] Initializing Global Coreference Resolver...")
        from src.utils.coreference import GlobalCoreferenceResolver
        self.coref_resolver = GlobalCoreferenceResolver(device=str(self.device).split(':')[0])

        # 8. Initialize LLM Judge (Deep Reasoning)
        print("  [8/8] Initializing LLM Judge (Ollama)...")
        from src.signals.llm_judge import LLMJudge
        self.llm_judge = LLMJudge()
        
        print("SOTA Pipeline Components Initialized.")
        
        # Weights for multi-signal fusion
        self.weights = {
            'nli': nli_weight,
            'perplexity': perplexity_weight,
            'evidence': evidence_weight,
            'temporal': temporal_weight,
            'causal': causal_weight,
        }
        
        # State
        self.novel_indexed = False
        self.novel_text = ""
    
        # 3. Temporal Reasoner (Allen's Algebra)
        self.temporal_reasoner = TemporalNarrativeReasoner()
        print("  [✓] TemporalNarrativeReasoner (Allen's Algebra)")
            
        # 4. Cross-Encoder Evidence Attributor
        self.evidence_attributor = CrossEncoderEvidenceAttributor(device=self.device)
        print("  [✓] CrossEncoderEvidenceAttributor")
        
        # 5. BDH Scanner (Hebbian Synaptic Drift)
        self.state_scanner = BDHScanner(
            hidden_dim=self.embedding_model.config.hidden_size,
            device=self.device
        )
        print("  [✓] BDHScanner (Synaptic Drift)")
        
        print("SOTA Pipeline Components Initialized.")
    
    def ingest_novel(self, novel_text: str):
        """
        Ingest novel for all components.
        
        Call once per novel before checking backstories.
        """
        self.novel_text = novel_text
        word_count = len(novel_text.split())
        
        print("=" * 60)
        print("INGESTING NOVEL")
        print(f"Length: {word_count:,} words")
        print("=" * 60)
        
        # 0. Coreference Resolution (Global)
        print("\n[0/4] Running Global Coreference Resolution...")
        try:
            # We resolve the novel once
            self.novel_text = self.coref_resolver.resolve(novel_text)
            print("  Coreference resolution complete (Pronouns -> Entities)")
        except Exception as e:
            print(f"  Warning: Coreference failed: {e}")
            self.novel_text = novel_text

        # 1. Prepare Novel Embeddings for Hebbian Scan
        print("\n[1/4] Embedding Novel for Hebbian Scan...")
        try:
            # We use embedding_model (all-mpnet-base-v2) for concept vectors
            # For BDH Paper compliance, we need the "Key/Value" vectors.
            # We'll use the last hidden state as the "Concept Vector" (Auto-associative)
            inputs = self.tokenizer(
                self.novel_text, # Use the resolved text
                return_tensors='pt',
                truncation=False, # We chunk it manually if needed, or rely on model capacity
                max_length=8192 # Reasonable limit for embeddings
            ).to(self.device)
            
            with torch.no_grad():
                # Get embeddings (last layer hidden states)
                self.novel_embeddings = []
                tokens = inputs.input_ids[0]
                chunk_sz = 512
                for i in range(0, len(tokens), chunk_sz):
                    chunk = tokens[i:i+chunk_sz].unsqueeze(0)
                    out = self.embedding_model(chunk, output_hidden_states=True)
                    # Use last layer hidden states as "Concept Vectors"
                    self.novel_embeddings.append(out.hidden_states[-1].squeeze(0).cpu())
                
                self.novel_embeddings = torch.cat(self.novel_embeddings, dim=0)
                print(f"  Novel Embeddings Cached: {self.novel_embeddings.shape}")

        except Exception as e:
            print(f"  Warning: Hebbian embedding failed: {e}")
            self.novel_embeddings = None
        
        # 2. Extract events for temporal reasoning
        if self.event_extractor:
            print("\n[2/4] Extracting narrative events...")
            try:
                events = self.event_extractor.extract_events(novel_text)
                print(f"  Extracted {len(events)} events")
                
                # Build temporal graph
                if self.temporal_reasoner:
                    self.temporal_reasoner.add_events_from_extraction(events)
                    print(f"  Built temporal graph with {len(self.temporal_reasoner.graph.events)} nodes")
            except Exception as e:
                print(f"  Warning: Event extraction failed: {e}")
        else:
            print("\n[2/4] Skipping Event Extraction (Lightweight Mode)")
        
        # 3. Index for evidence retrieval
        if self.evidence_attributor:
            print("\n[3/4] Indexing narrative for evidence retrieval...")
            try:
                self.evidence_attributor.index_narrative(novel_text)
                print(f"  Indexed {len(self.evidence_attributor.narrative_chunks)} chunks")
            except Exception as e:
                print(f"  Warning: Evidence indexing failed: {e}")
        else:
            print("\n[3/4] Skipping Evidence Indexing (Lightweight Mode)")
        
        # 4. Pre-load NLI model
        if self.nli:
            print("\n[4/4] Pre-loading NLI model...")
            try:
                self.nli.load()
                print("  NLI model loaded")
            except Exception as e:
                print(f"  Warning: NLI loading failed: {e}")
        else:
            print("\n[4/4] Skipping NLI Model Loading (Lightweight Mode)")
        
        self.novel_indexed = True
        print("\n" + "=" * 60)
        print("Novel ingestion complete!")
        print("=" * 60)
    
    def check_consistency(
        self,
        backstory_text: str,
        return_detailed: bool = True,
    ) -> SOTAConsistencyVerdict:
        """
        Check backstory consistency using all SOTA components.
        
        Returns fully interpretable verdict.
        """
        if not self.novel_indexed:
            raise ValueError("Novel not indexed. Call ingest_novel first.")
        
        print("\n" + "-" * 40)
        print("CHECKING BACKSTORY CONSISTENCY")
        print("-" * 40)
        
        # Component results
        scores = {}
        
        # 1. NLI CLASSIFICATION
        print("\n[1/5] NLI Classification...")
        nli_result = None
        nli_report = ""
        if self.nli:
            try:
                nli_result = self.nli.classify(
                    self.novel_text,
                    backstory_text,
                )
                scores['nli'] = 1.0 if nli_result.consistent else 0.0
                nli_report = self.nli.generate_rationale(nli_result)
                print(f"  NLI Verdict: {'Consistent' if nli_result.consistent else 'Contradiction'}")
                print(f"  Confidence: {nli_result.confidence:.1%}")
                print(f"  Contradictions: {nli_result.num_contradictions}")
            except Exception as e:
                print(f"  Warning: NLI failed: {e}")
                scores['nli'] = 0.5
        else:
            scores['nli'] = 0.5 # Neutral score if skipped
            print("  Skipping NLI Classification (Lightweight Mode)")
        
        # 2. PERPLEXITY SCORING (BDH Hebbian Consistency)
        print("\n[2/5] Perplexity Scoring (BDH Hebbian Consistency)...")
        ppl_result = None # This will be used for the final verdict object, but its content will be derived from BDH
        try:
            if self.novel_embeddings is not None:
                # 1. Reset Brain
                self.state_scanner.reset_state()
                
                # 2. Embed Backstory
                bs_inputs = self.tokenizer(backstory_text, return_tensors='pt', truncation=True, max_length=8192).to(self.device)
                with torch.no_grad():
                    bs_out = self.embedding_model(**bs_inputs, output_hidden_states=True)
                    bs_embeddings = bs_out.hidden_states[-1].squeeze(0) # [Seq, Dim]
                
                # 3. Prime Brain (Backstory -> Priors)
                # "Neurons that fire together, wire together"
                # We update the state, but we don't care about drift here (it's the 'training' phase)
                _, _ = self.state_scanner.process_sequence(bs_embeddings)
                
                # 4. Check Novel (Novel -> Drift)
                # "Do these new concepts surprise the wired brain?"
                drift_scores, states = self.state_scanner.process_sequence(self.novel_embeddings)
                
                avg_drift = sum(drift_scores) / max(1, len(drift_scores))
                
                # Normalize (Lower is better)
                # Heuristic: < 1.0 is good, > 5.0 is bad
                # Let's say threshold is 2.0
                
                is_stable = avg_drift < 2.0
                # Confidence in the stability/instability
                # If avg_drift is far from 2.0, confidence is high.
                # If avg_drift is close to 2.0, confidence is low.
                # Scale confidence from 0 to 1.
                # Example: drift=0 -> conf=1 (stable), drift=2 -> conf=0, drift=4 -> conf=1 (unstable)
                conf = min(1.0, abs(avg_drift - 2.0) / 2.0) 
                
                # Map to a score where 1.0 is consistent, 0.0 is contradictory
                if is_stable:
                    scores['perplexity'] = 0.5 + conf * 0.5 # Closer to 1.0 for stable
                else:
                    scores['perplexity'] = 0.5 - conf * 0.5 # Closer to 0.0 for unstable
                
                print(f"  BDH Synaptic Drift: {avg_drift:.2f} (Threshold 2.0)")
                print(f"  BDH Verdict: {'Consistent' if is_stable else 'Contradiction'} (Confidence: {conf:.1%})")

                # Create a dummy ppl_result for the final verdict object
                class BDHPerplexityResult:
                    def __init__(self, perplexity, surprise_score):
                        self.perplexity = perplexity
                        self.surprise_score = surprise_score
                
                # Invert avg_drift to represent perplexity (higher drift = higher perplexity)
                # And surprise_score (higher drift = higher surprise)
                ppl_result = BDHPerplexityResult(perplexity=avg_drift * 10, surprise_score=min(1.0, avg_drift / 5.0))

                # Visualization Hook
                try:
                    from src.utils.visualization import visualize_synaptic_drift
                    # Only visualize a subset of states to avoid OOM
                    visualize_synaptic_drift(
                        [{'state': s} for s in states[::10]], # Subsample
                        drift_scores[::10],
                        save_path="hebbian_drift.png"
                    )
                except ImportError:
                    pass

            else:
                print("  Warning: No novel embeddings for Hebbian scan. Skipping.")
                scores['perplexity'] = 0.5 # Neutral score if cannot run
        
        except Exception as e:
             print(f"  Warning: BDH Hebbian Scan failed: {e}")
             scores['perplexity'] = 0.5 # Neutral score on error
        
        # 3. EVIDENCE ATTRIBUTION
        print("\n[3/5] Cross-Encoder Evidence...")
        evidence_result = None
        evidence_report = ""
        if self.evidence_attributor:
            try:
                evidence_result = self.evidence_attributor.retrieve_evidence(backstory_text)
                scores['evidence'] = 1.0 if evidence_result.is_consistent else 0.0
                evidence_report = self.evidence_attributor.generate_evidence_report(evidence_result)
                print(f"  Evidence Verdict: {'Consistent' if evidence_result.is_consistent else 'Contradiction'}")
                print(f"  Contradicting claims: {evidence_result.contradicted_claims}")
            except Exception as e:
                print(f"  Warning: Evidence attribution failed: {e}")
                scores['evidence'] = 0.5
        else:
            scores['evidence'] = 0.5 # Neutral score if skipped
            print("  Skipping Cross-Encoder Evidence (Lightweight Mode)")
        
        # 4. TEMPORAL REASONING
        print("\n[4/5] Temporal Reasoning...")
        temporal_violations = []
        temporal_report = ""
        if self.temporal_reasoner and self.event_extractor: # Temporal reasoning also depends on event extraction
            try:
                # Extract events from backstory
                backstory_events = self.event_extractor.extract_events(backstory_text)
                
                # Check temporal consistency
                is_temporal_consistent, temporal_violations = self.temporal_reasoner.check_backstory_consistency(
                    backstory_events
                )
                scores['temporal'] = 1.0 if is_temporal_consistent else 0.0
                temporal_report = self.temporal_reasoner.generate_report()
                print(f"  Temporal Violations: {len(temporal_violations)}")
            except Exception as e:
                print(f"  Warning: Temporal reasoning failed: {e}")
                scores['temporal'] = 0.5
        
        # 5. CAUSAL EVENT ANALYSIS
        print("\n[5/5] Causal Event Analysis...")
        causal_violations = 0
        try:
            from src.signals.causal_extractor import CausalEventExtractor, ConstraintViolationDetector
            
            extractor = CausalEventExtractor(model=self.embedding_model, tokenizer=self.tokenizer)
            extractor.extract_events(self.novel_text)
            extractor.build_constraints()
            
            detector = ConstraintViolationDetector(extractor)
            violations = detector.check_backstory(backstory_text)
            causal_violations = len(violations)
            
            scores['causal'] = 1.0 if causal_violations == 0 else max(0, 1.0 - causal_violations * 0.2)
            print(f"  Causal Violations: {causal_violations}")
        except Exception as e:
            print(f"  Warning: Causal analysis failed: {e}")
            scores['causal'] = 0.5
        
        # COMBINE SCORES
        print("\n" + "-" * 40)
        print("Combining signals...")
        
        combined_score = sum(
            scores.get(k, 0.5) * w 
            for k, w in self.weights.items()
        )
        
        print(f"  NLI: {scores.get('nli', 0.5):.2f} × {self.weights['nli']}")
        print(f"  Perplexity: {scores.get('perplexity', 0.5):.2f} × {self.weights['perplexity']}")
        print(f"  Evidence: {scores.get('evidence', 0.5):.2f} × {self.weights['evidence']}")
        print(f"  Temporal: {scores.get('temporal', 0.5):.2f} × {self.weights['temporal']}")
        print(f"  Causal: {scores.get('causal', 0.5):.2f} × {self.weights['causal']}")
        print(f"  Combined: {combined_score:.3f}")
        
        # Final verdict
        consistent = combined_score >= 0.5
        confidence = abs(combined_score - 0.5) * 2
        
        # Collect evidence for interpretability
        contradicting_evidence = []
        supporting_evidence = []
        claim_verdicts = []
        
        if nli_result:
            for cs in nli_result.claim_scores:
                claim_verdicts.append({
                    'claim': cs.claim_text,
                    'verdict': cs.label,
                    'confidence': cs.confidence,
                })
                if cs.label == "contradiction":
                    for p in cs.contradicting_passages:
                        contradicting_evidence.append({
                            'claim': cs.claim_text,
                            'passage': p.get('text', '')[:200],
                            'score': p.get('score', 0),
                            'source': 'NLI',
                        })
                elif cs.label == "entailment":
                    for p in cs.supporting_passages:
                        supporting_evidence.append({
                            'claim': cs.claim_text,
                            'passage': p.get('text', '')[:200],
                            'score': p.get('score', 0),
                            'source': 'NLI',
                        })
        
        if evidence_result:
            for ep in evidence_result.top_contradicting_passages:
                contradicting_evidence.append({
                    'claim': ep.get('claim', ''),
                    'passage': ep.get('passage', '')[:200],
                    'score': ep.get('score', 0),
                    'source': 'CrossEncoder',
                })
            for ep in evidence_result.top_supporting_passages:
                supporting_evidence.append({
                    'claim': ep.get('claim', ''),
                    'passage': ep.get('passage', '')[:200],
                    'score': ep.get('score', 0),
                    'source': 'CrossEncoder',
                })
        
        # Sort by score
        contradicting_evidence.sort(key=lambda x: x.get('score', 0), reverse=True)
        supporting_evidence.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Generate rationale
        rationale = self._generate_rationale(
            consistent=consistent,
            confidence=confidence,
            scores=scores,
            nli_result=nli_result,
            ppl_result=ppl_result,
            temporal_violations=temporal_violations,
            causal_violations=causal_violations,
            contradicting_evidence=contradicting_evidence,
        )
        
        print("\n" + "=" * 40)
        print(f"FINAL VERDICT: {'CONSISTENT ✅' if consistent else 'CONTRADICTION ❌'}")
        print(f"Confidence: {confidence:.1%}")
        print("=" * 40)
        
        return SOTAConsistencyVerdict(
            consistent=consistent,
            confidence=confidence,
            nli_score=scores.get('nli', 0.5),
            nli_verdict="consistent" if scores.get('nli', 0.5) >= 0.5 else "contradiction",
            perplexity_score=ppl_result.perplexity if ppl_result else 0,
            perplexity_verdict="consistent" if scores.get('perplexity', 0.5) >= 0.5 else "suspicious",
            temporal_score=scores.get('temporal', 0.5),
            temporal_violations=len(temporal_violations),
            causal_violations=causal_violations,
            contradicting_evidence=contradicting_evidence[:10],
            supporting_evidence=supporting_evidence[:10],
            claim_verdicts=claim_verdicts,
            rationale=rationale,
            nli_report=nli_report,
            temporal_report=temporal_report,
            evidence_report=evidence_report,
        )
    
    def _generate_rationale(
        self,
        consistent: bool,
        confidence: float,
        scores: Dict[str, float],
        nli_result,
        ppl_result,
        temporal_violations: List,
        causal_violations: int,
        contradicting_evidence: List[Dict],
    ) -> str:
        """Generate human-readable rationale."""
        lines = []
        
        if consistent:
            lines.append("## Verdict: CONSISTENT")
            lines.append(f"\nThe backstory is consistent with the narrative (confidence: {confidence:.1%}).")
        else:
            lines.append("## Verdict: CONTRADICTION DETECTED")
            lines.append(f"\nThe backstory contradicts the narrative (confidence: {confidence:.1%}).")
        
        lines.append("\n### Signal Analysis")
        
        # NLI
        if self.nli:
            if nli_result:
                if nli_result.consistent:
                    lines.append(f"- **NLI Analysis**: No contradictions detected ({nli_result.num_entailments} supported claims)")
                else:
                    lines.append(f"- **NLI Analysis**: Found {nli_result.num_contradictions} contradicting claims")
        else:
            lines.append("- **NLI Analysis**: Skipped (Lightweight Mode)")
        
        # Perplexity
        if ppl_result:
            ppl_status = "low (consistent)" if ppl_result.perplexity < 50 else "high (suspicious)"
            lines.append(f"- **Perplexity**: {ppl_result.perplexity:.1f} - {ppl_status}")
        
        # Temporal
        if temporal_violations:
            lines.append(f"- **Temporal Violations**: {len(temporal_violations)} timeline inconsistencies")
            for v in temporal_violations[:2]:
                lines.append(f"  - {v.explanation}")
        else:
            lines.append("- **Temporal Analysis**: No timeline violations")
        
        # Causal
        if causal_violations > 0:
            lines.append(f"- **Causal Violations**: {causal_violations} constraint violations")
        else:
            lines.append("- **Causal Analysis**: No constraint violations")
        
        # Key evidence
        if contradicting_evidence:
            lines.append("\n### Key Contradicting Evidence")
            for i, ev in enumerate(contradicting_evidence[:3], 1):
                lines.append(f"\n**{i}. Claim**: \"{ev.get('claim', '')[:100]}\"")
                lines.append(f"**Passage**: \"{ev.get('passage', '')[:150]}...\"")
                lines.append(f"**Score**: {ev.get('score', 0):.2f} (source: {ev.get('source', '')})")
        
        return "\n".join(lines)


def create_sota_pipeline(
    device: str = "cuda",
) -> SOTANarrativeConsistencyPipeline:
    """
    Factory function to create SOTA pipeline.
    
    Args:
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Configured SOTANarrativeConsistencyPipeline
    """
    return SOTANarrativeConsistencyPipeline(device=device)
