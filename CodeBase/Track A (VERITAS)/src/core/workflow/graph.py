"""
LangGraph workflow definition for the 5-layer Narrative Auditor.

Based on: https://github.com/langchain-ai/langgraph
"""

from typing import Any

from langgraph.graph import StateGraph, START, END

from kdsh.workflow.state import NarrativeState, create_initial_state


# ============================================================================
# Layer 1: Atomic Constraint Decomposition
# ============================================================================

def layer1_decompose(state: NarrativeState) -> dict:
    """
    Layer 1: Decompose backstory into atomic constraints.
    
    Uses FActScore for fact extraction and BART-MNLI for classification.
    """
    from kdsh.layer1 import decompose_backstory
    
    try:
        constraint_graph = decompose_backstory(
            backstory=state["backstory"],
            backstory_id=state["backstory_id"]
        )
        
        return {
            "constraint_graph": constraint_graph,
            "constraints": list(constraint_graph.constraints),
            "current_layer": "layer1_complete"
        }
    except Exception as e:
        return {"error": f"Layer 1 error: {str(e)}", "current_layer": "layer1_error"}


# ============================================================================
# Layer 2: Dual Retrieval
# ============================================================================

def layer2_retrieve(state: NarrativeState) -> dict:
    """
    Layer 2: Build RAPTOR + HippoRAG indexes and retrieve evidence.
    """
    from kdsh.layer2 import DualRetriever
    
    try:
        retriever = DualRetriever(
            novel_text=state["novel_text"],
            novel_id=state["novel_id"]
        )
        
        retrieval_results = {}
        for constraint in state["constraints"]:
            result = retriever.retrieve(constraint)
            retrieval_results[constraint.id] = result
        
        return {
            "retrieval_results": retrieval_results,
            "current_layer": "layer2_complete"
        }
    except Exception as e:
        return {"error": f"Layer 2 error: {str(e)}", "current_layer": "layer2_error"}


# ============================================================================
# Layer 3a: Historian Verification
# ============================================================================

def layer3_historian(state: NarrativeState) -> dict:
    """
    Layer 3a: Evidence-based verification via Historian.
    """
    from kdsh.layer3 import Historian
    
    try:
        historian = Historian()
        
        historian_verdicts = {}
        for constraint in state["constraints"]:
            retrieval_result = state["retrieval_results"].get(constraint.id)
            if retrieval_result:
                verdict = historian.verify(constraint, retrieval_result)
                historian_verdicts[constraint.id] = verdict
        
        return {
            "historian_verdicts": historian_verdicts,
            "current_layer": "layer3a_complete"
        }
    except Exception as e:
        return {"error": f"Layer 3a error: {str(e)}", "current_layer": "layer3a_error"}


# ============================================================================
# Layer 3b: Simulator Verification
# ============================================================================

def layer3_simulator(state: NarrativeState) -> dict:
    """
    Layer 3b: Persona consistency verification via Simulator.
    """
    from kdsh.layer3 import Simulator
    
    try:
        simulator = Simulator()
        
        # Adopt persona from backstory
        persona = simulator.adopt_persona(
            backstory=state["backstory"],
            backstory_id=state["backstory_id"]
        )
        
        simulator_verdicts = {}
        for constraint in state["constraints"]:
            retrieval_result = state["retrieval_results"].get(constraint.id)
            scenes = retrieval_result.raptor_evidence if retrieval_result else []
            
            verdict = simulator.verify(
                persona=persona,
                constraint_claim=constraint.claim,
                constraint_id=constraint.id,
                scenes=scenes
            )
            simulator_verdicts[constraint.id] = verdict
        
        return {
            "persona": persona,
            "simulator_verdicts": simulator_verdicts,
            "current_layer": "layer3b_complete"
        }
    except Exception as e:
        return {"error": f"Layer 3b error: {str(e)}", "current_layer": "layer3b_error"}


# ============================================================================
# Layer 4: Evidence Ledger
# ============================================================================

def layer4_aggregate(state: NarrativeState) -> dict:
    """
    Layer 4: Aggregate evidence into structured ledger.
    """
    from kdsh.layer4 import EvidenceLedger
    
    try:
        ledger = EvidenceLedger()
        
        for constraint in state["constraints"]:
            historian_verdict = state["historian_verdicts"].get(constraint.id)
            simulator_verdict = state["simulator_verdicts"].get(constraint.id)
            
            if historian_verdict and simulator_verdict:
                ledger.aggregate(constraint, historian_verdict, simulator_verdict)
        
        return {
            "ledger_entries": ledger.get_all_entries(),
            "current_layer": "layer4_complete"
        }
    except Exception as e:
        return {"error": f"Layer 4 error: {str(e)}", "current_layer": "layer4_error"}


# ============================================================================
# Layer 5: Conservative Adjudication
# ============================================================================

def layer5_decide(state: NarrativeState) -> dict:
    """
    Layer 5: Apply conservative decision rules for final verdict.
    """
    from kdsh.layer5 import ConservativeAdjudicator
    
    try:
        adjudicator = ConservativeAdjudicator()
        
        final_verdict = adjudicator.decide(state["ledger_entries"])
        
        return {
            "final_verdict": final_verdict,
            "prediction": final_verdict.prediction,
            "confidence": final_verdict.confidence,
            "reasoning": final_verdict.reasoning_chain,
            "current_layer": "layer5_complete"
        }
    except Exception as e:
        return {"error": f"Layer 5 error: {str(e)}", "current_layer": "layer5_error"}


# ============================================================================
# Workflow Builder
# ============================================================================

def build_workflow() -> StateGraph:
    """
    Build the 5-layer LangGraph workflow.
    
    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(NarrativeState)
    
    # Add nodes for each layer
    workflow.add_node("layer1_decompose", layer1_decompose)
    workflow.add_node("layer2_retrieve", layer2_retrieve)
    workflow.add_node("layer3_historian", layer3_historian)
    workflow.add_node("layer3_simulator", layer3_simulator)
    workflow.add_node("layer4_aggregate", layer4_aggregate)
    workflow.add_node("layer5_decide", layer5_decide)
    
    # Define edges (sequential flow)
    workflow.add_edge(START, "layer1_decompose")
    workflow.add_edge("layer1_decompose", "layer2_retrieve")
    workflow.add_edge("layer2_retrieve", "layer3_historian")
    workflow.add_edge("layer3_historian", "layer3_simulator")
    workflow.add_edge("layer3_simulator", "layer4_aggregate")
    workflow.add_edge("layer4_aggregate", "layer5_decide")
    workflow.add_edge("layer5_decide", END)
    
    return workflow.compile()


# ============================================================================
# Pipeline Runner
# ============================================================================

def run_pipeline(
    backstory: str,
    novel_text: str,
    backstory_id: str = "default",
    novel_id: str = "default"
) -> dict:
    """
    Run the full 5-layer pipeline.
    
    Args:
        backstory: Character backstory text
        novel_text: Full novel text  
        backstory_id: Unique ID for the backstory
        novel_id: Unique ID for the novel
        
    Returns:
        Dictionary with prediction, confidence, reasoning, and full state
    """
    # Create workflow
    workflow = build_workflow()
    
    # Create initial state
    initial_state = create_initial_state(
        backstory=backstory,
        novel_text=novel_text,
        backstory_id=backstory_id,
        novel_id=novel_id
    )
    
    # Run workflow
    final_state = workflow.invoke(initial_state)
    
    # Extract result
    return {
        "prediction": final_state.get("prediction", 0),
        "confidence": final_state.get("confidence", 0.0),
        "reasoning": final_state.get("reasoning", []),
        "error": final_state.get("error"),
        "verdict": final_state.get("final_verdict"),
        "full_state": final_state
    }


# Convenience function for single prediction
def predict(backstory: str, novel_text: str) -> tuple[int, float, list[str]]:
    """
    Make a single prediction.
    
    Args:
        backstory: Character backstory text
        novel_text: Full novel text
        
    Returns:
        Tuple of (prediction, confidence, reasoning)
    """
    result = run_pipeline(backstory, novel_text)
    return (
        result["prediction"],
        result["confidence"],
        result["reasoning"]
    )
