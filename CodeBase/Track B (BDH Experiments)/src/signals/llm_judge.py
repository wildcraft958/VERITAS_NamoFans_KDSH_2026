
import requests
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class LLMVerdict:
    consistent: bool
    confidence: float
    rationale: str
    model_used: str

class LLMJudge:
    """
    Deep Reasoning Judge using local LLMs (via Ollama).
    
    Acts as the final arbiter for ambiguous cases by performing 
    RAG-style reasoning over the retrieved evidence.
    """
    
    def __init__(self, model_name: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self._available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            # Simple check
            response = requests.get(self.base_url)
            if response.status_code == 200:
                print(f"  [✓] Ollama detected at {self.base_url}")
                return True
        except requests.exceptions.RequestException:
            print(f"  [!] Ollama NOT detected at {self.base_url}. LLM Judge disabled.")
        return False

    def judge(self, claim: str, evidence: List[Dict[str, str]]) -> LLMVerdict:
        """
        Judge the consistency of a claim given evidence.
        
        Args:
            claim: The user's backstory claim
            evidence: List of dicts with 'passage' and 'score'
            
        Returns:
            LLMVerdict with rationale
        """
        if not self._available or not evidence:
            return LLMVerdict(True, 0.5, "LLM unavailable or no evidence.", "none")

        # Format context
        context_str = "\n\n".join([f"Passage {i+1}: {e['passage']}" for i, e in enumerate(evidence)])
        
        prompt = f"""You are an expert narrative consistency analyzer. 
        Determine if the following CLAIM is consistent with the provided NOVEL EXCERPTS.
        
        NOVEL EXCERPTS:
        {context_str}
        
        CLAIM: "{claim}"
        
        task:
        1. Analyze if the claim contradicts any facts in the excerpts.
        2. If there is a direct contradiction, label as INCONSISTENT.
        3. If the claim is supported or neutral (not contradicted), label as CONSISTENT.
        
        Output JSON format only:
        {{
            "consistent": true/false,
            "confidence": 0.0 to 1.0,
            "rationale": "One sentence explanation citing specific facts."
        }}
        """
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('response', '{}')
            
            data = json.loads(response_text)
            
            return LLMVerdict(
                consistent=data.get('consistent', True),
                confidence=float(data.get('confidence', 0.5)),
                rationale=data.get('rationale', "No rationale provided."),
                model_used=self.model_name
            )
            
        except Exception as e:
            print(f"LLM Judge Error: {e}")
            return LLMVerdict(True, 0.0, f"Error: {str(e)}", "error")

