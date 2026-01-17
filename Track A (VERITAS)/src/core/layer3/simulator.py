"""
Simulator: Persona consistency verification via character roleplay.

The Simulator adopts the character's persona from the backstory and
tests whether the novel's events are consistent with that persona.
"""

from dataclasses import dataclass, field
from typing import Literal

from core.layer2.unified_retriever import Evidence, RetrievalResult
from core.models import llm_complete


@dataclass
class Persona:
    """Character persona extracted from backstory."""
    
    name: str
    age_description: str
    background: str
    traits: list[str]
    skills: list[str]
    beliefs: list[str]
    fears: list[str]
    
    def to_prompt(self) -> str:
        """Convert persona to roleplay prompt."""
        traits_str = ", ".join(self.traits) if self.traits else "unknown"
        skills_str = ", ".join(self.skills) if self.skills else "none specified"
        
        return f"""You are {self.name}.
Age/Life Stage: {self.age_description}
Background: {self.background}
Personality Traits: {traits_str}
Skills: {skills_str}
Core Beliefs: {', '.join(self.beliefs) if self.beliefs else 'not specified'}
Fears: {', '.join(self.fears) if self.fears else 'not specified'}"""


@dataclass
class SceneAlignment:
    """Alignment score for a single scene."""
    
    scene_text: str
    alignment_score: float
    in_character_aspects: list[str]
    out_of_character_aspects: list[str]
    reasoning: str


@dataclass
class SimulatorVerdict:
    """Verdict from the Simulator verification stream."""
    
    constraint_id: str
    status: Literal["IN_CHARACTER", "OUT_OF_CHARACTER", "AMBIGUOUS"]
    persona_alignment: float  # 0.0 to 1.0
    scene_alignments: list[SceneAlignment] = field(default_factory=list)
    out_of_character_moments: list[str] = field(default_factory=list)
    reasoning: str = ""
    
    @property
    def is_consistent(self) -> bool:
        return self.status == "IN_CHARACTER"


PERSONA_EXTRACTION_PROMPT = """Extract a character persona from this backstory.

BACKSTORY:
{backstory}

Output a JSON object with:
- "name": Character's name (or "Unknown" if not specified)
- "age_description": Age or life stage description
- "background": Brief background summary
- "traits": List of personality traits
- "skills": List of skills or abilities
- "beliefs": List of core beliefs or values
- "fears": List of fears or anxieties

Output ONLY valid JSON:"""


ALIGNMENT_CHECK_PROMPT = """You are roleplaying as this character:

{persona}

SCENE FROM THE NOVEL:
{scene}

CLAIM FROM BACKSTORY:
{claim}

As this character, evaluate:
1. Does the scene align with who you are based on your backstory?
2. Would you (as this character) behave this way or be in this situation?
3. Are there any aspects that feel "out of character"?

Output a JSON object with:
- "alignment_score": 0.0 (completely out of character) to 1.0 (perfectly in character)
- "in_character_aspects": List of aspects that fit your persona
- "out_of_character_aspects": List of aspects that don't fit
- "reasoning": Brief explanation

Output ONLY valid JSON:"""


class Simulator:
    """
    Persona consistency verification via character roleplay.
    
    The Simulator:
    1. Extracts a persona from the backstory
    2. "Roleplays" through novel scenes as that character
    3. Identifies out-of-character moments
    4. Returns an alignment score
    """
    
    def __init__(self, model: str | None = None):
        """
        Initialize the Simulator.
        
        Args:
            model: Optional model override
        """
        self.model = model
        self._persona_cache: dict[str, Persona] = {}
    
    def adopt_persona(self, backstory: str, backstory_id: str = "default") -> Persona:
        """
        Extract and adopt a persona from the backstory.
        
        Args:
            backstory: Character backstory text
            backstory_id: Cache key for persona
            
        Returns:
            Persona object
        """
        if backstory_id in self._persona_cache:
            return self._persona_cache[backstory_id]
        
        prompt = PERSONA_EXTRACTION_PROMPT.format(backstory=backstory)
        
        response = llm_complete(
            prompt=prompt,
            system_prompt="Extract character information. Output only valid JSON.",
            model=self.model,
            temperature=0.0,
            max_tokens=500
        )
        
        # Parse response
        import json
        import re
        
        try:
            response = response.strip()
            if "```json" in response:
                match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
                if match:
                    response = match.group(1)
            
            data = json.loads(response)
            
            persona = Persona(
                name=data.get("name", "Unknown"),
                age_description=data.get("age_description", ""),
                background=data.get("background", ""),
                traits=data.get("traits", []),
                skills=data.get("skills", []),
                beliefs=data.get("beliefs", []),
                fears=data.get("fears", [])
            )
            
        except (json.JSONDecodeError, KeyError):
            persona = Persona(
                name="Unknown",
                age_description="",
                background=backstory[:500],
                traits=[],
                skills=[],
                beliefs=[],
                fears=[]
            )
        
        self._persona_cache[backstory_id] = persona
        return persona
    
    def verify(
        self,
        persona: Persona,
        constraint_claim: str,
        constraint_id: str,
        scenes: list[Evidence]
    ) -> SimulatorVerdict:
        """
        Verify persona consistency against novel scenes.
        
        Args:
            persona: Character persona
            constraint_claim: The claim being verified
            constraint_id: ID of the constraint
            scenes: Retrieved evidence/scenes from the novel
            
        Returns:
            SimulatorVerdict with alignment scores
        """
        scene_alignments = []
        out_of_character_moments = []
        
        # Check alignment for each scene
        for scene in scenes[:5]:  # Limit to 5 scenes
            alignment = self._check_scene_alignment(
                persona, constraint_claim, scene.text
            )
            scene_alignments.append(alignment)
            
            if alignment.out_of_character_aspects:
                out_of_character_moments.extend(alignment.out_of_character_aspects)
        
        # Calculate overall alignment
        if scene_alignments:
            avg_alignment = sum(a.alignment_score for a in scene_alignments) / len(scene_alignments)
        else:
            avg_alignment = 0.5  # Neutral if no scenes
        
        # Determine status
        if avg_alignment >= 0.7:
            status = "IN_CHARACTER"
        elif avg_alignment <= 0.4:
            status = "OUT_OF_CHARACTER"
        else:
            status = "AMBIGUOUS"
        
        # Build reasoning
        if out_of_character_moments:
            reasoning = f"Found {len(out_of_character_moments)} out-of-character aspects: " + \
                        "; ".join(out_of_character_moments[:3])
        else:
            reasoning = f"Persona alignment score: {avg_alignment:.2f}"
        
        return SimulatorVerdict(
            constraint_id=constraint_id,
            status=status,
            persona_alignment=avg_alignment,
            scene_alignments=scene_alignments,
            out_of_character_moments=out_of_character_moments[:5],
            reasoning=reasoning
        )
    
    def _check_scene_alignment(
        self,
        persona: Persona,
        claim: str,
        scene: str
    ) -> SceneAlignment:
        """Check alignment for a single scene."""
        prompt = ALIGNMENT_CHECK_PROMPT.format(
            persona=persona.to_prompt(),
            scene=scene[:2000],
            claim=claim
        )
        
        response = llm_complete(
            prompt=prompt,
            system_prompt="You are roleplaying as the character. Evaluate consistency honestly.",
            model=self.model,
            temperature=0.0,
            max_tokens=400
        )
        
        # Parse response
        import json
        import re
        
        try:
            response = response.strip()
            if "```json" in response:
                match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
                if match:
                    response = match.group(1)
            
            data = json.loads(response)
            
            return SceneAlignment(
                scene_text=scene[:200],
                alignment_score=float(data.get("alignment_score", 0.5)),
                in_character_aspects=data.get("in_character_aspects", []),
                out_of_character_aspects=data.get("out_of_character_aspects", []),
                reasoning=data.get("reasoning", "")
            )
            
        except (json.JSONDecodeError, ValueError):
            return SceneAlignment(
                scene_text=scene[:200],
                alignment_score=0.5,
                in_character_aspects=[],
                out_of_character_aspects=[],
                reasoning="Could not evaluate"
            )
