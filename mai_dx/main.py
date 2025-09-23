"""
MAI Diagnostic Orchestrator (MAI-DxO)

This script provides a complete implementation of the "Sequential Diagnosis with Language Models"
paper, using the `swarms` framework. It simulates a virtual panel of physician-agents to perform
iterative medical diagnosis with cost-effectiveness optimization.

Based on the paper: "Sequential Diagnosis with Language Models"
(arXiv:2506.22405v1) by Nori et al.

Key Features:
- Virtual physician panel with specialized roles (Hypothesis, Test-Chooser, Challenger, Stewardship, Checklist)
- Multiple operational modes (instant, question_only, budgeted, no_budget, ensemble)
- Comprehensive cost tracking and budget management
- Clinical accuracy evaluation with 5-point Likert scale
- Gatekeeper system for realistic clinical information disclosure
- Ensemble methods for improved diagnostic accuracy

Example Usage:
    # Standard MAI-DxO usage
    orchestrator = MaiDxOrchestrator(model_name="gpt-4o")
    result = orchestrator.run(initial_case_info, full_case_details, ground_truth)

    # Budget-constrained variant
    budgeted_orchestrator = MaiDxOrchestrator.create_variant("budgeted", budget=5000)

    # Ensemble approach
    ensemble_result = orchestrator.run_ensemble(initial_case_info, full_case_details, ground_truth)
"""

# Enable debug mode if environment variable is set
import os
import json
import re
import sys
import time
import ast # BUGFIX: Import ast for safe literal evaluation
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Union, Literal, Optional
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

# Import Windows compatibility layer before swarms
if sys.platform == "win32":
    # Add parent directory to path so we can import from webui
    import pathlib
    webui_path = pathlib.Path(__file__).parent.parent / "webui"
    if str(webui_path) not in sys.path:
        sys.path.insert(0, str(webui_path))
    
    try:
        from swarms_compat import ensure_windows_compatibility
        ensure_windows_compatibility()
    except ImportError:
        # Fallback to inline shim if webui module not available
        import types as _types
        _uv = _types.ModuleType("uvloop")
        _uv.install = lambda: None  # type: ignore
        sys.modules["uvloop"] = _uv
        
        # Also create MCP shims inline as fallback
        if "mcp.client.streamable_http" not in sys.modules:
            mcp_client_streamable = _types.ModuleType("mcp.client.streamable_http")
            def streamablehttp_client(*args, **kwargs):
                return None
            mcp_client_streamable.streamablehttp_client = streamablehttp_client
            sys.modules["mcp.client.streamable_http"] = mcp_client_streamable

from swarms import Agent, Conversation
import litellm
from dotenv import load_dotenv

load_dotenv()

# Configure Loguru with beautiful formatting and features
logger.remove()  # Remove default handler

# Console handler with beautiful colors
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)


if os.getenv("MAIDX_DEBUG", "").lower() in ("1", "true", "yes"):
    logger.add(
        "logs/maidx_debug_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="3 days",
    )
    logger.info(
        "🛠 Debug logging enabled - logs will be written to logs/ directory"
    )

# File handler for persistent logging (optional - uncomment if needed)
# logger.add(
#     "logs/mai_dxo_{time:YYYY-MM-DD}.log",
#     rotation="1 day",
#     retention="7 days",
#     level="DEBUG",
#     format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
#     compression="zip"
# )

# --- Data Structures and Enums ---


class AgentRole(Enum):
    """Enumeration of roles for the virtual physician panel."""

    HYPOTHESIS = "Dr. Hypothesis"
    TEST_CHOOSER = "Dr. Test-Chooser"
    CHALLENGER = "Dr. Challenger"
    STEWARDSHIP = "Dr. Stewardship"
    CHECKLIST = "Dr. Checklist"
    CONSENSUS = "Consensus Coordinator"
    GATEKEEPER = "Gatekeeper"
    JUDGE = "Judge"


@dataclass
class CaseState:
    """Structured state management for diagnostic process - addresses Category 2.1

    Backward-compatibility: accept either `initial_vignette` (preferred) or
    the legacy/mistyped `initial_presentation` without crashing.
    """
    initial_vignette: str = ""
    # Accept legacy/mistyped key to avoid runtime crashes from stale callers
    initial_presentation: Optional[str] = None
    evidence_log: List[str] = field(default_factory=list)
    differential_diagnosis: Dict[str, float] = field(default_factory=dict)
    tests_performed: List[str] = field(default_factory=list)
    questions_asked: List[str] = field(default_factory=list)
    cumulative_cost: int = 0
    iteration: int = 0
    last_actions: List['Action'] = field(default_factory=list)  # For stagnation detection

    def __post_init__(self) -> None:
        # If the preferred field wasn't provided, fall back to the legacy name
        if not self.initial_vignette and self.initial_presentation:
            self.initial_vignette = self.initial_presentation
        if not self.initial_vignette:
            raise ValueError("CaseState requires initial_vignette (or legacy initial_presentation).")

    def add_evidence(self, evidence: str, source: str = "System"):
        """Add new evidence to the case"""
        # BUGFIX: Ensure that the full hidden prompt from the Gatekeeper is not logged.
        # This prevents information leakage to other agents in subsequent rounds.
        # Only log the response content, not the entire conversational turn object.
        clean_evidence = evidence
        if "Full Case Details (for your reference only):" in evidence:
            # Extract just the Gatekeeper's response part
            parts = re.split(r'gatekeeper:', evidence, flags=re.IGNORECASE)
            if len(parts) > 1:
                clean_evidence = parts[-1].strip()
            else:
                # Fallback if the specific format changes
                clean_evidence = "Gatekeeper provided new clinical information."

        self.evidence_log.append(f"[Turn {self.iteration}] {source}: {clean_evidence}")

    def update_differential(self, diagnosis_dict: Dict[str, float]):
        """Update differential diagnosis probabilities and log the changes."""
        changes = []
        # Sort for consistent log output
        for diag, new_prob in sorted(diagnosis_dict.items(), key=lambda item: item[1], reverse=True):
            old_prob = self.differential_diagnosis.get(diag)
            if old_prob is None:
                changes.append(f"+ '{diag}' ({new_prob:.0%})")
            elif abs(old_prob - new_prob) > 0.01: # Log only significant changes
                changes.append(f"~ '{diag}' ({old_prob:.0%} -> {new_prob:.0%})")

        if changes:
            logger.info(f"📊 Differential updated: {', '.join(changes)}")
        else:
            logger.debug("Differential diagnosis received but no significant changes detected.")
        
        self.differential_diagnosis.update(diagnosis_dict)

    def add_test(self, test_name: str):
        """Record a test that was performed and log it as a state change."""
        # Normalize test name for accurate duplicate checking
        normalized_test = test_name.strip().lower()
        self.tests_performed.append(normalized_test)
        logger.info(f"📋 State Change: Test recorded -> '{normalized_test}'")

    def add_question(self, question: str):
        """Record a question that was asked and log it as a state change."""
        self.questions_asked.append(question)
        logger.info(f"📋 State Change: Question recorded -> '{question[:80]}...'")

    def is_stagnating(self, new_action: 'Action') -> bool:
        """Detect if the system is stuck in a loop - addresses Category 1.2"""
        if len(self.last_actions) < 2:
            return False

        # Check if the new action is identical to recent ones
        for last_action in self.last_actions[-2:]:
            if (last_action.action_type == new_action.action_type and
                last_action.content == new_action.content):
                return True
        return False

    def add_action(self, action: 'Action'):
        """Add action to history and maintain sliding window"""
        self.last_actions.append(action)
        if len(self.last_actions) > 3:  # Keep only last 3 actions
            self.last_actions.pop(0)

    def get_max_confidence(self) -> float:
        """Get the maximum confidence from differential diagnosis"""
        if not self.differential_diagnosis:
            return 0.0
        return max(self.differential_diagnosis.values())

    def get_leading_diagnosis(self) -> str:
        """Get the diagnosis with highest confidence"""
        if not self.differential_diagnosis:
            return "No diagnosis formulated"
        return max(self.differential_diagnosis.items(), key=lambda x: x[1])[0]

    def summarize_evidence(self) -> str:
        """Create a concise summary of evidence for token efficiency"""
        if len(self.evidence_log) <= 3:  # More aggressive summarization
            return "\n".join(self.evidence_log)

        # Keep first 2 and last 3 entries, summarize middle
        summary_parts = []
        summary_parts.extend(self.evidence_log[:2])

        if len(self.evidence_log) > 5:
            middle_count = len(self.evidence_log) - 5
            summary_parts.append(f"[... {middle_count} additional findings ...]")

        summary_parts.extend(self.evidence_log[-3:])
        return "\n".join(summary_parts)


@dataclass
class DeliberationState:
    """Structured state for panel deliberation - addresses Category 1.1"""
    hypothesis_analysis: str = ""
    test_chooser_analysis: str = ""
    challenger_analysis: str = ""
    stewardship_analysis: str = ""
    checklist_analysis: str = ""
    situational_context: str = ""
    stagnation_detected: bool = False
    retry_count: int = 0

    def to_consensus_prompt(self, retry_instruction: str = "") -> str:
        """Generate a structured prompt for the consensus coordinator - no truncation, let agent self-regulate"""

        prompt = f"""
You are the Consensus Coordinator. Here is the panel's analysis:

**Differential Diagnosis (Dr. Hypothesis):**
{self.hypothesis_analysis or 'Not yet formulated'}

**Test Recommendations (Dr. Test-Chooser):**
{self.test_chooser_analysis or 'None provided'}

**Critical Challenges (Dr. Challenger):**
{self.challenger_analysis or 'None identified'}

**Cost Assessment (Dr. Stewardship):**
{self.stewardship_analysis or 'Not evaluated'}

**Quality Control (Dr. Checklist):**
{self.checklist_analysis or 'No issues noted'}
"""
        if retry_instruction:
            prompt += f"\n**SUPERVISOR INSTRUCTION:** {retry_instruction}\n"

        if self.stagnation_detected:
            prompt += "\n**STAGNATION DETECTED** - The panel is repeating actions. You MUST make a decisive choice or provide final diagnosis."

        if self.situational_context:
            prompt += f"\n**Situational Context:** {self.situational_context}"

        prompt += "\n\nBased on this comprehensive panel input, use the make_consensus_decision function to provide your structured action."
        return prompt


@dataclass
class DiagnosisResult:
    """Stores the final result of a diagnostic session."""

    final_diagnosis: str
    ground_truth: str
    accuracy_score: float
    accuracy_reasoning: str
    total_cost: int
    iterations: int
    conversation_history: str


class Action(BaseModel):
    """Pydantic model for a structured action decided by the consensus agent."""

    action_type: Literal["ask", "test", "diagnose"] = Field(
        ..., description="The type of action to perform."
    )
    content: Union[str, List[str]] = Field(
        ...,
        description="The content of the action (question, test name, or diagnosis).",
    )
    reasoning: str = Field(
        ..., description="The reasoning behind choosing this action."
    )


# ------------------------------------------------------------------
# Strongly-typed models for function-calling arguments (type safety)
# ------------------------------------------------------------------


class ConsensusArguments(BaseModel):
    """Typed model for the `make_consensus_decision` function call."""

    action_type: Literal["ask", "test", "diagnose"]
    content: Union[str, List[str]]
    reasoning: str


class DifferentialDiagnosisItem(BaseModel):
    """Single differential diagnosis item returned by Dr. Hypothesis."""

    diagnosis: str
    probability: float
    rationale: str
    site: Optional[str] = None  # Added for anatomic specificity


class HypothesisArguments(BaseModel):
    """Typed model for the `update_differential_diagnosis` function call."""

    summary: str
    differential_diagnoses: List[DifferentialDiagnosisItem]
    key_evidence: str
    contradictory_evidence: Optional[str] = None

class JudgeArguments(BaseModel):
    """Typed model for the `provide_judgement` function call."""
    score: float = Field(..., ge=1, le=5)
    justification: str

# --- Main Orchestrator Class ---


class MaiDxOrchestrator:
    """
    Implements the MAI Diagnostic Orchestrator (MAI-DxO) framework.
    This class orchestrates a virtual panel of AI agents to perform sequential medical diagnosis,
    evaluates the final diagnosis, and tracks costs.
    
    Enhanced with structured deliberation and proper state management as per research paper.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",  # Match the primary model used in the research paper
        max_iterations: int = 10,
        initial_budget: int = 10000,
        mode: str = "no_budget",  # "instant", "question_only", "budgeted", "no_budget", "ensemble"
        physician_visit_cost: int = 300,
        enable_budget_tracking: bool = False,
        interactive: bool = False, # Add this line
        request_delay: float = 8.0,  # seconds to wait between model calls to mitigate rate-limits
    ):
        """
        Initializes the MAI-DxO system with improved architecture.

        Args:
            model_name (str): The language model to be used by all agents.
            max_iterations (int): The maximum number of diagnostic loops.
            initial_budget (int): The starting budget for diagnostic tests.
            mode (str): The operational mode of MAI-DxO.
            physician_visit_cost (int): Cost per physician visit.
            enable_budget_tracking (bool): Whether to enable budget tracking.
            interactive (bool): Whether to run in interactive mode, pausing for user input.
            request_delay (float): Seconds to wait between model calls to mitigate rate-limits.
        """
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.initial_budget = initial_budget
        self.mode = mode
        self.physician_visit_cost = physician_visit_cost
        self.enable_budget_tracking = enable_budget_tracking
        self.interactive = interactive # Add this line
        self.request_delay = max(request_delay, 0)
        self.max_total_tokens_per_request = 25000
        self.cumulative_cost = 0
        self.differential_diagnosis = "Not yet formulated."
        self.conversation = Conversation(
            time_enabled=True, autosave=False, save_enabled=False
        )
        self.case_state = None
        self.last_result = None  # Add this line to store the last result
        self.test_cost_db = {
            "default": 150, "cbc": 50, "complete blood count": 50, "complete blood count (cbc) with differential": 50,
            "fbc": 50, "chest x-ray": 200, "chest xray": 200, "xray": 200, "radiograph": 200, "mri": 1500,
            "mri brain": 1800, "mri neck": 1600, "magnetic resonance imaging": 1500, "ct scan": 1200,
            "ct chest": 1300, "ct abdomen": 1400, "computed tomography": 1200, "biopsy": 800, "core biopsy": 900,
            "fine needle aspiration": 800, "fna": 800, "fna biopsy": 800, "immunohistochemistry": 400, "ihc": 400,
            "fish test": 500, "fish": 500, "ultrasound": 300, "ultrasound of neck": 300, "ecg": 100, "ekg": 100,
            "blood glucose": 30, "liver function tests": 80, "renal function": 70, "toxic alcohol panel": 200,
            "urinalysis": 40, "culture": 150, "pathology": 600,
        }
        self._init_agents()
        logger.info(
            f"🥼 MAI Diagnostic Orchestrator initialized successfully in '{mode}' mode with budget ${initial_budget:,}"
        )
        self.agent_prompts = self._load_agent_prompts()

    def _get_agent_max_tokens(self, role: AgentRole) -> int:
        """Get max_tokens for each agent based on their role - agents will self-regulate based on token guidance"""
        token_limits = {
            # Reasonable limits - agents will adjust their verbosity based on token guidance
            AgentRole.HYPOTHESIS: 1200,   # Function calling keeps this structured, but allow room for quality
            AgentRole.TEST_CHOOSER: 800,  # Need space for test rationale
            AgentRole.CHALLENGER: 800,    # Need space for critical analysis
            AgentRole.STEWARDSHIP: 600,
            AgentRole.CHECKLIST: 400,
            AgentRole.CONSENSUS: 500,     # Function calling is efficient
            AgentRole.GATEKEEPER: 1000,  # Needs to provide detailed clinical findings
            AgentRole.JUDGE: 700,
        }
        return token_limits.get(role, 600)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for English)"""
        return len(text) // 4

    def _generate_token_guidance(self, input_tokens: int, max_output_tokens: int, total_tokens: int, agent_role: AgentRole) -> str:
        """Generate dynamic token guidance for agents to self-regulate their responses"""

        # Determine urgency level based on token usage
        if total_tokens > self.max_total_tokens_per_request:
            urgency = "CRITICAL"
            strategy = "Be extremely concise. Prioritize only the most essential information."
        elif total_tokens > self.max_total_tokens_per_request * 0.8:
            urgency = "HIGH"
            strategy = "Be concise and focus on key points. Avoid elaborate explanations."
        elif total_tokens > self.max_total_tokens_per_request * 0.6:
            urgency = "MODERATE"
            strategy = "Be reasonably concise while maintaining necessary detail."
        else:
            urgency = "LOW"
            strategy = "You can provide detailed analysis within your allocated tokens."

        # Role-specific guidance
        role_specific_guidance = {
            AgentRole.HYPOTHESIS: "Focus on top 2-3 diagnoses with probabilities. Prioritize summary over detailed pathophysiology.",
            AgentRole.TEST_CHOOSER: "Recommend 1-2 highest-yield tests. Focus on which hypotheses they'll help differentiate.",
            AgentRole.CHALLENGER: "Identify 1-2 most critical biases or alternative diagnoses. Be direct and specific.",
            AgentRole.STEWARDSHIP: "Focus on cost-effectiveness assessment. Recommend cheaper alternatives where applicable.",
            AgentRole.CHECKLIST: "Provide concise quality check. Flag critical issues only.",
            AgentRole.CONSENSUS: "Function calling enforces structure. Focus on clear reasoning.",
            AgentRole.GATEKEEPER: "Provide specific clinical findings. Be factual and complete but not verbose.",
            AgentRole.JUDGE: "Provide score and focused justification. Be systematic but concise."
        }.get(agent_role, "Be concise and focused.")

        guidance = f"""
[TOKEN MANAGEMENT - {urgency} PRIORITY]
Input: {input_tokens} tokens | Your Output Limit: {max_output_tokens} tokens | Total: {total_tokens} tokens
Strategy: {strategy}
Role Focus: {role_specific_guidance}

IMPORTANT: Adjust your response length and detail level based on this guidance. Prioritize the most critical information for your role.
"""

        return guidance

    def _init_agents(self) -> None:
        """Initializes all required agents with their specific roles and prompts."""

        # Define the structured output tool for consensus decisions
        consensus_tool = {
            "type": "function",
            "function": {
                "name": "make_consensus_decision",
                "description": "Make a structured consensus decision for the next diagnostic action",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_type": {
                            "type": "string",
                            "enum": ["ask", "test", "diagnose"],
                            "description": "The type of action to perform"
                        },
                        "content": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}}
                            ],
                            "description": "The specific content of the action (question(s), test name(s), or diagnosis)"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "The detailed reasoning behind this decision, synthesizing panel input"
                        }
                    },
                    "required": ["action_type", "content", "reasoning"]
                }
            }
        }

        # Define structured output tool for differential diagnosis
        hypothesis_tool = {
            "type": "function",
            "function": {
                "name": "update_differential_diagnosis",
                "description": "Update the differential diagnosis with structured probabilities and reasoning",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "One-sentence summary of primary diagnostic conclusion and confidence"
                        },
                        "differential_diagnoses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "diagnosis": {"type": "string", "description": "The diagnosis name"},
                                    "probability": {"type": "number", "minimum": 0, "maximum": 1, "description": "Probability as decimal (0.0-1.0)"},
                                    "rationale": {"type": "string", "description": "Brief rationale for this diagnosis"}
                                },
                                "required": ["diagnosis", "probability", "rationale"]
                            },
                            "minItems": 2,
                            "maxItems": 5,
                            "description": "Top 2-5 differential diagnoses with probabilities"
                        },
                        "key_evidence": {
                            "type": "string",
                            "description": "Key supporting evidence for leading hypotheses"
                        },
                        "contradictory_evidence": {
                            "type": "string",
                            "description": "Critical contradictory evidence that must be addressed"
                        }
                    },
                    "required": ["summary", "differential_diagnoses", "key_evidence"]
                }
            }
        }
        
        # Define structured output tool for the Judge
        judge_tool = {
            "type": "function",
            "function": {
                "name": "provide_judgement",
                "description": "Provide a structured score and justification for the diagnosis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 1, "maximum": 5, "description": "The clinical accuracy score from 1 to 5."},
                        "justification": {"type": "string", "description": "The detailed reasoning for the assigned score."}
                    },
                    "required": ["score", "justification"]
                }
            }
        }


        self.agents = {}
        for role in AgentRole:
            if role == AgentRole.CONSENSUS:
                # FIX: Force the agent to call this specific function for reliability.
                self.agents[role] = Agent(
                    agent_name=role.value,
                    system_prompt=self._get_prompt_for_role(role),
                    model_name=self.model_name,
                    max_loops=1,
                    tools_list_dictionary=[consensus_tool],
                    tool_choice={"type": "function", "function": {"name": "make_consensus_decision"}},
                    print_on=False,
                    max_tokens=self._get_agent_max_tokens(role),
                    output_raw_json_from_tool_call=True,
                    function_calling_format_type="OpenAI",
                    output_type="json",
                )
            elif role == AgentRole.HYPOTHESIS:
                # FIX: Force the agent to call this specific function for reliability.
                self.agents[role] = Agent(
                    agent_name=role.value,
                    system_prompt=self._get_prompt_for_role(role),
                    model_name=self.model_name,
                    max_loops=1,
                    tools_list_dictionary=[hypothesis_tool],
                    tool_choice={"type": "function", "function": {"name": "update_differential_diagnosis"}},
                    print_on=False,
                    max_tokens=self._get_agent_max_tokens(role),
                    output_raw_json_from_tool_call=True,
                    function_calling_format_type="OpenAI",
                    output_type="json",
                )
            elif role == AgentRole.JUDGE:
                # FIX: Force the agent to call this specific function for reliability.
                self.agents[role] = Agent(
                    agent_name=role.value,
                    system_prompt=self._get_prompt_for_role(role),
                    model_name=self.model_name,
                    max_loops=1,
                    tools_list_dictionary=[judge_tool],
                    tool_choice={"type": "function", "function": {"name": "provide_judgement"}},
                    print_on=False,
                    max_tokens=self._get_agent_max_tokens(role),
                    output_raw_json_from_tool_call=True,
                    function_calling_format_type="OpenAI",
                    output_type="json",
                )
            else:
                # Regular agents without function calling
                self.agents[role] = Agent(
                    agent_name=role.value,
                    system_prompt=self._get_prompt_for_role(role),
                    model_name=self.model_name,
                    max_loops=1,
                    output_type="str",
                    print_on=False,
                    max_tokens=self._get_agent_max_tokens(role),
                )

    def _get_dynamic_context(self, role: AgentRole, case_state: CaseState) -> str:
        """Generate dynamic context for agents based on current situation - addresses Category 4.2"""
        remaining_budget = self.initial_budget - case_state.cumulative_cost

        # Calculate confidence from differential diagnosis
        max_confidence = max(case_state.differential_diagnosis.values()) if case_state.differential_diagnosis else 0

        context = ""

        if role == AgentRole.STEWARDSHIP and remaining_budget < 1000:
            context = f"""
**SITUATIONAL CONTEXT: URGENT**
The remaining budget is critically low (${remaining_budget}). All recommendations must be focused on maximum cost-effectiveness. Veto any non-essential or high-cost tests.
"""

        elif role == AgentRole.HYPOTHESIS and max_confidence > 0.75:
            context = f"""
**SITUATIONAL CONTEXT: FINAL STAGES**
The panel is converging on a diagnosis (current max confidence: {max_confidence:.0%}). Your primary role now is to confirm the leading hypothesis or state what single piece of evidence is needed to reach >85% confidence.
"""

        elif role == AgentRole.CONSENSUS and case_state.iteration > 5:
            context = f"""
**SITUATIONAL CONTEXT: EXTENDED CASE**
This case has gone through {case_state.iteration} iterations. Focus on decisive actions that will lead to a definitive diagnosis rather than additional exploratory steps.
"""

        return context

    def _get_prompt_for_role(self, role: AgentRole, case_state: CaseState = None) -> str:
        """Returns the system prompt for a given agent role with dynamic context."""

        # Add dynamic context if case_state is provided
        dynamic_context = ""
        if case_state:
            dynamic_context = self._get_dynamic_context(role, case_state)

        # --- Compact, token-efficient prompts ---
        base_prompts = {
            AgentRole.HYPOTHESIS: f"""{dynamic_context}

MANDATE: Keep an up-to-date, probability-ranked differential.

DIRECTIVES:
1. Return 2-5 diagnoses (prob 0-1) with 1-line rationale.
2. List key supporting & contradictory evidence.
3. Include anatomic location/site from evidence (e.g., 'of the pharynx') in diagnoses.

You MUST call update_differential_diagnosis().""",

            AgentRole.TEST_CHOOSER: f"""{dynamic_context}

MANDATE: Pick the highest-yield tests.

DIRECTIVES:
1. Suggest ≤3 tests that best separate current diagnoses.
2. Note target hypothesis & info-gain vs cost.

Limit: focus on top 1-2 critical points.""",

            AgentRole.CHALLENGER: f"""{dynamic_context}

MANDATE: Expose the biggest flaw or bias.

DIRECTIVES:
1. Name the key bias/contradiction.
2. Propose an alternate diagnosis or falsifying test.

Reply concisely (top 2 issues).""",

            AgentRole.STEWARDSHIP: f"""{dynamic_context}

MANDATE: Ensure cost-effective care.

DIRECTIVES:
1. Rate proposed tests (High/Mod/Low value).
2. Suggest cheaper equivalents where possible.

Be brief; highlight savings.""",

            AgentRole.CHECKLIST: f"""{dynamic_context}

MANDATE: Guarantee quality & consistency.

DIRECTIVES:
1. Flag invalid tests or logic gaps.
2. Note safety concerns.

Return bullet list of critical items.""",

            AgentRole.CONSENSUS: f"""{dynamic_context}

MANDATE: Decide the next action.

DECISION RULES:
1. If confidence >85% & no major objection → diagnose.
2. Else address Challenger's top concern.
3. Else order highest info-gain (cheapest) test.
4. Else ask the most informative question.

You MUST call make_consensus_decision().""",
        }

        # Use existing prompts for other roles, just add dynamic context
        if role not in base_prompts:
            return dynamic_context + self._get_original_prompt_for_role(role)

        return base_prompts[role]

    def _get_original_prompt_for_role(self, role: AgentRole) -> str:
        """Returns original system prompts for roles not yet updated"""
        prompts = {
            AgentRole.HYPOTHESIS: (
                """
                You are Dr. Hypothesis, a specialist in maintaining differential diagnoses. Your role is critical to the diagnostic process.

                CORE RESPONSIBILITIES:
                - Maintain a probability-ranked differential diagnosis with the top 3 most likely conditions
                - Update probabilities using Bayesian reasoning after each new finding
                - Consider both common and rare diseases appropriate to the clinical context
                - Explicitly track how new evidence changes your diagnostic thinking

                APPROACH:
                1. Start with the most likely diagnoses based on presenting symptoms
                2. For each new piece of evidence, consider:
                   - How it supports or refutes each hypothesis
                   - Whether it suggests new diagnoses to consider
                   - How it changes the relative probabilities
                3. Always explain your Bayesian reasoning clearly

                OUTPUT FORMAT:
                Provide your updated differential diagnosis with:
                - Top 3 diagnoses with probability estimates (percentages)
                - Brief rationale for each
                - Key evidence supporting each hypothesis
                - Evidence that contradicts or challenges each hypothesis

                Remember: Your differential drives the entire diagnostic process. Be thorough, evidence-based, and adaptive.
                """
            ),
            AgentRole.TEST_CHOOSER: (
                """
                You are Dr. Test-Chooser, a specialist in diagnostic test selection and information theory.

                CORE RESPONSIBILITIES:
                - Select up to 3 diagnostic tests per round that maximally discriminate between leading hypotheses
                - Optimize for information value, not just clinical reasonableness
                - Consider test characteristics: sensitivity, specificity, positive/negative predictive values
                - Balance diagnostic yield with patient burden and resource utilization

                SELECTION CRITERIA:
                1. Information Value: How much will this test change diagnostic probabilities?
                2. Discriminatory Power: How well does it distinguish between competing hypotheses?
                3. Clinical Impact: Will the result meaningfully alter management?
                4. Sequential Logic: What should we establish first before ordering more complex tests?

                APPROACH:
                - For each proposed test, explicitly state which hypotheses it will help confirm or exclude
                - Consider both positive and negative results and their implications
                - Think about test sequences (e.g., basic labs before advanced imaging)
                - Avoid redundant tests that won't add new information

                OUTPUT FORMAT:
                For each recommended test:
                - Test name (be specific)
                - Primary hypotheses it will help evaluate
                - Expected information gain
                - How results will change management decisions

                Focus on tests that will most efficiently narrow the differential diagnosis.
                """
            ),
            AgentRole.CHALLENGER: (
                """
                You are Dr. Challenger, the critical thinking specialist and devil's advocate.

                CORE RESPONSIBILITIES:
                - Identify and challenge cognitive biases in the diagnostic process
                - Highlight contradictory evidence that might be overlooked
                - Propose alternative hypotheses and falsifying tests
                - Guard against premature diagnostic closure

                COGNITIVE BIASES TO WATCH FOR:
                1. Anchoring: Over-reliance on initial impressions
                2. Confirmation bias: Seeking only supporting evidence
                3. Availability bias: Overestimating probability of recently seen conditions
                4. Representativeness: Ignoring base rates and prevalence
                5. Search satisficing: Stopping at "good enough" explanations

                YOUR APPROACH:
                - Ask "What else could this be?" and "What doesn't fit?"
                - Challenge assumptions and look for alternative explanations
                - Propose tests that could disprove the leading hypothesis
                - Consider rare diseases when common ones don't fully explain the picture
                - Advocate for considering multiple conditions simultaneously

                OUTPUT FORMAT:
                - Specific biases you've identified in the current reasoning
                - Evidence that contradicts the leading hypotheses
                - Alternative diagnoses to consider
                - Tests that could falsify current assumptions
                - Red flags or concerning patterns that need attention

                Be constructively critical - your role is to strengthen diagnostic accuracy through rigorous challenge.
                """
            ),
            AgentRole.STEWARDSHIP: (
                """
                You are Dr. Stewardship, the resource optimization and cost-effectiveness specialist.

                CORE RESPONSIBILITIES:
                - Enforce cost-conscious, high-value care
                - Advocate for cheaper alternatives when diagnostically equivalent
                - Challenge low-yield, expensive tests
                - Balance diagnostic thoroughness with resource stewardship

                COST-VALUE FRAMEWORK:
                1. High-Value Tests: Low cost, high diagnostic yield, changes management
                2. Moderate-Value Tests: Moderate cost, specific indication, incremental value
                3. Low-Value Tests: High cost, low yield, minimal impact on decisions
                4. No-Value Tests: Any cost, no diagnostic value, ordered out of habit

                ALTERNATIVE STRATEGIES:
                - Could patient history/physical exam provide this information?
                - Is there a less expensive test with similar diagnostic value?
                - Can we use a staged approach (cheap test first, expensive if needed)?
                - Does the test result actually change management?

                YOUR APPROACH:
                - Review all proposed tests for necessity and value
                - Suggest cost-effective alternatives
                - Question tests that don't clearly advance diagnosis
                - Advocate for asking questions before ordering expensive tests
                - Consider the cumulative cost burden

                OUTPUT FORMAT:
                - Assessment of proposed tests (high/moderate/low/no value)
                - Specific cost-effective alternatives
                - Questions that might obviate need for testing
                - Recommended modifications to testing strategy
                - Cumulative cost considerations

                Your goal: Maximum diagnostic accuracy at minimum necessary cost.
                """
            ),
            AgentRole.CHECKLIST: (
                """
                You are Dr. Checklist, the quality assurance and consistency specialist.

                CORE RESPONSIBILITIES:
                - Perform silent quality control on all panel deliberations
                - Ensure test names are valid and properly specified
                - Check internal consistency of reasoning across panel members
                - Flag logical errors or contradictions in the diagnostic approach

                QUALITY CHECKS:
                1. Test Validity: Are proposed tests real and properly named?
                2. Logical Consistency: Do the recommendations align with the differential?
                3. Evidence Integration: Are all findings being considered appropriately?
                4. Process Adherence: Is the panel following proper diagnostic methodology?
                5. Safety Checks: Are any critical possibilities being overlooked?

                SPECIFIC VALIDATIONS:
                - Test names match standard medical terminology
                - Proposed tests are appropriate for the clinical scenario
                - No contradictions between different panel members' reasoning
                - All significant findings are being addressed
                - No gaps in the diagnostic logic

                OUTPUT FORMAT:
                - Brief validation summary (✓ Clear / ⚠ Issues noted)
                - Any test name corrections needed
                - Logical inconsistencies identified
                - Missing considerations or gaps
                - Process improvement suggestions

                Keep your feedback concise but comprehensive. Flag any issues that could compromise diagnostic quality.
                """
            ),
            AgentRole.CONSENSUS: (
                """
                You are the Consensus Coordinator, responsible for synthesizing the virtual panel's expertise into a single, optimal decision.

                CORE RESPONSIBILITIES:
                - Integrate input from Dr. Hypothesis, Dr. Test-Chooser, Dr. Challenger, Dr. Stewardship, and Dr. Checklist
                - Decide on the single best next action: 'ask', 'test', or 'diagnose'
                - Balance competing priorities: accuracy, cost, efficiency, and thoroughness
                - Ensure the chosen action advances the diagnostic process optimally

                DECISION FRAMEWORK:
                1. DIAGNOSE: Choose when diagnostic certainty is sufficiently high (>85%) for the leading hypothesis
                2. TEST: Choose when tests will meaningfully discriminate between hypotheses
                3. ASK: Choose when history/exam questions could provide high-value information

                SYNTHESIS PROCESS:
                - Weight Dr. Hypothesis's confidence level and differential
                - Consider Dr. Test-Chooser's information value analysis
                - Incorporate Dr. Challenger's alternative perspectives
                - Respect Dr. Stewardship's cost-effectiveness concerns
                - Address any quality issues raised by Dr. Checklist

                OUTPUT REQUIREMENTS:
                Provide a JSON object with this exact structure:
                {
                   "action_type": "ask" | "test" | "diagnose",
                   "content": "specific question(s), test name(s), or final diagnosis",
                   "reasoning": "clear justification synthesizing panel input"
                }

                For action_type "ask": content should be specific patient history or physical exam questions
                For action_type "test": content should be properly named diagnostic tests (up to 3)
                For action_type "diagnose": content should be the complete, specific final diagnosis

                Make the decision that best advances accurate, cost-effective diagnosis.
                """
            ),
            AgentRole.GATEKEEPER: (
                """
                You are the Gatekeeper, the clinical information oracle with complete access to the patient case file.

                CORE RESPONSIBILITIES:
                - Provide objective, specific clinical findings when explicitly requested
                - Serve as the authoritative source for all patient information
                - Generate realistic synthetic findings for tests not in the original case
                - Maintain clinical realism while preventing information leakage

                RESPONSE PRINCIPLES:
                1. OBJECTIVITY: Provide only factual findings, never interpretations or impressions
                2. SPECIFICITY: Give precise, detailed results when tests are properly ordered
                3. REALISM: Ensure all responses reflect realistic clinical scenarios
                4. NO HINTS: Never provide diagnostic clues or suggestions
                5. CONSISTENCY: Maintain coherence across all provided information

                HANDLING REQUESTS:
                - Patient History Questions: Provide relevant history from case file or realistic details
                - Physical Exam: Give specific examination findings as would be documented
                - Diagnostic Tests: Provide exact results as specified or realistic synthetic values
                - Vague Requests: Politely ask for more specific queries
                - Invalid Requests: Explain why the request cannot be fulfilled

                SYNTHETIC FINDINGS GUIDELINES:
                When generating findings not in the original case:
                - Ensure consistency with established diagnosis and case details
                - Use realistic reference ranges and values
                - Maintain clinical plausibility
                - Avoid pathognomonic findings unless specifically diagnostic

                RESPONSE FORMAT:
                - Direct, clinical language
                - Specific measurements with reference ranges when applicable
                - Clear organization of findings
                - Professional medical terminology

                Your role is crucial: provide complete, accurate clinical information while maintaining the challenge of the diagnostic process.
                """
            ),
            AgentRole.JUDGE: (
                """
                You are the Judge, the diagnostic accuracy evaluation specialist.

                CORE RESPONSIBILITIES:
                - Evaluate candidate diagnoses against ground truth using a rigorous clinical rubric
                - Provide fair, consistent scoring based on clinical management implications
                - Consider diagnostic substance over terminology differences
                - Account for acceptable medical synonyms and equivalent formulations

                EVALUATION RUBRIC (5-point Likert scale):

                SCORE 5 (Perfect/Clinically Superior):
                - Clinically identical to reference diagnosis
                - May be more specific than reference (adding relevant detail)
                - No incorrect or unrelated additions
                - Treatment approach would be identical

                SCORE 4 (Mostly Correct - Minor Incompleteness):
                - Core disease correctly identified
                - Minor qualifier or component missing/mis-specified
                - Overall management largely unchanged
                - Clinically appropriate diagnosis

                SCORE 3 (Partially Correct - Major Error):
                - Correct general disease category
                - Major error in etiology, anatomic site, or critical specificity
                - Would significantly alter workup or prognosis
                - Partially correct but clinically concerning gaps

                SCORE 2 (Largely Incorrect):
                - Shares only superficial features with correct diagnosis
                - Wrong fundamental disease process
                - Would misdirect clinical workup
                - Partially contradicts case details

                SCORE 1 (Completely Incorrect):
                - No meaningful overlap with correct diagnosis
                - Wrong organ system or disease category
                - Would likely lead to harmful care
                - Completely inconsistent with clinical presentation

                EVALUATION PROCESS:
                1. Compare core disease entity
                2. Assess etiology/causative factors
                3. Evaluate anatomic specificity
                4. Consider diagnostic completeness
                5. Judge clinical management implications

                OUTPUT FORMAT:
                - Score (1-5) with clear label
                - Detailed justification referencing specific rubric criteria
                - Explanation of how diagnosis would affect clinical management
                - Note any acceptable medical synonyms or equivalent terminology

                Maintain high standards while recognizing legitimate diagnostic variability in medical practice.
                """
            ),
        }
        return prompts[role]

    def _parse_json_response(self, response: str, retry_count: int = 0) -> Dict[str, Any]:
        """Safely parses a JSON string with retry logic - addresses Category 3.2"""
        try:
            # Handle agent response wrapper - extract actual content
            if isinstance(response, dict):
                # Handle swarms Agent response format
                if 'role' in response and 'content' in response:
                    response = response['content']
                elif 'content' in response:
                    response = response['content']
                else:
                    # Try to extract any string value from dict
                    response = str(response)
            elif hasattr(response, 'content'):
                response = response.content
            elif not isinstance(response, str):
                # Convert to string if it's some other type
                response = str(response)

            # Extract the actual response content from the agent response
            if isinstance(response, str):
                # Handle markdown-formatted JSON
                if "```json" in response:
                    # Extract JSON content between ```json and ```
                    start_marker = "```json"
                    end_marker = "```"
                    start_idx = response.find(start_marker)
                    if start_idx != -1:
                        start_idx += len(start_marker)
                        end_idx = response.find(end_marker, start_idx)
                        if end_idx != -1:
                            json_content = response[
                                start_idx:end_idx
                            ].strip()
                            return json.loads(json_content)

                # Try to find JSON-like content in the response
                lines = response.split("\n")
                json_lines = []
                in_json = False
                brace_count = 0

                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line.startswith("{") and not in_json:
                        in_json = True
                        json_lines = [line]  # Start fresh
                        brace_count = line.count("{") - line.count(
                            "}"
                        )
                    elif in_json:
                        json_lines.append(line)
                        brace_count += line.count("{") - line.count(
                            "}"
                        )
                        if (
                            brace_count <= 0
                        ):  # Balanced braces, end of JSON
                            break

                if json_lines and in_json:
                    json_content = "\n".join(json_lines)
                    return json.loads(json_content)

                # Try to extract JSON from text that might contain other content
                import re

                # Look for JSON pattern in the text - more comprehensive regex
                json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                matches = re.findall(json_pattern, response, re.DOTALL)

                for match in matches:
                    try:
                        parsed = json.loads(match)
                        # Validate that it has the expected action structure
                        if isinstance(parsed, dict) and 'action_type' in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        continue

                # Direct parsing attempt as fallback
                try:
                    return json.loads(response)
                except json.JSONDecodeError as e:
                    # --- Fallback Sanitization ---
                    # Attempt to strip any leading table/frame characters (e.g., │, ╭, ╰) that may wrap each line
                    try: # Use find() to avoid ValueError
                        start_curly = response.find('{')
                        end_curly = response.rfind('}')
                        if start_curly != -1 and end_curly != -1 and end_curly > start_curly:
                            candidate = response[start_curly:end_curly + 1]
                            sanitized_lines = []
                            for line in candidate.splitlines():
                                # Remove common frame characters and leading whitespace
                                line = line.lstrip('│|╭╰╯├─┤ ').rstrip('│|╭╰╯├─┤ ')
                                sanitized_lines.append(line)
                            candidate_clean = '\n'.join(sanitized_lines)
                            return json.loads(candidate_clean)
                    except Exception:
                        # Still failing, raise original error to trigger retry logic
                        try:
                            # --- Ultimate Fallback: Regex extraction ---
                            import re
                            atype = re.search(r'"action_type"\s*:\s*"(ask|test|diagnose)"', response, re.IGNORECASE)
                            content_match = re.search(r'"content"\s*:\s*"([^"]+?)"', response, re.IGNORECASE | re.DOTALL)
                            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+?)"', response, re.IGNORECASE | re.DOTALL)
                            if atype and content_match and reasoning_match:
                                return {
                                    "action_type": atype.group(1).lower(),
                                    "content": content_match.group(1).strip(),
                                    "reasoning": reasoning_match.group(1).strip(),
                                }
                        except Exception:
                            pass
                        raise e

        except (
            json.JSONDecodeError,
            IndexError,
            AttributeError,
        ) as e:
            logger.error(f"Failed to parse JSON response. Error: {e}")
            logger.debug(
                f"Response content: {response[:500]}..."
            )  # Log first 500 chars

            # Return the error for potential retry instead of immediately falling back
            raise e

    def _parse_json_with_retry(self, consensus_agent: Agent, consensus_prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Parse JSON with retry logic for robustness - addresses Category 3.2"""
        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    response = consensus_agent.run(consensus_prompt)
                else:
                    # Retry with error feedback
                    retry_prompt = f"""
{consensus_prompt}

**CRITICAL: RETRY REQUIRED - ATTEMPT {attempt + 1}**
Your previous response could not be parsed as JSON. You MUST respond with ONLY a valid JSON object in exactly this format:

{{
   "action_type": "ask" | "test" | "diagnose",
   "content": "your content here",
   "reasoning": "your reasoning here"
}}

Do NOT include any other text, markdown formatting, or explanations. Only the raw JSON object.
NO SYSTEM MESSAGES, NO WRAPPER FORMAT. JUST THE JSON.
"""
                    response = consensus_agent.run(retry_prompt)

                # Handle different response types from swarms Agent
                response_text = ""
                if hasattr(response, 'content'):
                    response_text = response.content
                elif isinstance(response, dict):
                    # Handle swarms Agent response wrapper
                    if 'role' in response and 'content' in response:
                        response_text = response['content']
                    elif 'content' in response:
                        response_text = response['content']
                    else:
                        response_text = str(response)
                elif isinstance(response, str):
                    response_text = response
                else:
                    response_text = str(response)

                # Log the response for debugging
                logger.debug(f"Parsing attempt {attempt + 1}, response type: {type(response)}")
                logger.debug(f"Response content preview: {str(response_text)[:200]}...")

                return self._parse_json_response(response_text, attempt)

            except Exception as e:
                logger.warning(f"JSON parsing attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    # Final fallback after all retries
                    logger.error("All JSON parsing attempts failed, using fallback")
                    return {
                        "action_type": "ask",
                        "content": "Could you please clarify the next best step? The previous analysis was inconclusive.",
                        "reasoning": f"Fallback due to JSON parsing error after {max_retries + 1} attempts.",
                    }

        # Should never reach here, but just in case
        return {
            "action_type": "ask",
            "content": "Please provide more information about the patient's condition.",
            "reasoning": "Unexpected fallback in JSON parsing.",
        }

    def _estimate_cost(self, tests: Union[List[str], str]) -> int:
        """Estimates the cost of diagnostic tests with enhanced logging for transparency."""
        if isinstance(tests, str):
            tests = [tests]

        total_cost = 0
        for test in tests:
            test_lower = test.lower().strip()
            cost_found = False

            # Strategy 1: Exact match (no log needed for the ideal case)
            if test_lower in self.test_cost_db:
                total_cost += self.test_cost_db[test_lower]
                cost_found = True
                continue

            # Strategy 2: Partial match (find best matching key)
            best_match = None
            best_match_length = 0
            for cost_key in self.test_cost_db:
                if cost_key in test_lower or test_lower in cost_key:
                    if len(cost_key) > best_match_length:
                        best_match = cost_key
                        best_match_length = len(cost_key)

            if best_match:
                estimated_cost = self.test_cost_db[best_match]
                total_cost += estimated_cost
                logger.debug(
                    f"Cost Audit: No exact match for '{test_lower}'. Used partial match '{best_match}' (${estimated_cost})."
                )
                cost_found = True
                continue
            
            # Strategy 3: Keyword-based matching
            keyword_matches = {
                ("biopsy", "tissue"): ("biopsy", 800),
                ("mri", "magnetic"): ("mri", 1500),
                ("ct", "computed tomography"): ("ct scan", 1200),
                ("xray", "x-ray", "radiograph"): ("chest x-ray", 200),
                ("blood", "serum", "plasma"): ("default blood test", 100),
                ("culture", "sensitivity"): ("culture", 150),
                ("immunohistochemistry", "ihc"): ("immunohistochemistry", 400),
            }

            for keywords, (log_key, cost_val) in keyword_matches.items():
                if any(keyword in test_lower for keyword in keywords):
                    estimated_cost = self.test_cost_db.get(log_key, cost_val)
                    total_cost += estimated_cost
                    logger.debug(
                        f"Cost Audit: No exact match for '{test_lower}'. Used keyword match '{log_key}' (${estimated_cost})."
                    )
                    cost_found = True
                    break 
            
            if cost_found:
                continue

            # Strategy 4: Default cost for unknown tests
            if not cost_found:
                default_cost = self.test_cost_db["default"]
                total_cost += default_cost
                logger.debug(
                    f"Cost Audit: No match found for '{test_lower}'. Used default cost (${default_cost})."
                )

        return total_cost

    def _run_panel_deliberation(self, case_state: CaseState) -> Action:
        """Orchestrates one round of structured debate among the virtual panel - addresses Category 1.1"""
        logger.info(
            "🩺 Virtual medical panel deliberation commenced - analyzing patient case"
        )
        logger.debug(
            "Panel members: Dr. Hypothesis, Dr. Test-Chooser, Dr. Challenger, Dr. Stewardship, Dr. Checklist"
        )

        # Initialize structured deliberation state instead of conversational chaining
        deliberation_state = DeliberationState()

        # Prepare concise case context for each agent (token-optimized)
        remaining_budget = self.initial_budget - case_state.cumulative_cost
        budget_status = (
            "EXCEEDED"
            if remaining_budget < 0
            else f"${remaining_budget:,}"
        )

        # Full context - let agents self-regulate based on token guidance
        base_context = f"""
=== DIAGNOSTIC CASE STATUS - ROUND {case_state.iteration} ===

INITIAL PRESENTATION:
{case_state.initial_vignette}

EVIDENCE GATHERED:
{case_state.summarize_evidence()}

CURRENT STATE:
- Tests Performed: {', '.join(case_state.tests_performed) if case_state.tests_performed else 'None'}
- Questions Asked: {len(case_state.questions_asked)}
- Cumulative Cost: ${case_state.cumulative_cost:,}
- Remaining Budget: {budget_status}
- Mode: {self.mode}
        """

        # Check mode-specific constraints
        if self.mode == "instant":
            # For instant mode, skip deliberation and go straight to diagnosis
            action_dict = {
                "action_type": "diagnose",
                "content": case_state.get_leading_diagnosis(),
                "reasoning": (
                    "Instant diagnosis mode - providing immediate assessment based on initial presentation"
                ),
            }
            return Action(**action_dict)

        # Check for stagnation before running deliberation
        stagnation_detected = False
        if len(case_state.last_actions) >= 2:
            last_action = case_state.last_actions[-1]
            # Use a simpler check for stagnation
            if case_state.is_stagnating(last_action):
                   stagnation_detected = True
                   deliberation_state.stagnation_detected = True
                   logger.warning("🔄 Stagnation detected - will force different action")


        # Generate dynamic situational context for all agents
        deliberation_state.situational_context = self._generate_situational_context(case_state, remaining_budget)

        # Run each specialist agent in parallel-like fashion with structured output
        # Each agent gets the same base context plus their role-specific dynamic prompt
        try:
            # Dr. Hypothesis - Differential diagnosis and probability assessment
            logger.info("🧠 Dr. Hypothesis analyzing differential diagnosis...")
            hypothesis_prompt = self._get_prompt_for_role(AgentRole.HYPOTHESIS, case_state) + "\n\n" + base_context
            start_time = time.perf_counter()
            
            # Yield agent status update
            yield {
                "type": "agent_status",
                "agent_id": "hypothesis",
                "status": "thinking"
            }
            
            hypothesis_response = self._safe_agent_run(
                self.agents[AgentRole.HYPOTHESIS], hypothesis_prompt, agent_role=AgentRole.HYPOTHESIS
            )
            duration = time.perf_counter() - start_time
            logger.info(f"⏱️  Agent '{AgentRole.HYPOTHESIS.value}' completed in {duration:.2f}s.")

            logger.debug(f"Raw response from Dr. Hypothesis (pre-parsing): {hypothesis_response}")

            # Update case state with new differential (supports both function calls and text)
            self._update_differential_from_hypothesis(case_state, hypothesis_response)

            # Store the analysis for deliberation state (convert to text format for other agents)
            if hasattr(hypothesis_response, 'content'):
                deliberation_state.hypothesis_analysis = hypothesis_response.content
            else:
                deliberation_state.hypothesis_analysis = str(hypothesis_response)

            # Yield agent update with content
            yield {
                "type": "agent_update",
                "agent": "hypothesis",
                "content": deliberation_state.hypothesis_analysis[:500]  # First 500 chars for UI
            }
            
            # Yield agent status completed
            yield {
                "type": "agent_status",
                "agent_id": "hypothesis", 
                "status": "completed"
            }

            # --- Yield state update for UI ---
            yield {
                "type": "state_update",
                "differential_diagnosis": self.differential_diagnosis,
                "cumulative_cost": case_state.cumulative_cost,
                "iteration": case_state.iteration
            }

            # Dr. Test-Chooser - Information value optimization
            logger.info("🔬 Dr. Test-Chooser selecting optimal tests...")
            test_chooser_prompt = self._get_prompt_for_role(AgentRole.TEST_CHOOSER, case_state) + "\n\n" + base_context
            if self.mode == "question_only":
                test_chooser_prompt += "\n\nIMPORTANT: This is QUESTION-ONLY mode. You may ONLY recommend patient questions, not diagnostic tests."
            start_time = time.perf_counter()
            
            yield {"type": "agent_status", "agent_id": "test_chooser", "status": "thinking"}
            
            deliberation_state.test_chooser_analysis = self._safe_agent_run(
                self.agents[AgentRole.TEST_CHOOSER], test_chooser_prompt, agent_role=AgentRole.TEST_CHOOSER
            )
            duration = time.perf_counter() - start_time
            logger.info(f"⏱️  Agent '{AgentRole.TEST_CHOOSER.value}' completed in {duration:.2f}s.")
            
            yield {
                "type": "agent_update",
                "agent": "test_chooser",
                "content": str(deliberation_state.test_chooser_analysis)[:500]
            }
            yield {"type": "agent_status", "agent_id": "test_chooser", "status": "completed"}

            # Dr. Challenger - Bias identification and alternative hypotheses
            logger.info("🤔 Dr. Challenger challenging assumptions...")
            challenger_prompt = self._get_prompt_for_role(AgentRole.CHALLENGER, case_state) + "\n\n" + base_context
            start_time = time.perf_counter()
            
            yield {"type": "agent_status", "agent_id": "challenger", "status": "thinking"}
            
            deliberation_state.challenger_analysis = self._safe_agent_run(
                self.agents[AgentRole.CHALLENGER], challenger_prompt, agent_role=AgentRole.CHALLENGER
            )
            duration = time.perf_counter() - start_time
            logger.info(f"⏱️  Agent '{AgentRole.CHALLENGER.value}' completed in {duration:.2f}s.")
            
            yield {
                "type": "agent_update", 
                "agent": "challenger",
                "content": str(deliberation_state.challenger_analysis)[:500]
            }
            yield {"type": "agent_status", "agent_id": "challenger", "status": "completed"}

            # Dr. Stewardship - Cost-effectiveness analysis
            logger.info("💰 Dr. Stewardship evaluating cost-effectiveness...")
            stewardship_prompt = self._get_prompt_for_role(AgentRole.STEWARDSHIP, case_state) + "\n\n" + base_context
            if self.enable_budget_tracking:
                stewardship_prompt += f"\n\nBUDGET TRACKING ENABLED - Current cost: ${case_state.cumulative_cost}, Remaining: ${remaining_budget}"
            start_time = time.perf_counter()
            
            yield {"type": "agent_status", "agent_id": "stewardship", "status": "thinking"}
            
            deliberation_state.stewardship_analysis = self._safe_agent_run(
                self.agents[AgentRole.STEWARDSHIP], stewardship_prompt, agent_role=AgentRole.STEWARDSHIP
            )
            duration = time.perf_counter() - start_time
            logger.info(f"⏱️  Agent '{AgentRole.STEWARDSHIP.value}' completed in {duration:.2f}s.")
            
            yield {
                "type": "agent_update",
                "agent": "stewardship", 
                "content": str(deliberation_state.stewardship_analysis)[:500]
            }
            yield {"type": "agent_status", "agent_id": "stewardship", "status": "completed"}

            # Dr. Checklist - Quality assurance
            logger.info("✅ Dr. Checklist performing quality control...")
            checklist_prompt = self._get_prompt_for_role(AgentRole.CHECKLIST, case_state) + "\n\n" + base_context
            start_time = time.perf_counter()
            
            yield {"type": "agent_status", "agent_id": "checklist", "status": "thinking"}
            
            deliberation_state.checklist_analysis = self._safe_agent_run(
                self.agents[AgentRole.CHECKLIST], checklist_prompt, agent_role=AgentRole.CHECKLIST
            )
            duration = time.perf_counter() - start_time
            logger.info(f"⏱️  Agent '{AgentRole.CHECKLIST.value}' completed in {duration:.2f}s.")
            
            yield {
                "type": "agent_update",
                "agent": "checklist",
                "content": str(deliberation_state.checklist_analysis)[:500]
            }
            yield {"type": "agent_status", "agent_id": "checklist", "status": "completed"}

            # Consensus Coordinator - Final decision synthesis using structured state
            logger.info("🤝 Consensus Coordinator synthesizing panel decision...")

            yield {"type": "agent_status", "agent_id": "consensus", "status": "thinking"}

            # Generate the structured consensus prompt
            consensus_prompt = deliberation_state.to_consensus_prompt()

            # Add mode-specific constraints to consensus
            if self.mode == "budgeted" and remaining_budget <= 0:
                consensus_prompt += "\n\nBUDGET CONSTRAINT: Budget exceeded - must either ask questions or provide final diagnosis."

            # Use function calling with retry logic for robust structured output
            action_dict = self._get_consensus_with_retry(consensus_prompt)
            
            yield {"type": "agent_status", "agent_id": "consensus", "status": "completed"}

            # Validate action based on mode constraints
            action = Action(**action_dict)

            # Apply mode-specific validation and corrections
            action = self._validate_and_correct_action(action, case_state, remaining_budget, deliberation_state, consensus_prompt)

            return action

        except Exception as e:
            logger.error(f"Error during panel deliberation: {e}", exc_info=True)
            # Fallback action
            return Action(
                action_type="ask",
                content="Could you please provide more information about the patient's current condition?",
                reasoning=f"Fallback due to panel deliberation error: {str(e)}",
            )

    def _generate_situational_context(self, case_state: CaseState, remaining_budget: int) -> str:
        """Generate dynamic situational context based on current case state - addresses Category 4.2"""
        context_parts = []

        # Budget-related context
        if remaining_budget < 1000:
            context_parts.append(f"URGENT: Remaining budget critically low (${remaining_budget}). Focus on cost-effective actions.")
        elif remaining_budget < 2000:
            context_parts.append(f"WARNING: Budget running low (${remaining_budget}). Prioritize high-value tests.")

        # Diagnostic confidence context
        max_confidence = case_state.get_max_confidence()
        if max_confidence > 0.85:
            context_parts.append(f"FINAL STAGES: High confidence diagnosis available ({max_confidence:.0%}). Consider definitive action.")
        elif max_confidence > 0.70:
            context_parts.append(f"CONVERGING: Moderate confidence in leading diagnosis ({max_confidence:.0%}). Focus on confirmation.")

        # Iteration context
        if case_state.iteration > 7:
            context_parts.append(f"EXTENDED CASE: {case_state.iteration} rounds completed. Move toward decisive action.")
        elif case_state.iteration > 5:
            context_parts.append(f"PROLONGED: {case_state.iteration} rounds. Avoid further exploratory steps unless critical.")

        # Test/cost context
        if len(case_state.tests_performed) > 5:
            context_parts.append("EXTENSIVE TESTING: Many tests completed. Focus on synthesis rather than additional testing.")

        return " | ".join(context_parts) if context_parts else ""

    def _update_differential_from_hypothesis(self, case_state: CaseState, hypothesis_response):
        """Extract and update differential diagnosis from Dr. Hypothesis analysis - now supports both function calls and text"""
        try:
            processed_response = hypothesis_response
            if isinstance(hypothesis_response, str):
                try:
                    processed_response = json.loads(hypothesis_response)
                except json.JSONDecodeError:
                    # This is not a critical failure; it may just be a text response.
                    logger.debug("Hypothesis response was a string that could not be parsed as JSON. Treating as plain text.")
                    processed_response = None

            structured_data = self._extract_function_call_output(processed_response) if processed_response else None

            if structured_data and "differential_diagnoses" in structured_data:
                validated_data = HypothesisArguments(**structured_data)
                logger.success(f"✅ Tool-Use Success: Agent '{AgentRole.HYPOTHESIS.value}' provided a valid function call.")
                
                new_differential = {dx.diagnosis: dx.probability for dx in validated_data.differential_diagnoses}

                if new_differential:
                    case_state.update_differential(new_differential)
                    dx_text = f"{validated_data.summary}\n\nTop Diagnoses:\n"
                    for dx in validated_data.differential_diagnoses:
                        dx_text += f"- {dx.diagnosis}: {dx.probability:.0%} - {dx.rationale}\n"

                    if validated_data.key_evidence:
                        dx_text += f"\nKey Evidence: {validated_data.key_evidence}"
                    if validated_data.contradictory_evidence:
                        dx_text += f"\nContradictory Evidence: {validated_data.contradictory_evidence}"

                    self.differential_diagnosis = dx_text
                    logger.debug(f"Updated differential from function call: {new_differential}")
                    return

        except ValidationError as e:
            logger.warning(f"Tool-Use Failure: Validation failed for '{AgentRole.HYPOTHESIS.value}'. State will not be updated from this tool call.")
            try:
                error_details = json.loads(e.json())
                logger.debug(f"Pydantic Validation Error Details:\n{json.dumps(error_details, indent=2)}")
            except Exception:
                logger.debug(f"Raw Pydantic Validation Error: {e}")

        except (KeyError, TypeError) as e:
             logger.warning(f"Tool-Use Failure: Could not parse structured differential from '{AgentRole.HYPOTHESIS.value}': {e}. State will not be updated.")

        # Fallback to treating the response as plain text
        hypothesis_text = str(hypothesis_response)
        if hasattr(hypothesis_response, 'content'):
            hypothesis_text = hypothesis_response.content
        self.differential_diagnosis = hypothesis_text

    def _validate_and_correct_action(self, action: Action, case_state: CaseState, remaining_budget: int, deliberation_state: DeliberationState, original_consensus_prompt: str) -> Action:
        """
        Validate and correct actions based on mode constraints and context.
        BUGFIX: Correctly handles redundant test detection and properly uses the
        newly generated action from a retry loop.
        """

        # Interactive UX: avoid instant diagnosis on the very first round so
        # clinicians always see a question/test proposal and can respond.
        if self.interactive and action.action_type == "diagnose" and case_state.iteration <= 1:
            logger.info("Interactive mode: preventing immediate diagnosis on round 1. Converting to an ASK for clinician input.")
            action.action_type = "ask"
            action.content = "Please provide more details from history or initial labs to guide the next step."
            action.reasoning = "Interactive mode policy: obtain at least one piece of clinician-provided information before diagnosing."
            return action

        # Mode-specific validations
        if self.mode == "question_only" and action.action_type == "test":
            logger.warning("Test ordering attempted in question-only mode, converting to ask action")
            action.action_type = "ask"
            action.content = "Can you provide more details about the patient's symptoms and history?"
            action.reasoning = "Mode constraint: question-only mode active"
            return action

        if self.mode == "budgeted" and action.action_type == "test" and remaining_budget <= 0:
            logger.warning("Test ordering attempted with insufficient budget, converting to diagnose action")
            action.action_type = "diagnose"
            action.content = case_state.get_leading_diagnosis()
            action.reasoning = "Budget constraint: insufficient funds for additional testing"
            return action

        # BUGFIX: Implement active supervisor logic to prevent indecisiveness.
        # Rule A: Forbid redundant testing.
        if action.action_type == "test":
            # --- FIX #1: Normalize the proposed test content correctly, handling both string and list ---
            proposed_tests = action.content if isinstance(action.content, list) else [action.content]
            normalized_proposed_tests = {str(t).strip().lower() for t in proposed_tests}

            # --- FIX #2: Check if ANY of the proposed tests are already in the performed list ---
            performed_tests_set = set(case_state.tests_performed)
            redundant_tests = normalized_proposed_tests.intersection(performed_tests_set)

            if redundant_tests:
                # --- FIX #3: Log the ACTUAL redundant test for clarity, not a stale variable ---
                redundant_test_name = next(iter(redundant_tests))
                logger.warning(f"SUPERVISOR: Consensus agent proposed a redundant test ('{redundant_test_name}'). Requesting a new action.")
                
                # Re-run consensus with a one-time instruction to break the loop
                retry_instruction = f"CRITICAL ERROR: You proposed the test '{action.content}', which has already been performed. Re-evaluate the panel's input and propose a new, different, non-redundant action."
                retry_prompt = deliberation_state.to_consensus_prompt(retry_instruction=retry_instruction)
                action_dict = self._get_consensus_with_retry(retry_prompt)

                # --- FIX #4: CRUCIAL - Create and return the NEW action immediately. ---
                # This prevents the old, invalid action from continuing through the logic flow.
                new_action = Action(**action_dict)
                logger.info(f"SUPERVISOR: Corrected action is now -> {new_action.action_type.upper()}: {new_action.content}")
                return new_action

        # Rule B: Intelligent Stagnation Fallback.
        if deliberation_state.stagnation_detected:
            logger.warning("Stagnation detected by orchestrator.")
            # Check if the action proposed by consensus is ALSO the same as the last one
            is_new_action_stagnant = (len(case_state.last_actions) > 0 and
                                      action.action_type == case_state.last_actions[-1].action_type and
                                      action.content == case_state.last_actions[-1].content)

            if is_new_action_stagnant:
                logger.warning("Consensus agent also proposed a stagnant action. Forcing a clinically relevant termination.")
                action.action_type = "diagnose"
                action.content = "Diagnosis stalled after initial tests proved inconclusive. Escalation to advanced diagnostics (e.g., imaging, biopsy) is recommended."
                action.reasoning = "Forced diagnosis: The panel is stuck in a low-yield loop, indicating a need to escalate the diagnostic strategy."
                return action
            else:
                logger.info("Consensus agent proposed a new, non-stagnant action. Proceeding with its decision.")

        # High confidence threshold
        if action.action_type != "diagnose" and case_state.get_max_confidence() > 0.90:
            logger.info("Very high confidence reached, recommending diagnosis")
            action.action_type = "diagnose"
            action.content = case_state.get_leading_diagnosis()
            action.reasoning = "High confidence threshold reached - proceeding to final diagnosis"
        
        return action

    def _interact_with_gatekeeper(
        self, action: Action, full_case_details: str
    ) -> str:
        """Sends the panel's action to the Gatekeeper and returns its response."""
        gatekeeper = self.agents[AgentRole.GATEKEEPER]

        if action.action_type == "ask":
            request = f"Question: {action.content}"
        elif action.action_type == "test":
            tests = action.content if isinstance(action.content, list) else [action.content]
            request = f"Tests ordered: {', '.join(tests)}"
        else:
            return "No interaction needed for 'diagnose' action."

        # The Gatekeeper needs the full case to act as an oracle
        prompt = f"""
        Full Case Details (for your reference only):
        ---
        {full_case_details}
        ---
        
        Request from Diagnostic Agent:
        {request}
        """

        logger.debug(f"Full prompt being sent to Gatekeeper:\n{prompt}")

        response = self._safe_agent_run(gatekeeper, prompt, agent_role=AgentRole.GATEKEEPER)
        return response

    def _judge_diagnosis(
        self, candidate_diagnosis: str, ground_truth: str
    ) -> Dict[str, Any]:
        """Uses the Judge agent to evaluate the final diagnosis using structured function calling."""
        judge = self.agents[AgentRole.JUDGE]

        prompt = f"""
        Please evaluate the following diagnosis against the ground truth using the `provide_judgement` function.
        Ground Truth: "{ground_truth}"
        Candidate Diagnosis: "{candidate_diagnosis}"

        You MUST call the `provide_judgement` function to submit your evaluation.
        """
        response = self._safe_agent_run(judge, prompt, agent_role=AgentRole.JUDGE)
        logger.debug(f"Raw Judge response: {response}")
        logger.debug(f"Type of response variable for Judge: {type(response)}")

        try:
            processed_response = response
            if isinstance(response, str):
                try:
                    processed_response = json.loads(response)
                except json.JSONDecodeError:
                    raise ValueError("Judge response was a string but could not be parsed as JSON.")

            judgement_dict = self._extract_function_call_output(processed_response)

            if not judgement_dict:
                raise ValueError("Could not extract structured judgement from Judge agent's response.")

            validated_args = JudgeArguments(**judgement_dict)
            logger.success(f"✅ Tool-Use Success: Agent '{AgentRole.JUDGE.value}' provided a valid function call.")
            score = validated_args.score
            reasoning = validated_args.justification

        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"Tool-Use Failure: Could not parse structured response from '{AgentRole.JUDGE.value}'. Error: {e}")
            if isinstance(e, ValidationError):
                try:
                    error_details = json.loads(e.json())
                    logger.debug(f"Pydantic Validation Error Details:\n{json.dumps(error_details, indent=2)}")
                except Exception:
                    logger.debug(f"Raw Pydantic Validation Error: {e}")

            logger.debug(f"Raw Judge response causing error: {response}")
            score = 0.0
            reasoning = f"Could not parse judge's structured response: {str(e)}"

        logger.info(f"Judge evaluation: Score={score}, Reasoning preview: {reasoning[:100]}...")
        return {"score": score, "reasoning": reasoning}

    def run_gen(self, initial_case_info: str, full_case_details: str, ground_truth_diagnosis: str):
        """
        Generator wrapper around run() that forwards streaming updates (logs, state updates,
        pauses) and relays clinician inputs back into the underlying generator when running
        in interactive mode. When the underlying run() generator completes, this function
        returns the final DiagnosisResult.
        """
        # Initial UI hint
        incoming = yield {"type": "log", "message": "Starting diagnostic session..."}
        try:
            # Start the underlying diagnostic process (run() is a generator when interactive)
            runner = self.run(
                initial_case_info=initial_case_info,
                full_case_details=full_case_details,
                ground_truth_diagnosis=ground_truth_diagnosis,
            )

            # Relay messages from the underlying generator, and pass any input back
            while True:
                try:
                    if incoming is None:
                        update = next(runner)
                    else:
                        update = runner.send(incoming)
                        incoming = None
                    # Yield update to caller and capture any input sent back on resume
                    incoming = yield update
                except StopIteration as e:
                    # Underlying run() completed and returned a DiagnosisResult
                    result = e.value
                    return result
        except Exception as e:
            # Surface any fatal error to the UI stream
            yield {"type": "error", "message": f"Diagnostic run failed: {e}"}
            raise

    def run(
        self,
        initial_case_info: str,
        full_case_details: str,
        ground_truth_diagnosis: str,
    ) -> DiagnosisResult:
        """
        Executes the full sequential diagnostic process with structured state management.
        """
        start_time = time.time()
        self.last_result = None # Reset last result at the start of a run

        # Initialize structured case state
        case_state = CaseState(initial_vignette=initial_case_info)
        
        # Log initial visit cost as a state change
        old_cost = case_state.cumulative_cost
        case_state.cumulative_cost += self.physician_visit_cost
        self.cumulative_cost = case_state.cumulative_cost
        logger.info(f"💰 State Change: Cost updated -> ${old_cost:,} -> ${case_state.cumulative_cost:,} (Initial Visit)")

        self.case_state = case_state

        self.conversation.add(
            "System",
            f"Initial Case Information: {initial_case_info}",
        )
        case_state.add_evidence(f"Initial presentation: {initial_case_info}", source="System")

        final_diagnosis = None
        iteration_count = 0

        for i in range(self.max_iterations):
            iteration_count = i + 1
            case_state.iteration = iteration_count

            logger.debug(f"--- LOOP {iteration_count} STATE ---")
            logger.debug(f"Current CaseState: {case_state}")

            logger.info(
                f"--- Starting Diagnostic Loop {iteration_count}/{self.max_iterations} ---"
            )
            logger.info(
                f"Current cost: ${case_state.cumulative_cost:,} | Remaining budget: ${self.initial_budget - case_state.cumulative_cost:,}"
            )

            try:
                # Run the panel deliberation. This may be a generator (streaming
                # state updates) or a direct Action depending on implementation.
                deliberation_result = self._run_panel_deliberation(case_state)

                try:
                    from types import GeneratorType
                    is_generator = isinstance(deliberation_result, GeneratorType)
                except Exception:
                    is_generator = False

                if is_generator:
                    # Forward all yielded updates (e.g., state_update) and capture
                    # the returned Action from the generator's StopIteration.value
                    action = yield from deliberation_result
                else:
                    action = deliberation_result
                logger.info(
                    f"⚕️ Panel decision: {action.action_type.upper()} -> {action.content}"
                )
                logger.info(
                    f"💭 Medical reasoning: {action.reasoning}"
                )

                case_state.add_action(action)

                if action.action_type == "diagnose":
                    final_diagnosis = action.content
                    logger.info(
                        f"Final diagnosis proposed: {final_diagnosis}"
                    )
                    break

                if (
                    self.mode == "question_only"
                    and action.action_type == "test"
                ):
                    logger.warning(
                        "Test ordering blocked in question-only mode"
                    )
                    continue

                if (
                    self.mode == "budgeted"
                    and action.action_type == "test"
                ):
                    estimated_test_cost = self._estimate_cost(
                        action.content
                    )
                    if (
                        case_state.cumulative_cost + estimated_test_cost
                        > self.initial_budget
                    ):
                        logger.warning(
                            f"Test cost ${estimated_test_cost} would exceed budget. Skipping tests."
                        )
                        continue

                # --- Gatekeeper Interaction ---
                if self.interactive:
                    # In interactive mode, yield to the UI and wait for input
                    response = yield {
                        "type": "pause",
                        "action": action.dict(),
                    }
                else:
                    # In autonomous mode, query the AI Gatekeeper
                    response = self._interact_with_gatekeeper(
                        action, full_case_details
                    )

                clean_response = response.content if hasattr(response, 'content') else str(response)
                self.conversation.add("Gatekeeper", clean_response)
                case_state.add_evidence(clean_response, source="Gatekeeper")

                if action.action_type == "test":
                    test_cost = self._estimate_cost(action.content)
                    old_cost = case_state.cumulative_cost
                    case_state.cumulative_cost += test_cost
                    case_state.add_test(str(action.content)) # This now logs the test itself
                    self.cumulative_cost = case_state.cumulative_cost
                    logger.info(f"💰 State Change: Cost updated -> ${old_cost:,} -> ${case_state.cumulative_cost:,} (Test cost: ${test_cost:,})")
                    
                    # Send cost update to frontend
                    if self.interactive:
                        yield {
                            "type": "state_update",
                            "cumulative_cost": case_state.cumulative_cost,
                            "iteration": case_state.iteration
                        }
                
                elif action.action_type == "ask":
                    case_state.add_question(str(action.content)) # This now logs the question
                    logger.debug("No additional cost for questions in same visit")

                if (
                    self.mode == "budgeted"
                    and case_state.cumulative_cost >= self.initial_budget
                ):
                    logger.warning(
                        "Budget limit reached. Forcing final diagnosis."
                    )
                    final_diagnosis = case_state.get_leading_diagnosis()
                    break

            except Exception as e:
                logger.error(
                    f"Error in diagnostic loop {iteration_count}: {e}",
                    exc_info=True
                )
                continue

        else:
            final_diagnosis = case_state.get_leading_diagnosis()
            if final_diagnosis == "No diagnosis formulated":
                final_diagnosis = "Diagnosis not reached within maximum iterations."
            logger.warning(
                f"Max iterations ({self.max_iterations}) reached. Using best available diagnosis."
            )

        if isinstance(final_diagnosis, list):
            final_diagnosis = final_diagnosis[0] if final_diagnosis else ""
            
        if not final_diagnosis or final_diagnosis.strip() == "":
            final_diagnosis = (
                "Unable to determine diagnosis within constraints."
            )

        total_time = time.time() - start_time
        logger.info(
            f"Diagnostic session completed in {total_time:.2f} seconds"
        )

        logger.info("Evaluating final diagnosis...")
        try:
            judgement = self._judge_diagnosis(
                final_diagnosis, ground_truth_diagnosis
            )
        except Exception as e:
            logger.error(f"Error in diagnosis evaluation: {e}", exc_info=True)
            judgement = {
                "score": 0.0,
                "reasoning": f"Evaluation error: {str(e)}",
            }

        result = DiagnosisResult(
            final_diagnosis=final_diagnosis,
            ground_truth=ground_truth_diagnosis,
            accuracy_score=judgement["score"],
            accuracy_reasoning=judgement["reasoning"],
            total_cost=case_state.cumulative_cost,
            iterations=iteration_count,
            conversation_history=self.conversation.get_str(),
        )

        logger.info("--- DIAGNOSTIC SUMMARY ---")
        logger.info(f"  Final diagnosis: {final_diagnosis}")
        logger.info(f"  Ground truth: {ground_truth_diagnosis}")
        logger.info(f"  Accuracy score: {judgement['score']}/5.0")
        logger.info(f"  Total cost: ${case_state.cumulative_cost:,}")
        logger.info(f"  Iterations: {iteration_count}")

        self.last_result = result  # Add this line to save the result
        return result

    def run_ensemble(
        self,
        initial_case_info: str,
        full_case_details: str,
        ground_truth_diagnosis: str,
        num_runs: int = 3,
    ) -> DiagnosisResult:
        """
        Runs multiple independent diagnostic sessions and aggregates the results.

        Args:
            initial_case_info (str): The initial abstract of the case.
            full_case_details (str): The complete case file for the Gatekeeper.
            ground_truth_diagnosis (str): The correct final diagnosis for evaluation.
            num_runs (int): Number of independent runs to perform.

        Returns:
            DiagnosisResult: Aggregated result from ensemble runs.
        """
        logger.info(
            f"Starting ensemble run with {num_runs} independent sessions"
        )

        ensemble_results = []
        total_cost = 0

        for run_id in range(num_runs):
            logger.info(
                f"=== Ensemble Run {run_id + 1}/{num_runs} ==="
            )

            # Create a fresh orchestrator instance for each run
            run_orchestrator = MaiDxOrchestrator(
                model_name=self.model_name,
                max_iterations=self.max_iterations,
                initial_budget=self.initial_budget,
                mode="no_budget",  # Use no_budget for ensemble runs
                physician_visit_cost=self.physician_visit_cost,
                enable_budget_tracking=False,
            )

            # Run the diagnostic session
            result = run_orchestrator.run(
                initial_case_info,
                full_case_details,
                ground_truth_diagnosis,
            )
            ensemble_results.append(result)
            total_cost += result.total_cost

            logger.info(
                f"Run {run_id + 1} completed: {result.final_diagnosis} (Score: {result.accuracy_score})"
            )

        # Aggregate results using consensus
        final_diagnosis = self._aggregate_ensemble_diagnoses(
            [r.final_diagnosis for r in ensemble_results]
        )

        # Judge the aggregated diagnosis
        judgement = self._judge_diagnosis(
            final_diagnosis, ground_truth_diagnosis
        )

        # Calculate average metrics
        avg_iterations = sum(
            r.iterations for r in ensemble_results
        ) / len(ensemble_results)

        # Combine conversation histories
        combined_history = "\n\n=== ENSEMBLE RESULTS ===\n"
        for i, result in enumerate(ensemble_results):
            combined_history += f"\n--- Run {i+1} ---\n"
            combined_history += (
                f"Diagnosis: {result.final_diagnosis}\n"
            )
            combined_history += f"Score: {result.accuracy_score}\n"
            combined_history += f"Cost: ${result.total_cost:,}\n"
            combined_history += f"Iterations: {result.iterations}\n"

        combined_history += "\n--- Aggregated Result ---\n"
        combined_history += f"Final Diagnosis: {final_diagnosis}\n"
        combined_history += f"Reasoning: {judgement['reasoning']}\n"

        ensemble_result = DiagnosisResult(
            final_diagnosis=final_diagnosis,
            ground_truth=ground_truth_diagnosis,
            accuracy_score=judgement["score"],
            accuracy_reasoning=judgement["reasoning"],
            total_cost=total_cost,  # Sum of all runs
            iterations=int(avg_iterations),
            conversation_history=combined_history,
        )

        logger.info(
            f"Ensemble completed: {final_diagnosis} (Score: {judgement['score']})"
        )
        return ensemble_result

    def _aggregate_ensemble_diagnoses(
        self, diagnoses: List[str]
    ) -> str:
        """Aggregates multiple diagnoses from ensemble runs."""
        # Simple majority voting or use the most confident diagnosis
        if not diagnoses:
            return "No diagnosis available"

        # Remove any empty or invalid diagnoses
        valid_diagnoses = [
            d
            for d in diagnoses
            if d and d.strip() and "not reached" not in d.lower() and "unable to establish" not in d.lower()
        ]

        if not valid_diagnoses:
            return diagnoses[0] if diagnoses else "No valid diagnosis"

        # If all diagnoses are the same, return that
        if len(set(valid_diagnoses)) == 1:
            return valid_diagnoses[0]

        # Use an aggregator agent to select the best diagnosis
        try:
            aggregator_prompt = f"""
            You are a medical consensus aggregator. Given multiple diagnostic assessments from independent medical panels, 
            select the most accurate and complete diagnosis.
            
            Diagnoses to consider:
            {chr(10).join(f"{i+1}. {d}" for i, d in enumerate(valid_diagnoses))}
            
            Provide the single best diagnosis that represents the medical consensus. 
            Consider clinical accuracy, specificity, and completeness.
            """

            aggregator = Agent(
                agent_name="Ensemble Aggregator",
                system_prompt=aggregator_prompt,
                model_name=self.model_name,
                max_loops=1,
                print_on=False,
            )

            agg_resp = self._safe_agent_run(aggregator, aggregator_prompt)
            if hasattr(agg_resp, "content"):
                return agg_resp.content.strip()
            return str(agg_resp).strip()

        except Exception as e:
            logger.error(f"Error in ensemble aggregation: {e}")
            # Fallback to most common diagnosis
            from collections import Counter

            return Counter(valid_diagnoses).most_common(1)[0][0]

    @classmethod
    def create_variant(
        cls, variant: str, **kwargs
    ) -> "MaiDxOrchestrator":
        """
        Factory method to create different MAI-DxO variants as described in the paper.

        Args:
            variant (str): One of 'instant', 'question_only', 'budgeted', 'no_budget', 'ensemble'
            **kwargs: Additional parameters for the orchestrator

        Returns:
            MaiDxOrchestrator: Configured orchestrator instance
        """
        variant_configs = {
            "instant": {
                "mode": "instant",
                "max_iterations": 1,
                "enable_budget_tracking": False,
            },
            "question_only": {
                "mode": "question_only",
                "max_iterations": 10,
                "enable_budget_tracking": False,
            },
            "budgeted": {
                "mode": "budgeted",
                "max_iterations": 10,
                "enable_budget_tracking": True,
                "initial_budget": kwargs.get("budget", 5000),  # Fixed: map budget to initial_budget
            },
            "no_budget": {
                "mode": "no_budget",
                "max_iterations": 10,
                "enable_budget_tracking": False,
            },
            "ensemble": {
                "mode": "no_budget",
                "max_iterations": 10,
                "enable_budget_tracking": False,
            },
        }

        if variant not in variant_configs:
            raise ValueError(
                f"Unknown variant: {variant}. Choose from: {list(variant_configs.keys())}"
            )

        config = variant_configs[variant]
        config.update(kwargs)  # Allow overrides

        # Remove 'budget' parameter if present, as it's mapped to 'initial_budget'
        config.pop('budget', None)

        return cls(**config)

    # ------------------------------------------------------------------
    # Helper utilities – throttling & robust JSON parsing
    # ------------------------------------------------------------------

    def _safe_agent_run(
        self,
        agent: "Agent",  # type: ignore – forward reference
        prompt: str,
        retries: int = 3,
        agent_role: AgentRole = None,
    ) -> Any:
        """
        Safely call `agent.run` with enhanced logging, token guidance, and retry logic.
        This version directly inspects the agent's response to log token usage.
        """
        if agent_role is None:
            agent_role = AgentRole.CONSENSUS  # Default fallback

        # Token estimation logic for generating agent guidance remains the same
        estimated_input_tokens = self._estimate_tokens(prompt)
        max_output_tokens = self._get_agent_max_tokens(agent_role)
        total_estimated_tokens = estimated_input_tokens + max_output_tokens
        token_guidance = self._generate_token_guidance(
            estimated_input_tokens, max_output_tokens, total_estimated_tokens, agent_role
        )
        enhanced_prompt = f"{token_guidance}\n\n{prompt}"
        logger.debug(f"Agent {agent_role.value}: Est. Input={estimated_input_tokens}, Max Output={max_output_tokens}")

        base_delay = max(self.request_delay, 5.0)

        for attempt in range(retries + 1):
            current_delay = base_delay * (3 ** attempt) if attempt > 0 else base_delay
            if attempt > 0:
                logger.info(f"Retrying agent call (attempt {attempt + 1}/{retries + 1}), waiting {current_delay:.1f}s...")
            # time.sleep(current_delay) # This was commented out in the original script

            try:
                # Log the intent to call, specifying the model.
                logger.info(f"📞 Calling agent '{agent.agent_name}' (model: '{agent.model_name}')")
                response = agent.run(enhanced_prompt)
                logger.debug(f"Agent {agent_role.value} raw response: {response}")

                # Direct token usage extraction from the response object
                try:
                    usage = None
                    if isinstance(response, list) and len(response) > 0:
                        # The usage dict can be in the last or second-to-last message in the history
                        if response[-1] and isinstance(response[-1], dict) and "usage" in response[-1]:
                            usage = response[-1].get("usage")
                        elif len(response) > 1 and response[-2] and isinstance(response[-2], dict) and "usage" in response[-2]:
                            usage = response[-2].get("usage")

                    if usage and isinstance(usage, dict):
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        logger.info(
                            f"📊 Token Usage for '{agent.model_name}': "
                            f"Prompt={prompt_tokens}, Completion={completion_tokens}, Total={prompt_tokens + completion_tokens}"
                        )
                except Exception as e:
                    logger.debug(f"Could not extract token usage from agent response: {e}")

                return response
            except (
                litellm.InternalServerError,
                litellm.RateLimitError,
                Exception,
            ) as e:
                # More specific error logging to distinguish API vs. other errors
                err_msg = str(e).lower()
                if "rate_limit" in err_msg or "ratelimiterror" in err_msg or "429" in str(e):
                    logger.warning(
                        f"API Error: Rate limit encountered for '{agent.model_name}' (attempt {attempt + 1}/{retries + 1}). "
                        f"Will retry..."
                    )
                    continue  # Continue to the next attempt
                
                # For non-rate-limit errors, log with full traceback and propagate immediately
                logger.error(
                    f"Model/Content Error: An unexpected error occurred calling '{agent.model_name}' for agent '{agent.agent_name}'.",
                    exc_info=True  # Adds full traceback to the log for detailed debugging
                )
                raise

        # All retries exhausted, provide a more informative error
        raise RuntimeError(f"Maximum retries exceeded for agent '{agent.agent_name}' using model '{agent.model_name}'.")

    def _extract_function_call_output(self, raw_response: Any) -> Optional[Dict[str, Any]]:
        """
        FINAL FIX: A robust parser that correctly handles the swarms Agent's
        conversation history list to find and extract the tool call.
        """
        logger.debug(f"Attempting to extract function call from: {raw_response}")
        try:
            # The agent's response is always a list representing the conversation history.
            if not isinstance(raw_response, list) or not raw_response:
                logger.debug("Response is not a list or is empty.")
                return None

            # The tool call is in the last message of the history.
            last_message = raw_response[-1]
            if not isinstance(last_message, dict):
                logger.debug("Last item in history is not a dictionary.")
                return None

            # The actual tool call is inside the 'content' field, which is a list.
            content_list = last_message.get('content')
            if not isinstance(content_list, list) or not content_list:
                logger.debug("The 'content' of the last message is not a list or is empty.")
                return None

            # The first item in the content list is the tool call dictionary.
            tool_call_data = content_list[0]
            if not isinstance(tool_call_data, dict) or "function" not in tool_call_data:
                logger.debug("First item in content list is not a valid tool call dictionary.")
                return None
            
            # The arguments are a stringified JSON inside the 'function' dictionary.
            arguments_str = tool_call_data.get("function", {}).get("arguments")
            if not arguments_str or not isinstance(arguments_str, str):
                logger.debug("No arguments string found in the function call.")
                return None

            # Parse the stringified JSON to get the final arguments dictionary.
            return json.loads(arguments_str)

        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse tool call from raw response: {e}")
            logger.debug(f"Problematic raw response for parsing: {raw_response}")
            return None

    def _get_consensus_with_retry(self, consensus_prompt: str, max_retries: int = 2) -> Dict[str, Any]:
        """Get consensus decision with function call retry logic and enhanced logging."""
        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    # First attempt - use original prompt
                    response = self._safe_agent_run(
                        self.agents[AgentRole.CONSENSUS], consensus_prompt, agent_role=AgentRole.CONSENSUS
                    )
                else:
                    # Retry with explicit function call instruction
                    retry_prompt = f"""
{consensus_prompt}

**CRITICAL: RETRY ATTEMPT {attempt + 1}**
Your previous response did not properly use the make_consensus_decision function.
You MUST call the make_consensus_decision function with these exact parameters:
- action_type: "ask", "test", or "diagnose"
- content: specific question(s), test name(s), or diagnosis (string or array of strings)
- reasoning: your detailed reasoning

DO NOT respond with plain text. You MUST use the function call.
"""
                    response = self._safe_agent_run(
                        self.agents[AgentRole.CONSENSUS], retry_prompt, agent_role=AgentRole.CONSENSUS
                    )

                processed_response = response
                if isinstance(response, str):
                    try:
                        processed_response = json.loads(response)
                        logger.debug("Successfully parsed string response into a Python object.")
                    except json.JSONDecodeError:
                        logger.warning("Consensus response was a string but failed to parse as JSON. Proceeding with raw string.")

                response_preview = str(processed_response)[:500] if processed_response else "None"
                logger.debug(f"Consensus attempt {attempt + 1}, response type: {type(processed_response)}, preview: {response_preview}")

                action_dict = self._extract_function_call_output(processed_response)
                
                if not action_dict:
                    logger.warning(f"Tool-Use Failure: Could not extract 'make_consensus_decision' function call on attempt {attempt + 1}")
                    continue

                # Validate and enforce schema using ConsensusArguments for type safety
                validated_args = ConsensusArguments(**action_dict)
                logger.success(f"✅ Tool-Use Success: Agent '{AgentRole.CONSENSUS.value}' provided a valid function call on attempt {attempt + 1}.")
                return validated_args.dict()

            except ValidationError as e:
                logger.warning(f"Tool-Use Failure: Validation failed for '{AgentRole.CONSENSUS.value}' on attempt {attempt + 1}. Will retry.")
                # Pydantic's e.json() gives a structured error report. We can pretty-print it for clarity.
                try:
                    error_details = json.loads(e.json())
                    logger.debug(f"Pydantic Validation Error Details:\n{json.dumps(error_details, indent=2)}")
                except Exception:
                    logger.debug(f"Raw Pydantic Validation Error: {e}") # Fallback for raw error
                
                if attempt < max_retries:
                    continue
            except (KeyError, TypeError) as e:
                logger.warning(f"Tool-Use Failure: Error processing function call from '{AgentRole.CONSENSUS.value}' on attempt {attempt + 1}: {e}. Will retry.")
                if attempt < max_retries:
                    continue

        # Final fallback
        logger.error(f"All {max_retries + 1} tool-use attempts by '{AgentRole.CONSENSUS.value}' failed. Final fallback.")
        logger.critical("CRITICAL: Generating a fallback action. The agent's structured output is unparsable. This indicates a critical system failure.")
        
        return {
            "action_type": "ask",
            "content": "Could you please provide more information to guide the diagnostic process?",
            "reasoning": f"Final fallback after all function call attempts failed."
        }

    @staticmethod
    def _load_agent_prompts():
        config_path = Path(__file__).parent / "agent_configs.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return {
                role.value: MaiDxOrchestrator._get_default_prompt(role) for role in AgentRole
            }

    @staticmethod
    def _get_default_prompt(role: AgentRole) -> str:
        base_prompts = {
            AgentRole.HYPOTHESIS: """MANDATE: Keep an up-to-date, probability-ranked differential.

DIRECTIVES:
1. Return 2-5 diagnoses (prob 0-1) with 1-line rationale.
2. List key supporting & contradictory evidence.
3. Include anatomic location/site from evidence (e.g., 'of the pharynx') in diagnoses.

You MUST call update_differential_diagnosis().""",

            AgentRole.TEST_CHOOSER: """MANDATE: Pick the highest-yield tests.

DIRECTIVES:
1. Suggest ≤3 tests that best separate current diagnoses.
2. Note target hypothesis & info-gain vs cost.

Limit: focus on top 1-2 critical points.""",

            AgentRole.CHALLENGER: """MANDATE: Expose the biggest flaw or bias.

DIRECTIVES:
1. Name the key bias/contradiction.
2. Propose an alternate diagnosis or falsifying test.

Reply concisely (top 2 issues).""",

            AgentRole.STEWARDSHIP: """MANDATE: Ensure cost-effective care.

DIRECTIVES:
1. Rate proposed tests (High/Mod/Low value).
2. Suggest cheaper equivalents where possible.

Be brief; highlight savings.""",

            AgentRole.CHECKLIST: """MANDATE: Guarantee quality & consistency.

DIRECTIVES:
1. Flag invalid tests or logic gaps.
2. Note safety concerns.

Return bullet list of critical items.""",

            AgentRole.CONSENSUS: """MANDATE: Synthesize panel input into a cohesive decision.

DIRECTIVES:
1. Select top 1-3 diagnoses with evidence summary.
2. Recommend next actions with rationale.
3. Rate confidence (High/Medium/Low).

You MUST call make_consensus_decision().""",

            AgentRole.GATEKEEPER: """MANDATE: Control information release.

DIRECTIVES:
1. Reveal ONLY directly relevant facts.
2. Maintain clinical realism.

Limit to essentials.""",

            AgentRole.JUDGE: """MANDATE: Score diagnostic accuracy.

DIRECTIVES:
1. Rate 1-5 vs ground truth.
2. Explain matches/mismatches.

You MUST call provide_judgement()."""
        }
        return base_prompts.get(role, "")

    @staticmethod
    def update_agent_prompt(agent_name: str, new_prompt: str):
        config_path = Path(__file__).parent / "agent_configs.json"
        configs = MaiDxOrchestrator._load_agent_prompts()
        configs[agent_name] = new_prompt
        with open(config_path, 'w') as f:
            json.dump(configs, f, indent=4)

    @staticmethod
    def get_agent_prompts():
        return MaiDxOrchestrator._load_agent_prompts()
        
def run_mai_dxo_demo(
    case_info: str = None,
    case_details: str = None,
    ground_truth: str = None,
) -> Dict[str, DiagnosisResult]:
    """
    Convenience function to run a quick demonstration of MAI-DxO variants.

    Args:
        case_info (str): Initial case information. Uses default if None.
        case_details (str): Full case details. Uses default if None.
        ground_truth (str): Ground truth diagnosis. Uses default if None.

    Returns:
        Dict[str, DiagnosisResult]: Results from different MAI-DxO variants
    """
    # Use default case if not provided
    if not case_info:
        case_info = (
            "A 29-year-old woman was admitted to the hospital because of sore throat and peritonsillar swelling "
            "and bleeding. Symptoms did not abate with antimicrobial therapy."
        )

    if not case_details:
        case_details = """
        Patient: 29-year-old female.
        History: Onset of sore throat 7 weeks prior to admission. Worsening right-sided pain and swelling.
        No fevers, headaches, or gastrointestinal symptoms. Past medical history is unremarkable.
        Physical Exam: Right peritonsillar mass, displacing the uvula. No other significant findings.
        Initial Labs: FBC, clotting studies normal.
        MRI Neck: Showed a large, enhancing mass in the right peritonsillar space.
        Biopsy (H&E): Infiltrative round-cell neoplasm with high nuclear-to-cytoplasmic ratio and frequent mitotic figures.
        Biopsy (Immunohistochemistry): Desmin and MyoD1 diffusely positive. Myogenin multifocally positive.
        Biopsy (FISH): No FOXO1 (13q14) rearrangements detected.
        Final Diagnosis from Pathology: Embryonal rhabdomyosarcoma of the pharynx.
        """

    if not ground_truth:
        ground_truth = "Embryonal rhabdomyosarcoma of the pharynx"

    results = {}

    # Test key variants
    variants = ["no_budget", "budgeted", "question_only"]

    for variant in variants:
        try:
            logger.info(f"Running MAI-DxO variant: {variant}")

            if variant == "budgeted":
                orchestrator = MaiDxOrchestrator.create_variant(
                    variant,
                    budget=3000,
                    model_name="gemini/gemini-2.5-flash",  # Fixed: Use valid model name
                    max_iterations=3,
                )
            else:
                orchestrator = MaiDxOrchestrator.create_variant(
                    variant,
                    model_name="gemini/gemini-2.5-flash",  # Fixed: Use valid model name
                    max_iterations=3,
                )

            result = orchestrator.run(
                case_info, case_details, ground_truth
            )
            results[variant] = result

        except Exception as e:
            logger.error(f"Error running variant {variant}: {e}")
            results[variant] = None

    return results


if __name__ == "__main__":
    # Example case inspired by the paper's Figure 1
    initial_info = (
        "A 29-year-old woman was admitted to the hospital because of sore throat and peritonsillar swelling "
        "and bleeding. Symptoms did not abate with antimicrobial therapy."
    )

    full_case = """
    Patient: 29-year-old female.
    History: Onset of sore throat 7 weeks prior to admission. Worsening right-sided pain and swelling.
    No fevers, headaches, or gastrointestinal symptoms. Past medical history is unremarkable. No history of smoking or significant alcohol use.
    Physical Exam: Right peritonsillar mass, displacing the uvula. No other significant findings.
    Initial Labs: FBC, clotting studies normal.
    MRI Neck: Showed a large, enhancing mass in the right peritonsillar space.
    Biopsy (H&E): Infiltrative round-cell neoplasm with high nuclear-to-cytoplasmic ratio and frequent mitotic figures.
    Biopsy (Immunohistochemistry for Carcinoma): CD31, D2-40, CD34, ERG, GLUT-1, pan-cytokeratin, CD45, CD20, CD3 all negative. Ki-67: 60% nuclear positivity.
    Biopsy (Immunohistochemistry for Rhabdomyosarcoma): Desmin and MyoD1 diffusely positive. Myogenin multifocally positive.
    Biopsy (FISH): No FOXO1 (13q14) rearrangements detected.
    Final Diagnosis from Pathology: Embryonal rhabdomyosarcoma of the pharynx.
    """

    ground_truth = "Embryonal rhabdomyosarcoma of the pharynx"

    # --- Demonstrate Different MAI-DxO Variants ---
    try:
        print("\n" + "=" * 80)
        print(
            "    MAI DIAGNOSTIC ORCHESTRATOR (MAI-DxO) - SEQUENTIAL DIAGNOSIS BENCHMARK"
        )
        print(
            "                      Implementation based on the NEJM Research Paper"
        )
        print("=" * 80)

        # Test different variants as described in the paper
        variants_to_test = [
            (
                "no_budget",
                "Standard MAI-DxO with no budget constraints",
            ),
            # ("budgeted", "Budget-constrained MAI-DxO ($3000 limit)"),
            # (
            #     "question_only",
            #     "Question-only variant (no diagnostic tests)",
            # ),
        ]

        results = {}

        for variant_name, description in variants_to_test:
            print(f"\n{'='*60}")
            print(f"Testing Variant: {variant_name.upper()}")
            print(f"Description: {description}")
            print("=" * 60)

            # Create the variant
            if variant_name == "budgeted":
                orchestrator = MaiDxOrchestrator.create_variant(
                    variant_name,
                    budget=3000,
                    model_name="gpt-4o",
                    max_iterations=5, # Increased iterations
                )
            else:
                orchestrator = MaiDxOrchestrator.create_variant(
                    variant_name,
                    model_name="gpt-4o",
                    max_iterations=5, # Increased iterations
                )

            # Run the diagnostic process
            orchestrator.run(
                initial_case_info=initial_info,
                full_case_details=full_case,
                ground_truth_diagnosis=ground_truth,
            )
            result = orchestrator.last_result
            results[variant_name] = result

            # Display results
            if result:
                print(f"\n🚀 Final Diagnosis: {result.final_diagnosis}")
                print(f"🎯 Ground Truth: {result.ground_truth}")
                print(f"⭐ Accuracy Score: {result.accuracy_score}/5.0")
                print(f"   Reasoning: {result.accuracy_reasoning}")
                print(f"💰 Total Cost: ${result.total_cost:,}")
                print(f"🔄 Iterations: {result.iterations}")
                print(f"⏱️  Mode: {orchestrator.mode}")
            else:
                print("\n❌ Error: Orchestrator failed to produce a result.")


        # Demonstrate ensemble approach
        # print(f"\n{'='*60}")
        # print("Testing Variant: ENSEMBLE")
        # print(
        #     "Description: Multiple independent runs with consensus aggregation"
        # )
        # print("=" * 60)

        # ensemble_orchestrator = MaiDxOrchestrator.create_variant(
        #     "ensemble",
        #     model_name="gpt-4o",
        #     max_iterations=3,
        # )

        # ensemble_orchestrator.run_ensemble(
        #     initial_case_info=initial_info,
        #     full_case_details=full_case,
        #     ground_truth_diagnosis=ground_truth,
        #     num_runs=2,
        # )
        # ensemble_result = ensemble_orchestrator.last_result
        # results["ensemble"] = ensemble_result

        # if ensemble_result:
        #     print(
        #         f"\n🚀 Ensemble Diagnosis: {ensemble_result.final_diagnosis}"
        #     )
        #     print(f"🎯 Ground Truth: {ensemble_result.ground_truth}")
        #     print(
        #         f"⭐ Ensemble Score: {ensemble_result.accuracy_score}/5.0"
        #     )
        #     print(
        #         f"💰 Total Ensemble Cost: ${ensemble_result.total_cost:,}"
        #     )

        # --- Summary Comparison ---
        print(f"\n{'='*80}")
        print("                               RESULTS SUMMARY")
        print("=" * 80)
        print(
            f"{'Variant':<15} {'Diagnosis Match':<15} {'Score':<8} {'Cost':<12} {'Iterations':<12}"
        )
        print("-" * 80)

        for variant_name, result in results.items():
            if result:
                match_status = (
                    "✓ Match"
                    if result.accuracy_score >= 4.0
                    else "✗ No Match"
                )
                print(
                    f"{variant_name:<15} {match_status:<15} {result.accuracy_score:<8.1f} ${result.total_cost:<11,} {result.iterations:<12}"
                )
            else:
                print(f"{variant_name:<15} {'ERROR':<15} {'N/A':<8} {'N/A':<12} {'N/A':<12}")


        print(f"\n{'='*80}")
        print(
            "Implementation successfully demonstrates the MAI-DxO framework"
        )
        print(
            "as described in 'Sequential Diagnosis with Language Models' paper"
        )
        print("=" * 80)

    except Exception as e:
        logger.exception(
            f"An error occurred during the diagnostic session: {e}"
        )
        print(f"\n❌ Error occurred: {e}")
        print("Please check your model configuration and API keys.")