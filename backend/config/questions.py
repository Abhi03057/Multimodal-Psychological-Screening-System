"""
Guided Session Question Configuration
======================================
Defines the structured prompts used during a guided live screening session.
Each question targets a specific emotional domain to enable reactivity analysis.

Category semantics:
  - baseline  → neutral conversational state (used as reference)
  - stress    → stress-inducing prompt (compared against baseline)
  - anxiety   → anxiety-specific prompt (grouped with stress for reactivity)
  - positive  → positive recall (used for emotional range assessment)
"""

from typing import List, Dict

# ── Guided prompts served to the frontend ────────────────────────────────────
GUIDED_QUESTIONS: List[Dict[str, str]] = [
    {
        "id": "baseline",
        "prompt": "Can you briefly introduce yourself and describe your day?",
        "category": "baseline",
    },
    {
        "id": "stress",
        "prompt": "Can you describe a recent situation that made you feel stressed or overwhelmed?",
        "category": "stress",
    },
    {
        "id": "anxiety",
        "prompt": "What are the things that worry you the most these days?",
        "category": "anxiety",
    },
    {
        "id": "positive",
        "prompt": "Can you talk about something that made you feel happy recently?",
        "category": "positive",
    },
]

# ── Grouping for reactivity computation ──────────────────────────────────────
# Baseline group: emotionally neutral prompts
BASELINE_GROUP = {"baseline"}

# Stress group: prompts expected to elevate emotional arousal
STRESS_GROUP = {"stress", "anxiety"}

# All valid question types (for validation)
VALID_QUESTION_TYPES = {"baseline", "stress", "anxiety", "positive", "unknown"}
