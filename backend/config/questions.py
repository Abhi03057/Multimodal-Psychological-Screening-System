"""
Guided Session Question Configuration
======================================
Defines the structured prompts used during a guided live screening session.
"""
from typing import List, Dict

GUIDED_QUESTIONS: List[Dict[str, str]] = [
    {"id": 0, "prompt": "Can you briefly introduce yourself and describe your day?", "category": "baseline"},
    {"id": 1, "prompt": "Can you describe a recent situation that made you feel stressed or overwhelmed?", "category": "stress"},
    {"id": 2, "prompt": "What are the things that worry you the most these days?", "category": "anxiety"},
    {"id": 3, "prompt": "Can you talk about something that made you feel happy recently?", "category": "positive"},
]

BASELINE_GROUP = frozenset({"baseline"})
STRESS_GROUP   = frozenset({"stress", "anxiety"})
VALID_QUESTION_TYPES = frozenset({"baseline", "stress", "anxiety", "positive", "unknown"})
