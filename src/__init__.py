"""
TTC Security Attacks

Mechanistic Attacks on Test-Time Compute in Reasoning LLMs

This package provides attack implementations and evaluation tools for
testing the security of test-time compute mechanisms:
    - Process Reward Models (PRMs)
    - Self-Consistency Voting
    - MCTS/Tree Search

Example:
    >>> from ttc_security_attacks import NLBAAttack, TTCSecBenchmark
    >>> attack = NLBAAttack()
    >>> result = attack.attack(problem="...", correct_answer="...", target_wrong_answer="...")
"""

__version__ = "0.1.0"
__author__ = "H M Shujaat Zaheer"
__email__ = "shujabis@gmail.com"

from .attacks import NLBAAttack, SMVAAttack, MVNAAttack, AttackResult
from .evaluation import TTCSecBenchmark, compute_metrics

__all__ = [
    "NLBAAttack",
    "SMVAAttack", 
    "MVNAAttack",
    "AttackResult",
    "TTCSecBenchmark",
    "compute_metrics",
]
