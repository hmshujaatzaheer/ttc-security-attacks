"""
TTC Security Attacks Module

This module provides attack implementations for test-time compute mechanisms
in reasoning LLMs.

Attacks:
    - NLBAAttack: Natural Language Blindness Attack on PRMs
    - SMVAAttack: Single-Model Voting Attack on self-consistency
    - MVNAAttack: MCTS Value Network Attack on tree search
"""

from .base_attack import BaseAttack, AttackResult, AttackStrategy
from .nlba import NLBAAttack
from .smva import SMVAAttack
from .mvna import MVNAAttack

__all__ = [
    "BaseAttack",
    "AttackResult",
    "AttackStrategy",
    "NLBAAttack",
    "SMVAAttack",
    "MVNAAttack",
]
