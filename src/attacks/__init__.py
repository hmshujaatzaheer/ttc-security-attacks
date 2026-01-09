"""
TTC Security Attacks - Attack Implementations

This module provides implementations of three mechanistic attacks on
test-time compute mechanisms in reasoning LLMs:

- NLBA: Natural Language Blindness Attack on PRMs
- SMVA: Single-Model Voting Attack on self-consistency
- MVNA: MCTS Value Network Attack on tree search
"""

from .base_attack import BaseAttack, AttackResult, AttackConfig
from .nlba import NLBAAttack, NLBAConfig
from .smva import SMVAAttack, SMVAConfig
from .mvna import MVNAAttack, MVNAConfig

__all__ = [
    # Base classes
    "BaseAttack",
    "AttackResult",
    "AttackConfig",
    # NLBA
    "NLBAAttack",
    "NLBAConfig",
    # SMVA
    "SMVAAttack",
    "SMVAConfig",
    # MVNA
    "MVNAAttack",
    "MVNAConfig",
]

__version__ = "0.1.0"
