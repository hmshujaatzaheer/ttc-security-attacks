"""
Base Attack Class for TTC Security Attacks

This module provides the abstract base class for all attack implementations.
All attacks (NLBA, SMVA, MVNA) inherit from this class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import logging
import time

import torch
import numpy as np

logger = logging.getLogger(__name__)


class AttackStrategy(Enum):
    """Enumeration of attack strategies."""
    # NLBA Strategies
    NL_BLINDNESS = "nl_blindness"
    OOD_DIFFICULTY = "ood_difficulty"
    GRADIENT_INJECTION = "gradient_injection"
    
    # SMVA Strategies
    SAMPLING_BIAS = "sampling_bias"
    PARSE_EXPLOIT = "parse_exploit"
    EARLY_STOP = "early_stop"
    
    # MVNA Strategies
    VALUE_FOOLING = "value_fooling"
    UCT_MANIPULATION = "uct_manipulation"
    EXPANSION_BIAS = "expansion_bias"


@dataclass
class AttackResult:
    """Container for attack results."""
    success: bool
    problem: str
    target_answer: str
    achieved_answer: str
    
    # Attack-specific metrics
    prm_score: Optional[float] = None
    vote_flip_rate: Optional[float] = None
    path_deviation: Optional[float] = None
    
    # Trace information
    trace: Optional[List[str]] = None
    adversarial_prompt: Optional[str] = None
    
    # Timing
    attack_time: float = 0.0
    iterations: int = 0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "problem": self.problem,
            "target_answer": self.target_answer,
            "achieved_answer": self.achieved_answer,
            "prm_score": self.prm_score,
            "vote_flip_rate": self.vote_flip_rate,
            "path_deviation": self.path_deviation,
            "trace": self.trace,
            "adversarial_prompt": self.adversarial_prompt,
            "attack_time": self.attack_time,
            "iterations": self.iterations,
            "metadata": self.metadata
        }


class BaseAttack(ABC):
    """
    Abstract base class for all TTC security attacks.
    
    All attack implementations (NLBA, SMVA, MVNA) must inherit from this class
    and implement the required abstract methods.
    
    Attributes:
        name: Human-readable name of the attack
        device: Torch device for computation
        config: Attack configuration dictionary
        verbose: Whether to print progress information
    """
    
    def __init__(
        self,
        name: str,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """
        Initialize the base attack.
        
        Args:
            name: Name of the attack (e.g., "NLBA", "SMVA", "MVNA")
            device: Device to use ("cuda", "cpu", or specific GPU)
            config: Configuration dictionary
            verbose: Whether to print progress
        """
        self.name = name
        self.verbose = verbose
        self.config = config or {}
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Attack statistics
        self._attack_count = 0
        self._success_count = 0
        
        logger.info(f"Initialized {self.name} attack on device: {self.device}")
    
    @abstractmethod
    def attack(
        self,
        problem: str,
        correct_answer: str,
        target_wrong_answer: str,
        strategy: Optional[str] = None,
        **kwargs
    ) -> AttackResult:
        """
        Execute the attack on a given problem.
        
        Args:
            problem: The problem statement to attack
            correct_answer: The correct answer to the problem
            target_wrong_answer: The wrong answer we want to achieve
            strategy: Specific attack strategy to use
            **kwargs: Additional attack-specific parameters
        
        Returns:
            AttackResult containing attack outcome and metrics
        """
        pass
    
    @abstractmethod
    def _validate_inputs(
        self,
        problem: str,
        correct_answer: str,
        target_wrong_answer: str
    ) -> bool:
        """
        Validate attack inputs before execution.
        
        Args:
            problem: Problem statement
            correct_answer: Correct answer
            target_wrong_answer: Target wrong answer
        
        Returns:
            True if inputs are valid, raises exception otherwise
        """
        pass
    
    def attack_batch(
        self,
        problems: List[Dict[str, str]],
        strategy: Optional[str] = None,
        **kwargs
    ) -> List[AttackResult]:
        """
        Execute attacks on a batch of problems.
        
        Args:
            problems: List of dicts with 'problem', 'correct_answer', 'target_wrong_answer'
            strategy: Attack strategy to use
            **kwargs: Additional parameters
        
        Returns:
            List of AttackResults
        """
        results = []
        
        for i, prob in enumerate(problems):
            if self.verbose:
                logger.info(f"Attacking problem {i+1}/{len(problems)}")
            
            result = self.attack(
                problem=prob["problem"],
                correct_answer=prob["correct_answer"],
                target_wrong_answer=prob["target_wrong_answer"],
                strategy=strategy,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def attack_with_defense(
        self,
        problem: str,
        correct_answer: str,
        target_wrong_answer: str,
        defense: Any,
        strategy: Optional[str] = None,
        **kwargs
    ) -> Dict[str, AttackResult]:
        """
        Execute attack with and without defense for comparison.
        
        Args:
            problem: Problem statement
            correct_answer: Correct answer
            target_wrong_answer: Target wrong answer
            defense: Defense instance to test against
            strategy: Attack strategy
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with 'baseline' and 'defended' results
        """
        # Attack without defense
        baseline_result = self.attack(
            problem=problem,
            correct_answer=correct_answer,
            target_wrong_answer=target_wrong_answer,
            strategy=strategy,
            **kwargs
        )
        
        # Attack with defense
        defended_result = self.attack(
            problem=problem,
            correct_answer=correct_answer,
            target_wrong_answer=target_wrong_answer,
            strategy=strategy,
            defense=defense,
            **kwargs
        )
        
        return {
            "baseline": baseline_result,
            "defended": defended_result
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get attack statistics.
        
        Returns:
            Dictionary with attack statistics
        """
        success_rate = (
            self._success_count / self._attack_count 
            if self._attack_count > 0 
            else 0.0
        )
        
        return {
            "name": self.name,
            "total_attacks": self._attack_count,
            "successful_attacks": self._success_count,
            "success_rate": success_rate,
            "device": str(self.device)
        }
    
    def reset_statistics(self) -> None:
        """Reset attack statistics."""
        self._attack_count = 0
        self._success_count = 0
    
    def _log_result(self, result: AttackResult) -> None:
        """Log attack result and update statistics."""
        self._attack_count += 1
        if result.success:
            self._success_count += 1
        
        if self.verbose:
            status = "SUCCESS" if result.success else "FAILED"
            logger.info(
                f"[{self.name}] Attack {status} | "
                f"Time: {result.attack_time:.2f}s | "
                f"Iterations: {result.iterations}"
            )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', device='{self.device}')"
