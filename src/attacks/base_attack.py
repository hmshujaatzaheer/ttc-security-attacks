"""
Base Attack Framework for TTC Security Attacks

This module provides the abstract base class and common utilities
for all mechanistic attacks on test-time compute mechanisms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import time
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Enumeration of attack types."""
    NLBA = "natural_language_blindness"
    SMVA = "single_model_voting"
    MVNA = "mcts_value_network"


class AttackStatus(Enum):
    """Status of an attack attempt."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class AttackConfig:
    """Base configuration for all attacks.
    
    Attributes:
        name: Human-readable attack name
        attack_type: Type of attack from AttackType enum
        max_iterations: Maximum attack iterations
        timeout_seconds: Maximum time for attack attempt
        seed: Random seed for reproducibility
        device: Computation device ('cuda', 'cpu', 'auto')
        verbose: Enable detailed logging
        save_artifacts: Save intermediate attack artifacts
        output_dir: Directory for saving results
    """
    name: str = "base_attack"
    attack_type: AttackType = AttackType.NLBA
    max_iterations: int = 100
    timeout_seconds: float = 300.0
    seed: int = 42
    device: str = "auto"
    verbose: bool = True
    save_artifacts: bool = True
    output_dir: str = "./results"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "attack_type": self.attack_type.value,
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
            "seed": self.seed,
            "device": self.device,
            "verbose": self.verbose,
            "save_artifacts": self.save_artifacts,
            "output_dir": self.output_dir,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackConfig":
        """Create config from dictionary."""
        if "attack_type" in data and isinstance(data["attack_type"], str):
            data["attack_type"] = AttackType(data["attack_type"])
        return cls(**data)


@dataclass
class AttackResult:
    """Result of an attack attempt.
    
    Attributes:
        success: Whether attack achieved its goal
        status: Detailed status of the attack
        original_input: The original input/problem
        adversarial_output: The crafted adversarial output
        target_answer: The target wrong answer (if applicable)
        achieved_answer: The answer actually achieved
        confidence: Confidence score of the attack success
        iterations_used: Number of iterations consumed
        time_elapsed: Time taken for the attack
        metrics: Additional attack-specific metrics
        artifacts: Intermediate artifacts (traces, scores, etc.)
        error_message: Error message if attack failed
    """
    success: bool
    status: AttackStatus
    original_input: str
    adversarial_output: Optional[str] = None
    target_answer: Optional[str] = None
    achieved_answer: Optional[str] = None
    confidence: float = 0.0
    iterations_used: int = 0
    time_elapsed: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "status": self.status.value,
            "original_input": self.original_input,
            "adversarial_output": self.adversarial_output,
            "target_answer": self.target_answer,
            "achieved_answer": self.achieved_answer,
            "confidence": self.confidence,
            "iterations_used": self.iterations_used,
            "time_elapsed": self.time_elapsed,
            "metrics": self.metrics,
            "error_message": self.error_message,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save result to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.to_json())


class BaseAttack(ABC):
    """Abstract base class for all TTC security attacks.
    
    This class provides the common interface and utilities for implementing
    mechanistic attacks on test-time compute mechanisms.
    
    Subclasses must implement:
        - _setup(): Initialize attack-specific components
        - _execute_attack(): Core attack logic
        - _cleanup(): Clean up resources
    """
    
    def __init__(self, config: AttackConfig):
        """Initialize the attack.
        
        Args:
            config: Attack configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self._is_setup = False
        self._stats = {
            "total_attempts": 0,
            "successful_attacks": 0,
            "failed_attacks": 0,
            "total_time": 0.0,
            "average_iterations": 0.0,
        }
    
    @abstractmethod
    def _setup(self) -> None:
        """Initialize attack-specific components.
        
        Called once before any attacks are executed.
        Subclasses should load models, initialize resources, etc.
        """
        pass
    
    @abstractmethod
    def _execute_attack(
        self,
        input_data: str,
        target: Optional[str] = None,
        **kwargs
    ) -> AttackResult:
        """Execute the core attack logic.
        
        Args:
            input_data: The input to attack (problem, prompt, etc.)
            target: Optional target answer/output
            **kwargs: Additional attack-specific parameters
            
        Returns:
            AttackResult containing the outcome
        """
        pass
    
    @abstractmethod
    def _cleanup(self) -> None:
        """Clean up attack resources.
        
        Called when attack is no longer needed.
        Subclasses should release models, close connections, etc.
        """
        pass
    
    def setup(self) -> None:
        """Public setup method with error handling."""
        if self._is_setup:
            self.logger.warning("Attack already set up, skipping")
            return
        
        self.logger.info(f"Setting up {self.config.name}...")
        try:
            self._setup()
            self._is_setup = True
            self.logger.info("Setup complete")
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            raise
    
    def attack(
        self,
        input_data: str,
        target: Optional[str] = None,
        **kwargs
    ) -> AttackResult:
        """Execute an attack with timing and error handling.
        
        Args:
            input_data: The input to attack
            target: Optional target answer/output
            **kwargs: Additional parameters
            
        Returns:
            AttackResult containing the outcome
        """
        if not self._is_setup:
            self.setup()
        
        start_time = time.time()
        self._stats["total_attempts"] += 1
        
        try:
            # Execute with timeout
            result = self._execute_attack(input_data, target, **kwargs)
            result.time_elapsed = time.time() - start_time
            
            # Update statistics
            if result.success:
                self._stats["successful_attacks"] += 1
            else:
                self._stats["failed_attacks"] += 1
            
            self._stats["total_time"] += result.time_elapsed
            
            # Log result
            if self.config.verbose:
                self.logger.info(
                    f"Attack {'succeeded' if result.success else 'failed'} "
                    f"in {result.time_elapsed:.2f}s "
                    f"({result.iterations_used} iterations)"
                )
            
            return result
            
        except TimeoutError:
            elapsed = time.time() - start_time
            self._stats["failed_attacks"] += 1
            return AttackResult(
                success=False,
                status=AttackStatus.TIMEOUT,
                original_input=input_data,
                time_elapsed=elapsed,
                error_message=f"Attack timed out after {self.config.timeout_seconds}s"
            )
        except Exception as e:
            elapsed = time.time() - start_time
            self._stats["failed_attacks"] += 1
            self.logger.error(f"Attack error: {e}")
            return AttackResult(
                success=False,
                status=AttackStatus.ERROR,
                original_input=input_data,
                time_elapsed=elapsed,
                error_message=str(e)
            )
    
    def batch_attack(
        self,
        inputs: List[str],
        targets: Optional[List[str]] = None,
        **kwargs
    ) -> List[AttackResult]:
        """Execute attacks on multiple inputs.
        
        Args:
            inputs: List of inputs to attack
            targets: Optional list of target outputs
            **kwargs: Additional parameters
            
        Returns:
            List of AttackResults
        """
        if targets is None:
            targets = [None] * len(inputs)
        
        results = []
        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            if self.config.verbose:
                self.logger.info(f"Attacking input {i+1}/{len(inputs)}")
            result = self.attack(inp, tgt, **kwargs)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get attack statistics.
        
        Returns:
            Dictionary of statistics
        """
        total = self._stats["total_attempts"]
        if total > 0:
            self._stats["success_rate"] = self._stats["successful_attacks"] / total
            self._stats["average_time"] = self._stats["total_time"] / total
        else:
            self._stats["success_rate"] = 0.0
            self._stats["average_time"] = 0.0
        
        return self._stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset attack statistics."""
        self._stats = {
            "total_attempts": 0,
            "successful_attacks": 0,
            "failed_attacks": 0,
            "total_time": 0.0,
            "average_iterations": 0.0,
        }
    
    def cleanup(self) -> None:
        """Public cleanup method."""
        if self._is_setup:
            self.logger.info("Cleaning up...")
            self._cleanup()
            self._is_setup = False
    
    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config.name})"
