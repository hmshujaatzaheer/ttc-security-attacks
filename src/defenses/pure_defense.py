"""
PURE Defense Implementation

Process sUpervised Reinforcement lEarning (PURE)
Reference: Cheng et al., NeurIPS 2025, arXiv:2504.15275

PURE uses min-form credit assignment instead of summation,
which significantly alleviates reward hacking.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass  
class PUREConfig:
    """Configuration for PURE defense.
    
    Attributes:
        model_name: Base model for PURE
        min_form_enabled: Use min-form credit assignment
        value_range_limit: Limit value function range
        advantage_distribution: How to distribute advantages
    """
    model_name: str = "Qwen2.5-Math-7B"
    min_form_enabled: bool = True
    value_range_limit: float = 1.0
    advantage_distribution: str = "balanced"
    device: str = "auto"


class PUREDefense:
    """PURE defense against reward hacking.
    
    Key insight: Canonical summation-form credit assignment
    V(s) = Σ γ^t r_t easily induces LLMs to hack high-reward steps.
    
    PURE uses min-form: V(s) = min(r_1, r_2, ..., r_n)
    This limits value range and distributes advantages more reasonably.
    
    Example:
        >>> defense = PUREDefense(PUREConfig())
        >>> defense.setup()
        >>> robust_score = defense.score_trace(trace)
    """
    
    def __init__(self, config: PUREConfig):
        self.config = config
        self._initialized = False
        
    def setup(self) -> None:
        """Initialize PURE components."""
        logger.info("Setting up PURE defense...")
        self._initialized = True
        logger.info("PURE defense ready")
    
    def compute_value_minform(self, rewards: List[float]) -> float:
        """Compute value using min-form credit assignment.
        
        V(s) = min(r_1, r_2, ..., r_n)
        
        This prevents reward hacking by ensuring the value
        is limited by the weakest step.
        
        Args:
            rewards: List of step rewards
            
        Returns:
            Min-form value
        """
        if not rewards:
            return 0.0
        
        if self.config.min_form_enabled:
            return min(rewards)
        else:
            # Fallback to sum-form (not recommended)
            return sum(rewards)
    
    def compute_advantage(
        self,
        step_idx: int,
        rewards: List[float]
    ) -> float:
        """Compute advantage for a step using PURE's method.
        
        PURE distributes advantages more reasonably by considering
        the impact of each step on the min value.
        
        Args:
            step_idx: Index of the step
            rewards: All step rewards
            
        Returns:
            Advantage for the step
        """
        if not rewards or step_idx >= len(rewards):
            return 0.0
        
        current_min = min(rewards)
        
        # Compute counterfactual: what if this step was removed?
        rewards_without = rewards[:step_idx] + rewards[step_idx + 1:]
        counterfactual_min = min(rewards_without) if rewards_without else 0.0
        
        # Advantage is the difference
        advantage = current_min - counterfactual_min
        
        return advantage
    
    def score_trace(
        self,
        trace: List[str],
        step_rewards: Optional[List[float]] = None
    ) -> Tuple[float, List[float], List[float]]:
        """Score trace using PURE's min-form value.
        
        Args:
            trace: List of reasoning steps
            step_rewards: Optional pre-computed step rewards
            
        Returns:
            Tuple of (value, step_rewards, advantages)
        """
        if not self._initialized:
            self.setup()
        
        # Compute step rewards if not provided
        if step_rewards is None:
            step_rewards = [self._compute_step_reward(s) for s in trace]
        
        # Compute min-form value
        value = self.compute_value_minform(step_rewards)
        
        # Compute advantages
        advantages = [
            self.compute_advantage(i, step_rewards)
            for i in range(len(step_rewards))
        ]
        
        return value, step_rewards, advantages
    
    def detect_reward_hacking(
        self,
        trace: List[str],
        step_rewards: Optional[List[float]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Detect potential reward hacking attempts.
        
        PURE's min-form makes it harder to hack because:
        1. Value is bounded by worst step
        2. Advantages are distributed fairly
        
        Args:
            trace: Trace to analyze
            step_rewards: Optional step rewards
            
        Returns:
            Tuple of (is_hacking, details)
        """
        value, rewards, advantages = self.score_trace(trace, step_rewards)
        
        # Check for hacking indicators
        reward_variance = np.var(rewards) if rewards else 0
        max_advantage = max(advantages) if advantages else 0
        min_reward = min(rewards) if rewards else 0
        
        # High variance + high max advantage suggests manipulation
        is_hacking = (reward_variance > 0.3 and max_advantage > 0.5)
        
        # Also check for "all high except one" pattern
        high_count = sum(1 for r in rewards if r > 0.8)
        if high_count >= len(rewards) - 1 and min_reward < 0.3:
            is_hacking = True
        
        return is_hacking, {
            "value": value,
            "reward_variance": reward_variance,
            "max_advantage": max_advantage,
            "min_reward": min_reward,
            "high_reward_count": high_count,
        }
    
    def _compute_step_reward(self, step: str) -> float:
        """Compute reward for a single step."""
        # Simplified reward computation
        has_math = any(c in step for c in "+-*/=")
        has_reasoning = any(w in step.lower() for w in ["because", "therefore"])
        length_penalty = min(len(step) / 200, 1.0)
        
        reward = 0.4 + 0.2 * has_math + 0.2 * has_reasoning + 0.2 * length_penalty
        return float(np.clip(reward, 0, self.config.value_range_limit))
    
    def cleanup(self) -> None:
        """Release resources."""
        self._initialized = False
