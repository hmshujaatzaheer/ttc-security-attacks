"""
CRA Defense Implementation

Causal Reward Adjustment (CRA)
Reference: Song et al., arXiv:2508.04216, 2025

CRA mitigates reward hacking by estimating the true reward using
causal inference - training sparse autoencoders on PRM activations
and correcting confounding via backdoor adjustment.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CRAConfig:
    """Configuration for CRA defense.
    
    Attributes:
        prm_model: PRM model to defend
        sae_hidden_dim: Hidden dimension for sparse autoencoder
        sparsity_coefficient: Sparsity regularization
        backdoor_adjustment: Enable backdoor adjustment
        confound_threshold: Threshold for confound detection
    """
    prm_model: str = "math-shepherd-mistral-7b-prm"
    sae_hidden_dim: int = 512
    sparsity_coefficient: float = 0.1
    backdoor_adjustment: bool = True
    confound_threshold: float = 0.3
    device: str = "auto"


class SparseAutoencoder:
    """Sparse autoencoder for interpretable feature recovery."""
    
    def __init__(self, input_dim: int, hidden_dim: int, sparsity: float):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity = sparsity
        self._encoder = None
        self._decoder = None
    
    def encode(self, activations: np.ndarray) -> np.ndarray:
        """Encode activations to sparse features."""
        # Simplified - in production use actual SAE
        features = activations @ np.random.randn(activations.shape[-1], self.hidden_dim)
        # Apply sparsity (top-k)
        k = int(self.hidden_dim * (1 - self.sparsity))
        mask = np.argsort(np.abs(features), axis=-1)[:, -k:]
        sparse_features = np.zeros_like(features)
        for i, m in enumerate(mask):
            sparse_features[i, m] = features[i, m]
        return sparse_features
    
    def decode(self, features: np.ndarray) -> np.ndarray:
        """Decode sparse features back to activations."""
        return features @ np.random.randn(self.hidden_dim, self.input_dim)


class CRADefense:
    """Causal Reward Adjustment defense.
    
    CRA addresses reward hacking from a causal inference perspective:
    1. Train SAE on PRM activations to recover interpretable features
    2. Identify confounding semantic features
    3. Apply backdoor adjustment to estimate true reward
    
    Advantages:
    - Does not require modifying the policy model
    - Does not require retraining PRM
    - Principled causal framework
    
    Example:
        >>> defense = CRADefense(CRAConfig())
        >>> defense.setup()
        >>> true_reward = defense.adjust_reward(trace, raw_prm_score)
    """
    
    def __init__(self, config: CRAConfig):
        self.config = config
        self._initialized = False
        self._sae = None
        self._confound_features: List[int] = []
        
    def setup(self) -> None:
        """Initialize CRA components."""
        logger.info("Setting up CRA defense...")
        
        # Initialize sparse autoencoder
        self._sae = SparseAutoencoder(
            input_dim=768,  # Typical transformer hidden dim
            hidden_dim=self.config.sae_hidden_dim,
            sparsity=self.config.sparsity_coefficient
        )
        
        # Identify confounding features (would be learned in practice)
        self._confound_features = [0, 5, 10, 15]  # Mock confounders
        
        self._initialized = True
        logger.info("CRA defense ready")
    
    def extract_activations(self, trace: List[str]) -> np.ndarray:
        """Extract PRM activations for trace.
        
        In production, run trace through PRM and extract
        intermediate activations.
        """
        # Mock activation extraction
        activations = []
        for step in trace:
            # Generate mock activation based on step
            np.random.seed(hash(step) % 2**32)
            act = np.random.randn(768)
            activations.append(act)
        return np.array(activations)
    
    def identify_confounders(
        self,
        features: np.ndarray,
        rewards: np.ndarray
    ) -> List[int]:
        """Identify confounding features.
        
        Features that correlate with reward but not with
        actual correctness are confounders.
        """
        confounders = []
        for i in range(features.shape[1]):
            # Check correlation with rewards
            if features[:, i].std() > 0:
                corr = np.corrcoef(features[:, i], rewards)[0, 1]
                if abs(corr) > self.config.confound_threshold:
                    confounders.append(i)
        return confounders
    
    def backdoor_adjust(
        self,
        raw_reward: float,
        features: np.ndarray,
        confound_indices: List[int]
    ) -> float:
        """Apply backdoor adjustment to get true reward.
        
        P(R | do(X)) = Σ_c P(R | X, C=c) P(C=c)
        
        We marginalize over confounding features to get
        the causal effect of the reasoning on reward.
        """
        if not self.config.backdoor_adjustment:
            return raw_reward
        
        # Extract confound values
        confound_values = features[confound_indices] if len(confound_indices) > 0 else []
        
        # Estimate adjustment factor
        # (simplified - in practice use proper causal estimation)
        if len(confound_values) > 0:
            confound_effect = np.mean(np.abs(confound_values)) * 0.1
            adjusted = raw_reward - confound_effect
        else:
            adjusted = raw_reward
        
        return float(np.clip(adjusted, 0, 1))
    
    def adjust_reward(
        self,
        trace: List[str],
        raw_prm_score: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Adjust PRM reward using causal analysis.
        
        Args:
            trace: Reasoning trace
            raw_prm_score: Raw PRM score to adjust
            
        Returns:
            Tuple of (adjusted_score, diagnostics)
        """
        if not self._initialized:
            self.setup()
        
        # Extract activations
        activations = self.extract_activations(trace)
        
        # Encode to sparse features
        features = self._sae.encode(activations)
        avg_features = features.mean(axis=0)
        
        # Apply backdoor adjustment
        adjusted = self.backdoor_adjust(
            raw_prm_score,
            avg_features,
            self._confound_features
        )
        
        diagnostics = {
            "raw_score": raw_prm_score,
            "adjusted_score": adjusted,
            "adjustment": raw_prm_score - adjusted,
            "num_confounders": len(self._confound_features),
            "feature_sparsity": (features == 0).mean(),
        }
        
        return adjusted, diagnostics
    
    def detect_reward_hacking(
        self,
        trace: List[str],
        raw_prm_score: float
    ) -> Tuple[bool, float]:
        """Detect reward hacking via large adjustment.
        
        If CRA adjustment is large, the raw reward was likely
        influenced by confounding (potential hacking).
        """
        adjusted, diag = self.adjust_reward(trace, raw_prm_score)
        
        adjustment_magnitude = abs(diag["adjustment"])
        is_hacking = adjustment_magnitude > 0.2
        
        return is_hacking, adjustment_magnitude
    
    def cleanup(self) -> None:
        """Release resources."""
        self._sae = None
        self._initialized = False
