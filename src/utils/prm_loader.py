"""PRM model loading utilities."""
from typing import Optional, Dict, Any
import logging
logger = logging.getLogger(__name__)

class PRMWrapper:
    """Wrapper for Process Reward Models."""
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
    
    def load(self):
        """Load the PRM model."""
        logger.info(f"Loading PRM: {self.model_name}")
        # In production: load from HuggingFace
        self._model = "mock_model"
        logger.info("PRM loaded successfully")
    
    def score(self, trace: str) -> float:
        """Score a reasoning trace."""
        if self._model is None:
            self.load()
        # Mock scoring
        import numpy as np
        return float(np.random.uniform(0.5, 0.9))

def load_prm(model_name: str, device: str = "auto") -> PRMWrapper:
    """Load a PRM model by name."""
    wrapper = PRMWrapper(model_name, device)
    wrapper.load()
    return wrapper

SUPPORTED_PRMS = {
    "math-shepherd": "math-shepherd-mistral-7b-prm",
    "skywork": "Skywork-o1-Open-PRM-Qwen-2.5-7B",
}
