"""LLM sampling utilities."""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class SamplingConfig:
    """Configuration for LLM sampling."""
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024
    n: int = 1

def sample_paths(
    model: Any,
    prompt: str,
    config: SamplingConfig
) -> List[str]:
    """Sample multiple reasoning paths from model."""
    # Mock sampling - in production use actual API
    paths = []
    for i in range(config.n):
        np.random.seed(i)
        answer = np.random.randint(1, 100)
        path = f"Step 1: Analyze problem\nStep 2: Apply method\nStep 3: Calculate\nAnswer: {answer}"
        paths.append(path)
    return paths

def extract_answer(response: str) -> Optional[str]:
    """Extract answer from model response."""
    import re
    match = re.search(r'[Aa]nswer[:\s]+(\d+)', response)
    return match.group(1) if match else None
