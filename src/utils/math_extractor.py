"""Mathematical expression extraction utilities."""
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MathPattern:
    """Represents an extracted mathematical pattern."""
    expression: str
    pattern_type: str
    position: Tuple[int, int]

def extract_math(text: str) -> List[MathPattern]:
    """Extract mathematical expressions from text."""
    patterns = []
    
    # Equations
    for match in re.finditer(r'(\d+\s*[+\-*/=]\s*\d+)', text):
        patterns.append(MathPattern(
            expression=match.group(1),
            pattern_type="equation",
            position=match.span()
        ))
    
    # Variables
    for match in re.finditer(r'([a-zA-Z])\s*=\s*(\d+)', text):
        patterns.append(MathPattern(
            expression=match.group(0),
            pattern_type="assignment",
            position=match.span()
        ))
    
    return patterns

def validate_math(expression: str) -> bool:
    """Validate a mathematical expression."""
    try:
        # Simple validation
        return bool(re.match(r'^[\d\s+\-*/=().x]+$', expression))
    except:
        return False
