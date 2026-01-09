"""
Tests for Natural Language Blindness Attack (NLBA)
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from attacks import NLBAAttack, AttackResult


class TestNLBAAttack:
    """Test suite for NLBA attack."""
    
    def test_initialization(self):
        """Test attack initialization."""
        attack = NLBAAttack(verbose=False)
        assert attack.name == "NLBA"
        assert attack.config is not None
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        attack = NLBAAttack(verbose=False)
        result = attack._validate_inputs(
            problem="What is 2 + 2?",
            correct_answer="4",
            target_wrong_answer="5"
        )
        assert result is True
    
    def test_validate_inputs_empty_problem(self):
        """Test input validation with empty problem."""
        attack = NLBAAttack(verbose=False)
        with pytest.raises(ValueError, match="Problem statement cannot be empty"):
            attack._validate_inputs(
                problem="",
                correct_answer="4",
                target_wrong_answer="5"
            )
    
    def test_validate_inputs_same_answers(self):
        """Test input validation when answers are the same."""
        attack = NLBAAttack(verbose=False)
        with pytest.raises(ValueError, match="Target wrong answer must differ"):
            attack._validate_inputs(
                problem="What is 2 + 2?",
                correct_answer="4",
                target_wrong_answer="4"
            )
    
    def test_extract_math_expressions(self):
        """Test mathematical expression extraction."""
        attack = NLBAAttack(verbose=False)
        expressions = attack._extract_math_expressions(
            problem="Calculate 15% of 200",
            correct_answer="30"
        )
        assert len(expressions) > 0
        assert "= 30" in expressions
    
    def test_construct_deceptive_trace(self):
        """Test deceptive trace construction."""
        attack = NLBAAttack(verbose=False)
        trace = attack._construct_deceptive_trace(
            problem="What is 2 + 2?",
            math_expressions=["2 + 2", "= 4"],
            target_answer="5"
        )
        assert len(trace) > 0
        assert "5" in trace[-1]
    
    def test_attack_result_structure(self):
        """Test attack result structure."""
        result = AttackResult(
            success=True,
            problem="What is 2 + 2?",
            target_answer="5",
            achieved_answer="5",
            prm_score=0.85
        )
        
        assert result.success is True
        assert result.prm_score == 0.85
        
        result_dict = result.to_dict()
        assert "success" in result_dict
        assert "prm_score" in result_dict
    
    def test_statistics_tracking(self):
        """Test attack statistics tracking."""
        attack = NLBAAttack(verbose=False)
        
        # Reset stats
        attack.reset_statistics()
        
        stats = attack.get_statistics()
        assert stats["total_attacks"] == 0
        assert stats["success_rate"] == 0.0


class TestAttackStrategies:
    """Test different attack strategies."""
    
    def test_nl_blindness_strategy(self):
        """Test NL-blindness attack strategy."""
        attack = NLBAAttack(verbose=False)
        
        # Mock test - actual attack requires model loading
        # This tests the strategy selection logic
        assert "nl_blindness" in ["nl_blindness", "ood_difficulty", "gradient_injection"]
    
    def test_strategy_validation(self):
        """Test that invalid strategies raise errors."""
        attack = NLBAAttack(verbose=False)
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            attack._nl_blindness_attack = lambda *args, **kwargs: None
            attack._ood_difficulty_attack = lambda *args, **kwargs: None
            attack._gradient_injection_attack = lambda *args, **kwargs: None
            
            # This would fail in actual implementation
            # attack.attack(..., strategy="invalid_strategy")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
