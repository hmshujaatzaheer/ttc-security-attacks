"""Unit tests for SMVA attack."""
import unittest
import sys
sys.path.insert(0, '.')

from src.attacks import SMVAAttack, SMVAConfig
from src.attacks.base_attack import AttackStatus

class TestSMVAAttack(unittest.TestCase):
    """Test cases for Single-Model Voting Attack."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = SMVAConfig(
            model_name="mock-model",
            num_samples=10,
            seed=42
        )
        self.attack = SMVAAttack(self.config)
    
    def test_attack_initialization(self):
        """Test attack initializes correctly."""
        self.assertEqual(self.attack.config.name, "smva_attack")
    
    def test_attack_execution(self):
        """Test basic attack execution."""
        result = self.attack.attack(
            "What is 15% of 80?",
            target="15"
        )
        self.assertIsNotNone(result)
        self.assertIn("strategy", result.metrics)
    
    def test_sampling_bias_strategy(self):
        """Test sampling bias strategy."""
        self.config.strategy = "sampling_bias"
        result = self.attack.attack("Test problem", target="42")
        self.assertEqual(result.metrics["strategy"], "sampling_bias")
    
    def tearDown(self):
        """Clean up after tests."""
        self.attack.cleanup()

if __name__ == '__main__':
    unittest.main()
