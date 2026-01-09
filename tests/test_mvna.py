"""Unit tests for MVNA attack."""
import unittest
import sys
sys.path.insert(0, '.')

from src.attacks import MVNAAttack, MVNAConfig
from src.attacks.base_attack import AttackStatus

class TestMVNAAttack(unittest.TestCase):
    """Test cases for MCTS Value Network Attack."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MVNAConfig(
            mcts_system="mock-mcts",
            num_simulations=20,
            seed=42
        )
        self.attack = MVNAAttack(self.config)
    
    def test_attack_initialization(self):
        """Test attack initializes correctly."""
        self.assertEqual(self.attack.config.name, "mvna_attack")
    
    def test_attack_execution(self):
        """Test basic attack execution."""
        result = self.attack.attack("Solve: x^2 - 5x + 6 = 0")
        self.assertIsNotNone(result)
        self.assertIn("strategy", result.metrics)
    
    def test_value_fooling_strategy(self):
        """Test value fooling strategy."""
        self.config.strategy = "value_fooling"
        result = self.attack.attack("Test problem")
        self.assertEqual(result.metrics["strategy"], "value_fooling")
    
    def tearDown(self):
        """Clean up after tests."""
        self.attack.cleanup()

if __name__ == '__main__':
    unittest.main()
