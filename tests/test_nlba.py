"""Unit tests for NLBA attack."""
import unittest
import sys
sys.path.insert(0, '.')

from src.attacks import NLBAAttack, NLBAConfig
from src.attacks.base_attack import AttackStatus

class TestNLBAAttack(unittest.TestCase):
    """Test cases for Natural Language Blindness Attack."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NLBAConfig(
            prm_model_name="mock-prm",
            max_iterations=10,
            num_candidates=3,
            seed=42
        )
        self.attack = NLBAAttack(self.config)
    
    def test_attack_initialization(self):
        """Test attack initializes correctly."""
        self.assertEqual(self.attack.config.name, "nlba_attack")
        self.assertFalse(self.attack._is_setup)
    
    def test_attack_setup(self):
        """Test attack setup works."""
        self.attack.setup()
        self.assertTrue(self.attack._is_setup)
        self.assertIsNotNone(self.attack.prm_scorer)
    
    def test_attack_execution(self):
        """Test basic attack execution."""
        result = self.attack.attack(
            "Solve: 2x + 5 = 15",
            target="x = 10"
        )
        self.assertIsNotNone(result)
        self.assertIn(result.status, [AttackStatus.SUCCESS, AttackStatus.FAILURE])
        self.assertIn("prm_score", result.metrics)
    
    def test_batch_attack(self):
        """Test batch attack functionality."""
        inputs = [
            "Solve: x + 5 = 10",
            "What is 10% of 50?",
        ]
        results = self.attack.batch_attack(inputs)
        self.assertEqual(len(results), 2)
    
    def test_attack_statistics(self):
        """Test statistics tracking."""
        self.attack.attack("Test problem")
        stats = self.attack.get_statistics()
        self.assertEqual(stats["total_attempts"], 1)
    
    def tearDown(self):
        """Clean up after tests."""
        self.attack.cleanup()

if __name__ == '__main__':
    unittest.main()
