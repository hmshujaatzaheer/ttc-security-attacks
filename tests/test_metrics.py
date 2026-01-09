"""Unit tests for evaluation metrics."""
import unittest
import sys
sys.path.insert(0, '.')

from src.evaluation.metrics import (
    compute_asr,
    compute_score_inflation,
    compute_vote_flip_rate,
    compute_path_deviation
)

class TestMetrics(unittest.TestCase):
    """Test cases for evaluation metrics."""
    
    def test_compute_asr(self):
        """Test attack success rate computation."""
        self.assertEqual(compute_asr(50, 100), 0.5)
        self.assertEqual(compute_asr(0, 100), 0.0)
        self.assertEqual(compute_asr(100, 100), 1.0)
        self.assertEqual(compute_asr(0, 0), 0.0)
    
    def test_compute_score_inflation(self):
        """Test score inflation computation."""
        original = [0.5, 0.6, 0.7]
        attacked = [0.7, 0.8, 0.9]
        inflation = compute_score_inflation(original, attacked)
        self.assertAlmostEqual(inflation, 0.2, places=5)
    
    def test_compute_vote_flip_rate(self):
        """Test vote flip rate computation."""
        original = ['5', '5', '5', '10']
        attacked = ['5', '10', '10', '10']
        flip_rate = compute_vote_flip_rate(original, attacked)
        self.assertAlmostEqual(flip_rate, 0.5, places=5)
    
    def test_compute_path_deviation(self):
        """Test path deviation computation."""
        original = ['step1', 'step2', 'step3']
        attacked = ['step1', 'step2', 'step4']
        deviation = compute_path_deviation(original, attacked)
        self.assertGreater(deviation, 0)
        self.assertLess(deviation, 1)

if __name__ == '__main__':
    unittest.main()
