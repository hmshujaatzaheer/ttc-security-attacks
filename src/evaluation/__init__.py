"""
TTC Security Attacks - Evaluation Module

This module provides the TTC-Sec benchmark and evaluation metrics.
"""

from .ttc_sec_benchmark import TTCSecBenchmark, BenchmarkResult, BenchmarkConfig, compute_metrics

__all__ = [
    "TTCSecBenchmark",
    "BenchmarkResult",
    "BenchmarkConfig",
    "compute_metrics",
]
