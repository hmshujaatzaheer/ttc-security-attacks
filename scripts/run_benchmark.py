#!/usr/bin/env python3
"""
TTC Security Attacks - Benchmark Runner

Run the TTC-Sec benchmark to evaluate attacks on test-time compute mechanisms.

Usage:
    python scripts/run_benchmark.py --config configs/benchmark_config.yaml
    python scripts/run_benchmark.py --component prm_adv --model skywork-prm
    python scripts/run_benchmark.py --generate-report --output results/report.pdf
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import TTCSecBenchmark, BenchmarkConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TTC-Sec Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full benchmark
    python run_benchmark.py
    
    # Run specific component
    python run_benchmark.py --component prm_adv
    
    # Run with custom config
    python run_benchmark.py --config my_config.yaml
    
    # Reproduce results with fixed seed
    python run_benchmark.py --reproduce --seed 42
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to benchmark configuration file"
    )
    
    parser.add_argument(
        "--component",
        type=str,
        choices=["prm_adv", "sc_adv", "mcts_adv", "all"],
        default="all",
        help="Benchmark component to run"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to test"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Reproduce results with fixed settings"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate PDF report from results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("TTC-Sec Benchmark")
    logger.info("=" * 60)
    
    # Create config
    config = BenchmarkConfig(
        output_dir=args.output,
        seed=args.seed
    )
    
    # Initialize benchmark
    benchmark = TTCSecBenchmark(
        config=config,
        verbose=args.verbose
    )
    
    # Run appropriate component(s)
    if args.component == "all":
        logger.info("Running full benchmark...")
        results = benchmark.run_all()
    elif args.component == "prm_adv":
        logger.info("Running PRM-Adv benchmark...")
        results = {"prm_adv": benchmark.run_prm_adv()}
    elif args.component == "sc_adv":
        logger.info("Running SC-Adv benchmark...")
        results = {"sc_adv": benchmark.run_sc_adv()}
    elif args.component == "mcts_adv":
        logger.info("Running MCTS-Adv benchmark...")
        results = {"mcts_adv": benchmark.run_mcts_adv()}
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    
    if "overall_asr" in results:
        logger.info(f"Overall Attack Success Rate: {results['overall_asr']:.2%}")
        
        for comp, asr in results.get("component_asrs", {}).items():
            logger.info(f"  {comp}: {asr:.2%}")
    
    logger.info(f"\nResults saved to: {args.output}")
    
    if args.generate_report:
        logger.info("Generating PDF report...")
        # TODO: Implement report generation
        logger.warning("Report generation not yet implemented")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
