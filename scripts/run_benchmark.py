#!/usr/bin/env python3
"""Run TTC-Sec benchmark from command line."""
import argparse
import sys
sys.path.insert(0, '.')

from src.evaluation import TTCSecBenchmark, BenchmarkConfig, BenchmarkComponent

def main():
    parser = argparse.ArgumentParser(description='Run TTC-Sec Benchmark')
    parser.add_argument('--components', type=str, default='all',
                       help='Components to run: all, prm_adv, sc_adv, mcts_adv')
    parser.add_argument('--size', type=int, default=100, help='Dataset size')
    parser.add_argument('--defenses', action='store_true', help='Evaluate defenses')
    parser.add_argument('--output', type=str, default='./results/benchmark')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    components = []
    if args.components == 'all':
        components = [BenchmarkComponent.ALL]
    else:
        for c in args.components.split(','):
            components.append(BenchmarkComponent(c.strip()))
    
    config = BenchmarkConfig(
        components=components,
        dataset_size=args.size,
        defense_evaluation=args.defenses,
        output_dir=args.output,
        seed=args.seed
    )
    
    benchmark = TTCSecBenchmark(config)
    results = benchmark.run_full_benchmark()
    benchmark.print_summary()
    
    output_path = benchmark.save_results()
    print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    main()
