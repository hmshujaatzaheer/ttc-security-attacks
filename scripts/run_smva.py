#!/usr/bin/env python3
"""Run SMVA attack from command line."""
import argparse
import sys
sys.path.insert(0, '.')

from src.attacks import SMVAAttack, SMVAConfig

def main():
    parser = argparse.ArgumentParser(description='Run SMVA Attack')
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--target', type=str)
    parser.add_argument('--strategy', type=str, default='sampling_bias')
    parser.add_argument('--model', type=str, default='gpt-4')
    parser.add_argument('--samples', type=int, default=40)
    args = parser.parse_args()
    
    config = SMVAConfig(
        model_name=args.model,
        strategy=args.strategy,
        num_samples=args.samples
    )
    
    attack = SMVAAttack(config)
    result = attack.attack(args.problem, args.target)
    
    print(f"\n{'='*50}")
    print(f"SMVA Attack Result")
    print(f"{'='*50}")
    print(f"Vote Flipped: {result.success}")
    print(f"Target Answer: {result.target_answer}")
    print(f"Achieved Answer: {result.achieved_answer}")

if __name__ == '__main__':
    main()
