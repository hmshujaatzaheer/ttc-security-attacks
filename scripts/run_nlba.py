#!/usr/bin/env python3
"""Run NLBA attack from command line."""
import argparse
import sys
sys.path.insert(0, '.')

from src.attacks import NLBAAttack, NLBAConfig

def main():
    parser = argparse.ArgumentParser(description='Run NLBA Attack')
    parser.add_argument('--problem', type=str, required=True, help='Math problem to attack')
    parser.add_argument('--target', type=str, help='Target wrong answer')
    parser.add_argument('--strategy', type=str, default='nl_blindness',
                       choices=['nl_blindness', 'ood_difficulty', 'gradient_injection'])
    parser.add_argument('--prm', type=str, default='math-shepherd-mistral-7b-prm')
    parser.add_argument('--threshold', type=float, default=0.8)
    args = parser.parse_args()
    
    config = NLBAConfig(
        prm_model_name=args.prm,
        strategy=args.strategy,
        prm_threshold=args.threshold
    )
    
    attack = NLBAAttack(config)
    result = attack.attack(args.problem, args.target)
    
    print(f"\n{'='*50}")
    print(f"NLBA Attack Result")
    print(f"{'='*50}")
    print(f"Success: {result.success}")
    print(f"PRM Score: {result.confidence:.3f}")
    print(f"Iterations: {result.iterations_used}")
    if result.adversarial_output:
        print(f"\nAdversarial Trace:\n{result.adversarial_output[:500]}...")

if __name__ == '__main__':
    main()
