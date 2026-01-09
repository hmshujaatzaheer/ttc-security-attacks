#!/usr/bin/env python3
"""Run MVNA attack from command line."""
import argparse
import sys
sys.path.insert(0, '.')

from src.attacks import MVNAAttack, MVNAConfig

def main():
    parser = argparse.ArgumentParser(description='Run MVNA Attack')
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--target', type=str)
    parser.add_argument('--strategy', type=str, default='value_fooling')
    parser.add_argument('--system', type=str, default='mctsr')
    parser.add_argument('--simulations', type=int, default=100)
    args = parser.parse_args()
    
    config = MVNAConfig(
        mcts_system=args.system,
        strategy=args.strategy,
        num_simulations=args.simulations
    )
    
    attack = MVNAAttack(config)
    result = attack.attack(args.problem, args.target)
    
    print(f"\n{'='*50}")
    print(f"MVNA Attack Result")
    print(f"{'='*50}")
    print(f"Search Manipulated: {result.success}")
    print(f"Q-Value: {result.confidence:.3f}")

if __name__ == '__main__':
    main()
