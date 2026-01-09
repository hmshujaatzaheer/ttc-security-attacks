"""Visualization utilities for TTC-Sec benchmark results."""
from typing import Dict, List, Any, Optional
import json

def plot_results(results: List[Dict], output_path: str = "results.png") -> str:
    """Plot benchmark results (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        
        components = []
        asrs = []
        for r in results:
            components.append(f"{r['attack_type']}_{r['strategy']}")
            asrs.append(r['metrics'].get('attack_success_rate', 0))
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(components)), asrs)
        plt.xticks(range(len(components)), components, rotation=45, ha='right')
        plt.ylabel('Attack Success Rate')
        plt.title('TTC-Sec Benchmark Results')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    except ImportError:
        print("matplotlib not available for plotting")
        return ""

def create_report(results: List[Dict], output_path: str = "report.md") -> str:
    """Create markdown report from results."""
    lines = ["# TTC-Sec Benchmark Report\n"]
    lines.append("## Summary\n")
    
    for r in results:
        lines.append(f"### {r['attack_type'].upper()} - {r['strategy']}")
        lines.append(f"- Component: {r['component']}")
        lines.append(f"- ASR: {r['metrics'].get('attack_success_rate', 0):.2%}")
        lines.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    return output_path
