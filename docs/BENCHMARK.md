# TTC-Sec Benchmark Documentation

## Overview

TTC-Sec is the first comprehensive benchmark for test-time compute security,
following AgentDojo methodology (Debenedetti et al., NeurIPS 2024).

## Components

### PRM-Adv
Evaluates attacks on Process Reward Models:
- Targets: Math-Shepherd, Skywork-PRM
- Metrics: Attack Success Rate, Score Inflation

### SC-Adv
Evaluates attacks on Self-Consistency voting:
- Targets: GPT-4, Claude, LLaMA
- Metrics: Vote Flip Rate, Diversity Reduction

### MCTS-Adv
Evaluates attacks on Tree Search:
- Targets: MCTSr, LATS
- Metrics: Path Deviation, Value Corruption

## Running the Benchmark

```python
from src.evaluation import TTCSecBenchmark, BenchmarkConfig

config = BenchmarkConfig(
    dataset_size=100,
    defense_evaluation=True
)
benchmark = TTCSecBenchmark(config)
results = benchmark.run_full_benchmark()
benchmark.print_summary()
benchmark.save_results()
```

## Command Line

```bash
python scripts/run_benchmark.py --components all --size 100
```

## Metrics

| Metric | Description |
|--------|-------------|
| ASR | Attack Success Rate |
| Score Inflation | Average PRM score increase |
| Vote Flip Rate | Fraction of changed votes |
| Path Deviation | Divergence from original path |

## Results Format

Results are saved as JSON:
```json
{
  "component": "prm_adv",
  "attack_type": "nlba",
  "strategy": "nl_blindness",
  "metrics": {
    "attack_success_rate": 0.65,
    "avg_score_inflation": 0.25
  }
}
```
