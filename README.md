# TTC Security Attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Mechanistic Attacks on Test-Time Compute in Reasoning LLMs**

This repository provides the first comprehensive attack framework targeting test-time compute (TTC) mechanisms in reasoning LLMs. While Zaremba et al. (2025) established that increased inference compute improves adversarial robustness using black-box analysis, this work develops white-box mechanistic attacks targeting specific inference components.

## 🎯 Overview

Modern reasoning LLMs (o1, DeepSeek-R1, Gemini 2.0 Flash Thinking) rely on three key TTC mechanisms:

| Mechanism | Purpose | Our Attack |
|-----------|---------|------------|
| **Process Reward Models (PRMs)** | Score reasoning step quality | **NLBA** - Natural Language Blindness Attack |
| **Self-Consistency Voting** | Aggregate multiple reasoning paths | **SMVA** - Single-Model Voting Attack |
| **MCTS/Tree Search** | Explore reasoning space | **MVNA** - MCTS Value Network Attack |

## 🔬 Attack Implementations

### NLBA: Natural Language Blindness Attack

Exploits the finding by Ma et al. (AAAI 2025) that PRMs largely ignore natural language explanations:

```
R_θ(τ) = ∏ᵢ P_θ(correct | s₁,...,sᵢ)
```

**Attack Strategy**: Construct traces with valid mathematical expressions paired with misleading NL explanations to achieve high PRM scores for incorrect answers.

```python
from src.attacks import NLBAAttack, NLBAConfig

config = NLBAConfig(
    prm_model_name="math-shepherd-mistral-7b-prm",
    strategy="nl_blindness",
    prm_threshold=0.8
)

attack = NLBAAttack(config)
result = attack.attack(
    "Solve: 2x + 5 = 15",
    target="x = 10"  # Wrong answer (correct is x = 5)
)
print(f"Attack Success: {result.success}")
print(f"PRM Score: {result.metrics['prm_score']:.3f}")
```

### SMVA: Single-Model Voting Attack

The first attack on single-model self-consistency, distinct from multi-agent debate attacks (Amayuelas et al., EMNLP 2024):

```
a* = argmax_a Σᵢ 𝟙[aᵢ = a]
```

**Attack Strategy**: Inject sampling bias to reduce path diversity and shift majority vote toward target wrong answers.

```python
from src.attacks import SMVAAttack, SMVAConfig

config = SMVAConfig(
    model_name="gpt-4",
    strategy="sampling_bias",
    num_samples=40
)

attack = SMVAAttack(config)
result = attack.attack(
    "What is 15% of 80?",
    target="15"  # Wrong answer (correct is 12)
)
```

### MVNA: MCTS Value Network Attack

Extends adversarial game-playing research (Lan et al., NeurIPS 2022) from board games to LLM reasoning:

**Attack Strategy**: Identify blind spots in value networks and manipulate UCT exploration to force search toward adversarial subtrees.

```python
from src.attacks import MVNAAttack, MVNAConfig

config = MVNAConfig(
    mcts_system="mctsr",
    strategy="value_fooling",
    num_simulations=100
)

attack = MVNAAttack(config)
result = attack.attack("Solve: x² - 5x + 6 = 0")
```

## 📊 TTC-Sec Benchmark

The first comprehensive test-time compute security benchmark, following AgentDojo methodology (Debenedetti et al., NeurIPS 2024).

### Components

| Component | Target | Metrics |
|-----------|--------|---------|
| **PRM-Adv** | Process Reward Models | Attack Success Rate, Score Inflation |
| **SC-Adv** | Self-Consistency | Vote Flip Rate, Diversity Reduction |
| **MCTS-Adv** | Tree Search | Path Deviation, Value Corruption |

### Running the Benchmark

```python
from src.evaluation import TTCSecBenchmark, BenchmarkConfig

config = BenchmarkConfig(
    dataset_size=100,
    defense_evaluation=True,
    defenses=["prime", "pure", "cra"]
)

benchmark = TTCSecBenchmark(config)
results = benchmark.run_full_benchmark()
benchmark.print_summary()
benchmark.save_results()
```

Or via command line:

```bash
python scripts/run_benchmark.py --components all --size 100 --defenses
```

## 🛡️ Defense Implementations

We implement verified defenses for evaluation:

| Defense | Reference | Mechanism |
|---------|-----------|-----------|
| **PRIME** | Cui et al. (2025) | Implicit process rewards from outcome supervision |
| **PURE** | Cheng et al. (2025) | Min-form credit assignment |
| **CRA** | Song et al. (2025) | Causal reward adjustment via SAE |

```python
from src.defenses import PRIMEDefense, PUREDefense, CRADefense

# Detect potential attacks
defense = PRIMEDefense(config)
is_adversarial, confidence = defense.detect_attack(trace)
```

## 📁 Repository Structure

```
ttc-security-attacks/
├── src/
│   ├── attacks/           # Attack implementations
│   │   ├── nlba.py        # Natural Language Blindness Attack
│   │   ├── smva.py        # Single-Model Voting Attack
│   │   ├── mvna.py        # MCTS Value Network Attack
│   │   └── base_attack.py # Abstract base class
│   ├── defenses/          # Defense implementations
│   │   ├── prime_defense.py
│   │   ├── pure_defense.py
│   │   └── cra_defense.py
│   ├── evaluation/        # Benchmark suite
│   │   ├── ttc_sec_benchmark.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── utils/             # Utilities
├── configs/               # Configuration files
├── data/                  # Datasets and results
├── docs/                  # Documentation
├── notebooks/             # Demo notebooks
├── scripts/               # CLI scripts
└── tests/                 # Unit tests
```

## 🚀 Installation

### Quick Install

```bash
git clone https://github.com/hmshujaatzaheer/ttc-security-attacks.git
cd ttc-security-attacks
pip install -e .
```

### Development Install

```bash
pip install -e ".[dev,viz,notebooks]"
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA 11.8+ (for GPU support)

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for detailed instructions.

## 📖 Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Attack Methodology](docs/ATTACKS.md)
- [Benchmark Documentation](docs/BENCHMARK.md)

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📚 References

### Core Papers

1. **Test-Time Compute**: Zaremba et al. (2025). "Trading inference-time compute for adversarial robustness." arXiv:2501.18841.

2. **PRM Vulnerability**: Ma et al. (2025). "What are step-level reward models rewarding? Dissecting SRMs via pairwise preference probing." AAAI 2025.

3. **Multi-Agent Attacks**: Amayuelas et al. (2024). "MultiAgent Collaboration Attack." EMNLP Findings 2024.

4. **MCTS Adversarial**: Lan et al. (2022). "Are AlphaZero-like agents robust to adversarial perturbations?" NeurIPS 2022.

5. **AgentDojo Benchmark**: Debenedetti et al. (2024). "AgentDojo: A dynamic environment to evaluate prompt injection attacks." NeurIPS 2024.

### Defense Papers

6. **PRIME**: Cui et al. (2025). "Process Reinforcement through Implicit Rewards." arXiv:2502.01456.

7. **PURE**: Cheng et al. (2025). "Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs." NeurIPS 2025.

8. **CRA**: Song et al. (2025). "Causal Reward Adjustment for Mitigating Reward Hacking." arXiv:2508.04216.

### Related SPY Lab Work

9. **RLHF Poisoning**: Rando & Tramèr (2024). "Universal jailbreak backdoors from poisoned human feedback." ICLR 2024.

10. **CoT Monitorability**: Zolkowski et al. (2025). "Can reasoning models obfuscate reasoning?" arXiv:2510.19851.

## 📄 Citation

```bibtex
@misc{zaheer2025ttc,
  title={Mechanistic Attacks on Test-Time Compute in Reasoning LLMs},
  author={Zaheer, H M Shujaat},
  year={2025},
  url={https://github.com/hmshujaatzaheer/ttc-security-attacks}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This research is conducted for academic purposes to improve AI safety. The attacks demonstrated here are intended to:
- Identify vulnerabilities in reasoning mechanisms
- Inform the development of robust defenses
- Contribute to responsible AI development

Please use this code responsibly and in accordance with applicable laws and ethical guidelines.

## 👤 Author

**H M Shujaat Zaheer**
- Email: shujabis@gmail.com
- GitHub: [@hmshujaatzaheer](https://github.com/hmshujaatzaheer)

## 🙏 Acknowledgments

This work builds upon research from:
- SPY Lab (ETH Zürich)
- OpenAI Safety Team
- DeepSeek AI
- The broader AI safety research community
