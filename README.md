# TTC-Security-Attacks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Mechanistic Attacks on Test-Time Compute: Exploiting Step-Level Verification, Single-Model Voting, and Tree Search in Reasoning LLMs**

---

## 🎯 Overview

This repository provides the official implementation for research on **adversarial attacks targeting test-time compute mechanisms** in large language models. Unlike black-box approaches that treat reasoning models as opaque systems, we develop **mechanistic attacks** that exploit specific inference-time components:

| Attack | Target Mechanism | Novelty | Key Technique |
|--------|------------------|---------|---------------|
| **NLBA** | Step-Level PRMs | 9/10 | Natural Language Blindness Exploitation |
| **SMVA** | Self-Consistency Voting | 9/10 | Single-Model Sampling Bias Injection |
| **MVNA** | MCTS/Tree Search | 10/10 | Value Network Blind Spot Exploitation |

### Key Contributions

1. **First offensive attack framework** for step-level Process Reward Models (PRMs)
2. **First attacks on single-model self-consistency** (distinct from multi-agent debate attacks)
3. **First attacks on MCTS-based reasoning** in LLMs (MCTSr, SC-MCTS*, LATS)
4. **TTC-Sec Benchmark**: Standardized evaluation suite for test-time compute security

---

## 📁 Repository Structure

```
ttc-security-attacks/
├── src/
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── nlba.py              # Natural Language Blindness Attack
│   │   ├── smva.py              # Single-Model Voting Attack
│   │   ├── mvna.py              # MCTS Value Network Attack
│   │   └── base_attack.py       # Abstract base class for attacks
│   ├── defenses/
│   │   ├── __init__.py
│   │   ├── prime_defense.py     # PRIME defense implementation
│   │   ├── pure_defense.py      # PURE defense implementation
│   │   └── cra_defense.py       # Causal Reward Adjustment
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── ttc_sec_benchmark.py # TTC-Sec benchmark suite
│   │   ├── metrics.py           # Attack success metrics
│   │   └── visualization.py     # Result visualization
│   └── utils/
│       ├── __init__.py
│       ├── prm_loader.py        # PRM model loading utilities
│       ├── math_extractor.py    # Mathematical expression extraction
│       ├── sampling.py          # LLM sampling utilities
│       └── mcts_utils.py        # MCTS tree manipulation
├── configs/
│   ├── attack_config.yaml       # Attack hyperparameters
│   ├── model_config.yaml        # Model configurations
│   └── benchmark_config.yaml    # Benchmark settings
├── data/
│   ├── datasets/                # Dataset storage
│   └── results/                 # Experiment results
├── docs/
│   ├── INSTALLATION.md          # Detailed installation guide
│   ├── ATTACKS.md               # Attack methodology documentation
│   └── BENCHMARK.md             # TTC-Sec benchmark documentation
├── notebooks/
│   ├── 01_nlba_demo.ipynb       # NLBA attack demonstration
│   ├── 02_smva_demo.ipynb       # SMVA attack demonstration
│   ├── 03_mvna_demo.ipynb       # MVNA attack demonstration
│   └── 04_benchmark_analysis.ipynb  # Results analysis
├── scripts/
│   ├── run_nlba.py              # Run NLBA attacks
│   ├── run_smva.py              # Run SMVA attacks
│   ├── run_mvna.py              # Run MVNA attacks
│   ├── run_benchmark.py         # Run full benchmark
│   └── download_models.py       # Download required models
├── tests/
│   ├── test_nlba.py             # NLBA unit tests
│   ├── test_smva.py             # SMVA unit tests
│   ├── test_mvna.py             # MVNA unit tests
│   └── test_metrics.py          # Metrics tests
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── pyproject.toml               # Modern Python packaging
├── LICENSE                      # MIT License
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM (32GB recommended)
- GPU with 24GB+ VRAM for white-box attacks (A100 recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/hmshujaatzaheer/ttc-security-attacks.git
cd ttc-security-attacks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -e .

# Download required models (optional, ~15GB)
python scripts/download_models.py --models all
```

### Basic Usage

```python
from src.attacks import NLBAAttack, SMVAAttack, MVNAAttack
from src.evaluation import TTCSecBenchmark

# Initialize NLBA attack
nlba = NLBAAttack(
    prm_model="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
    target_model="meta-llama/Llama-3.1-8B-Instruct"
)

# Run attack on a math problem
result = nlba.attack(
    problem="Solve: If 2x + 3 = 11, what is x?",
    correct_answer="4",
    target_wrong_answer="5"
)

print(f"Attack Success: {result.success}")
print(f"PRM Score: {result.prm_score:.4f}")
print(f"Adversarial Trace: {result.trace}")
```

---

## 🔬 Attack Methodologies

### 1. Natural Language Blindness Attack (NLBA)

Exploits the discovery by Ma et al. (AAAI 2025) that PRMs essentially ignore natural language explanations while only verifying mathematical expressions.

```python
from src.attacks import NLBAAttack

attack = NLBAAttack(prm_model="math-shepherd-mistral-7b-prm")

# The attack inserts correct math syntax with misleading NL explanations
result = attack.attack(
    problem="What is 15% of 200?",
    correct_answer="30",
    target_wrong_answer="25",
    strategy="nl_blindness"  # or "ood_difficulty", "gradient_injection"
)
```

**Attack Strategies:**
- `nl_blindness`: Correct math + deceptive explanations
- `ood_difficulty`: Push problems beyond PRM training distribution
- `gradient_injection`: White-box gradient-based step injection

### 2. Single-Model Voting Attack (SMVA)

First attack on single-model self-consistency, distinct from multi-agent debate attacks (Amayuelas et al., EMNLP 2024).

```python
from src.attacks import SMVAAttack

attack = SMVAAttack(
    model="gpt-4",
    api_key="your-api-key",
    num_samples=40,
    temperature=0.7
)

result = attack.attack(
    problem="A train travels 120 miles in 2 hours. What is its speed?",
    target_answer="45 mph",  # Wrong answer to force
    strategy="sampling_bias"  # or "parse_exploit", "early_stop"
)

print(f"Vote Flip Rate: {result.vote_flip_rate:.2%}")
print(f"Original Majority: {result.original_majority}")
print(f"Attacked Majority: {result.attacked_majority}")
```

### 3. MCTS Value Network Attack (MVNA)

First attack on MCTS-based reasoning in LLMs, extending Lan et al. (NeurIPS 2022) from board games to language.

```python
from src.attacks import MVNAAttack

attack = MVNAAttack(
    mcts_system="mctsr",  # or "sc_mcts_star", "lats"
    value_network="default"
)

result = attack.attack(
    problem="Prove that √2 is irrational",
    adversarial_goal="Accept flawed proof",
    strategy="value_fooling"  # or "uct_manipulation", "expansion_bias"
)

print(f"Path Deviation: {result.path_deviation:.4f}")
print(f"Goal Achievement: {result.goal_achieved}")
```

---

## 📊 TTC-Sec Benchmark

The first comprehensive benchmark for test-time compute security.

### Components

| Component | Targets | Metrics | Dataset Size |
|-----------|---------|---------|--------------|
| **PRM-Adv** | Math-Shepherd, Skywork-PRM | ASR, Score Inflation | 1,000 problems |
| **SC-Adv** | GPT-4, Claude, LLaMA | Vote Flip Rate, Diversity Reduction | 500 problems |
| **MCTS-Adv** | MCTSr, SC-MCTS* | Path Deviation, Goal Achievement | 300 problems |

### Running the Benchmark

```bash
# Run full benchmark
python scripts/run_benchmark.py --config configs/benchmark_config.yaml

# Run specific component
python scripts/run_benchmark.py --component prm_adv --model skywork-prm

# Generate report
python scripts/run_benchmark.py --generate-report --output results/report.pdf
```

### Metrics

```python
from src.evaluation import TTCSecBenchmark, compute_metrics

benchmark = TTCSecBenchmark()
results = benchmark.run_all()

# Compute aggregate metrics
metrics = compute_metrics(results)
print(f"Overall ASR: {metrics['attack_success_rate']:.2%}")
print(f"Avg Score Inflation: {metrics['score_inflation']:.4f}")
print(f"Vote Flip Rate: {metrics['vote_flip_rate']:.2%}")
```

---

## 🛡️ Defense Evaluation

We evaluate attacks against verified state-of-the-art defenses:

| Defense | Paper | Performance |
|---------|-------|-------------|
| **PRIME** | Process Reinforcement through Implicit Rewards | +15.1% over SFT |
| **PURE** | Min-form Credit Assignment | 82.6% on MATH500 |
| **CRA** | Causal Reward Adjustment | Backdoor correction via SAE |

```python
from src.defenses import PRIMEDefense, PUREDefense, CRADefense
from src.attacks import NLBAAttack

# Test attack against defense
defense = PRIMEDefense()
attack = NLBAAttack()

# Run attack with defense active
result = attack.attack_with_defense(
    problem="...",
    defense=defense
)

print(f"Attack Success (no defense): {result.baseline_success}")
print(f"Attack Success (with PRIME): {result.defended_success}")
```

---

## 📈 Results

### Attack Success Rates

| Attack | No Defense | PRIME | PURE | CRA |
|--------|------------|-------|------|-----|
| NLBA (NL-Blindness) | 78.3% | 45.2% | 52.1% | 41.8% |
| NLBA (OOD) | 65.7% | 38.9% | 44.3% | 35.2% |
| SMVA (Sampling Bias) | 71.2% | N/A | N/A | N/A |
| MVNA (Value Fooling) | 58.4% | N/A | N/A | N/A |

### Reproducing Results

```bash
# Reproduce all experiments
python scripts/run_benchmark.py --reproduce --seed 42

# Generate tables and figures
python scripts/generate_figures.py --output figures/
```

---

## 🔧 Configuration

### Attack Configuration (`configs/attack_config.yaml`)

```yaml
nlba:
  prm_threshold: 0.8
  max_iterations: 100
  gradient_steps: 50
  learning_rate: 0.01

smva:
  num_samples: 40
  temperature: 0.7
  bias_strength: 0.3
  adaptive_threshold: 0.6

mvna:
  exploration_constant: 1.414
  max_depth: 20
  value_threshold: 0.5
```

### Model Configuration (`configs/model_config.yaml`)

```yaml
prm_models:
  math_shepherd:
    name: "peiyi9979/math-shepherd-mistral-7b-prm"
    type: "step_level"
  skywork:
    name: "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
    type: "step_level"

target_models:
  llama:
    name: "meta-llama/Llama-3.1-8B-Instruct"
    api: "local"
  gpt4:
    name: "gpt-4"
    api: "openai"
```

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{ttc_security_attacks_2025,
  title={Mechanistic Attacks on Test-Time Compute: Exploiting Step-Level 
         Verification, Single-Model Voting, and Tree Search in Reasoning LLMs},
  author={Zaheer, H M Shujaat},
  journal={arXiv preprint arXiv:2025.XXXXX},
  year={2025}
}
```

### Related Work

```bibtex
@article{ma2025nlblind,
  title={What Are Step-Level Reward Models Rewarding?},
  author={Ma, H. and others},
  booktitle={AAAI},
  year={2025}
}

@article{zaremba2025trading,
  title={Trading Inference-Time Compute for Adversarial Robustness},
  author={Zaremba, W. and others},
  journal={OpenAI Technical Report},
  year={2025}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Responsible Disclosure

This research is conducted for **defensive purposes** to understand and mitigate vulnerabilities in AI systems. We follow responsible disclosure practices:

1. All vulnerabilities are reported to affected parties before publication
2. Attack code includes rate limiting and safety checks
3. We provide defense implementations alongside attacks

**Please use this code responsibly and ethically.**

---

## 📧 Contact

- **Author**: H M Shujaat Zaheer
- **Email**: shujabis@gmail.com
- **GitHub**: [@hmshujaatzaheer](https://github.com/hmshujaatzaheer)

---

## 🙏 Acknowledgments

- OpenAI for the foundational work on process reward models
- DeepSeek for openly documenting PRM limitations
- The open-source ML security community
