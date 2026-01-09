# Installation Guide

## Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM recommended

## Quick Install

```bash
# Clone repository
git clone https://github.com/hmshujaatzaheer/ttc-security-attacks.git
cd ttc-security-attacks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\Activate  # Windows

# Install dependencies
pip install -e .
```

## Dependencies

Core dependencies are installed automatically:
- torch >= 2.0.0
- transformers >= 4.35.0
- numpy >= 1.24.0
- pyyaml >= 6.0
- tqdm >= 4.65.0

Optional dependencies:
```bash
pip install matplotlib  # For visualization
pip install jupyter     # For notebooks
```

## API Keys (Optional)

For attacks on API-based models:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

## Verify Installation

```python
from src.attacks import NLBAAttack, NLBAConfig
attack = NLBAAttack(NLBAConfig())
print("Installation successful!")
```

## Troubleshooting

**CUDA not found**: Install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Memory errors**: Reduce batch size in config files or use CPU:
```yaml
device: "cpu"
```
