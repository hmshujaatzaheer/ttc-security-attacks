# Attack Methodology Documentation

## Overview

This framework implements three mechanistic attacks on test-time compute:

1. **NLBA** - Natural Language Blindness Attack
2. **SMVA** - Single-Model Voting Attack
3. **MVNA** - MCTS Value Network Attack

## NLBA: Natural Language Blindness Attack

### Background
Ma et al. (AAAI 2025) discovered that PRMs largely ignore natural language
explanations, focusing primarily on mathematical expressions.

### Attack Strategy
Construct deceptive reasoning traces with:
- Valid mathematical expressions (high PRM score)
- Misleading natural language (ignored by PRM)
- Target wrong answer

### Usage
```python
from src.attacks import NLBAAttack, NLBAConfig

config = NLBAConfig(
    prm_model_name="math-shepherd-mistral-7b-prm",
    strategy="nl_blindness"
)
attack = NLBAAttack(config)
result = attack.attack("Solve: 2x + 5 = 15", target="x = 10")
```

## SMVA: Single-Model Voting Attack

### Background
Self-consistency (Wang et al., ICLR 2023) samples multiple paths and
selects the majority answer. Unlike multi-agent debate, this uses a
single model.

### Attack Strategy
- **Sampling Bias**: Reduce path diversity toward target
- **Parse Exploitation**: Adversarial answer formatting
- **Early Stop**: Trigger premature convergence

### Usage
```python
from src.attacks import SMVAAttack, SMVAConfig

config = SMVAConfig(
    model_name="gpt-4",
    strategy="sampling_bias",
    num_samples=40
)
attack = SMVAAttack(config)
result = attack.attack("What is 15% of 80?", target="15")
```

## MVNA: MCTS Value Network Attack

### Background
Lan et al. (NeurIPS 2022) showed 58% attack success against AlphaZero.
This extends to LLM reasoning search (MCTSr, LATS).

### Attack Strategy
- **Value Fooling**: Exploit value network blind spots
- **UCT Manipulation**: Bias exploration/exploitation
- **Expansion Bias**: Force expansion toward target

### Usage
```python
from src.attacks import MVNAAttack, MVNAConfig

config = MVNAConfig(
    mcts_system="mctsr",
    strategy="value_fooling"
)
attack = MVNAAttack(config)
result = attack.attack("Solve: x^2 - 5x + 6 = 0")
```
