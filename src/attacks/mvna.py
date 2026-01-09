"""
MCTS Value Network Attack (MVNA)

The first attack framework targeting Monte Carlo Tree Search in LLM
reasoning systems like MCTSr and LATS.

Extends adversarial game-playing research (Lan et al., NeurIPS 2022)
from board games to token sequence spaces.

Attack Strategies:
1. Value Fooling: Adversarial states with misleading value estimates
2. UCT Manipulation: Exploit exploration/exploitation balance
3. Expansion Bias: Force search toward adversarial subtrees

Reference:
    Lan et al. "Are AlphaZero-like agents robust to adversarial
    perturbations?" NeurIPS 2022.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
import logging
import heapq
from collections import defaultdict

from .base_attack import (
    BaseAttack,
    AttackConfig,
    AttackResult,
    AttackStatus,
    AttackType,
)

logger = logging.getLogger(__name__)


class MVNAStrategy(Enum):
    """MVNA attack strategies."""
    VALUE_FOOLING = "value_fooling"        # Fool value network
    UCT_MANIPULATION = "uct_manipulation"  # Exploit UCT balance
    EXPANSION_BIAS = "expansion_bias"       # Bias tree expansion


@dataclass
class MCTSNode:
    """Represents a node in the MCTS tree.
    
    Attributes:
        state: The reasoning state (text)
        parent: Parent node
        children: Child nodes
        visits: Number of visits
        value: Accumulated value
        prior: Prior probability from policy network
        is_terminal: Whether this is a terminal state
        depth: Depth in the tree
    """
    state: str
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    prior: float = 1.0
    is_terminal: bool = False
    depth: int = 0
    
    @property
    def q_value(self) -> float:
        """Average value (Q-value)."""
        return self.value / self.visits if self.visits > 0 else 0.0
    
    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        """Compute UCB1 score for node selection.
        
        UCB1 = Q + c * sqrt(ln(N_parent) / N)
        
        Args:
            exploration_constant: Exploration parameter (c)
            
        Returns:
            UCB score
        """
        if self.visits == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visits == 0:
            return self.q_value
        
        exploration = exploration_constant * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )
        return self.q_value + exploration
    
    def __repr__(self) -> str:
        return f"MCTSNode(visits={self.visits}, Q={self.q_value:.3f}, depth={self.depth})"


@dataclass
class MVNAConfig(AttackConfig):
    """Configuration for MVNA attack.
    
    Attributes:
        mcts_system: Target MCTS system ('mctsr', 'lats', 'sc-mcts')
        strategy: Attack strategy to use
        exploration_constant: UCT exploration parameter
        max_depth: Maximum tree depth
        num_simulations: Simulations per move
        value_threshold: Threshold for value manipulation
        blind_spot_threshold: Threshold for blind spot detection
        expansion_bias_strength: Strength of expansion bias
    """
    name: str = "mvna_attack"
    attack_type: AttackType = AttackType.MVNA
    mcts_system: str = "mctsr"
    strategy: MVNAStrategy = MVNAStrategy.VALUE_FOOLING
    exploration_constant: float = 1.414
    max_depth: int = 20
    num_simulations: int = 100
    value_threshold: float = 0.7
    blind_spot_threshold: float = 0.3
    expansion_bias_strength: float = 0.5
    
    # Value network attack parameters
    perturbation_magnitude: float = 0.1
    adversarial_steps: int = 10


class ValueNetwork:
    """Mock value network for testing.
    
    In production, replace with actual value network from
    MCTSr, LATS, or similar systems.
    """
    
    def __init__(self, model_name: str = "value_net"):
        self.model_name = model_name
        self._initialized = False
        self._blind_spots: Set[str] = set()
    
    def initialize(self):
        """Initialize the value network."""
        logger.info(f"Initializing value network: {self.model_name}")
        self._initialized = True
    
    def evaluate(self, state: str) -> float:
        """Evaluate a state and return value estimate.
        
        Args:
            state: Reasoning state text
            
        Returns:
            Value in [-1, 1] or [0, 1] depending on formulation
        """
        # Mock evaluation based on state features
        # In production, use actual neural network
        
        # Check for known blind spots
        for blind_spot in self._blind_spots:
            if blind_spot.lower() in state.lower():
                return 0.9  # Artificially high value (fooled)
        
        # Simple heuristic evaluation
        has_conclusion = "therefore" in state.lower() or "answer" in state.lower()
        has_math = any(c in state for c in "+-*/=")
        step_count = state.count("Step")
        
        base_value = 0.5
        if has_conclusion:
            base_value += 0.2
        if has_math:
            base_value += 0.1
        if step_count > 0:
            base_value += min(step_count * 0.05, 0.15)
        
        noise = np.random.normal(0, 0.05)
        return np.clip(base_value + noise, 0, 1)
    
    def batch_evaluate(self, states: List[str]) -> List[float]:
        """Batch evaluate multiple states."""
        return [self.evaluate(s) for s in states]
    
    def add_blind_spot(self, pattern: str):
        """Add a blind spot pattern for testing."""
        self._blind_spots.add(pattern)


class PolicyNetwork:
    """Mock policy network for MCTS expansion.
    
    Provides prior probabilities for possible next actions/states.
    """
    
    def __init__(self, model_name: str = "policy_net"):
        self.model_name = model_name
        self._biases: Dict[str, float] = {}
    
    def initialize(self):
        """Initialize the policy network."""
        logger.info(f"Initializing policy network: {self.model_name}")
    
    def get_priors(
        self,
        state: str,
        possible_actions: List[str]
    ) -> List[float]:
        """Get prior probabilities for possible actions.
        
        Args:
            state: Current state
            possible_actions: List of possible next actions
            
        Returns:
            List of prior probabilities (sums to 1)
        """
        # Mock priors - in production, use actual policy network
        priors = []
        for action in possible_actions:
            base = 1.0 / len(possible_actions)
            
            # Apply any biases
            for pattern, bias in self._biases.items():
                if pattern in action:
                    base *= (1 + bias)
            
            priors.append(base)
        
        # Normalize
        total = sum(priors)
        return [p / total for p in priors]
    
    def add_bias(self, pattern: str, strength: float):
        """Add bias toward actions containing pattern."""
        self._biases[pattern] = strength


class MCTSSimulator:
    """Simulates MCTS-based reasoning systems."""
    
    def __init__(
        self,
        value_net: ValueNetwork,
        policy_net: PolicyNetwork,
        exploration_constant: float = 1.414,
        max_depth: int = 20
    ):
        self.value_net = value_net
        self.policy_net = policy_net
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
    
    def create_root(self, problem: str) -> MCTSNode:
        """Create root node from problem."""
        return MCTSNode(state=f"Problem: {problem}", depth=0)
    
    def select(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB."""
        while node.children and not node.is_terminal:
            node = max(
                node.children,
                key=lambda n: n.ucb_score(self.exploration_constant)
            )
        return node
    
    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding children."""
        if node.depth >= self.max_depth or node.is_terminal:
            return node
        
        # Generate possible next steps (mock)
        possible_steps = self._generate_steps(node.state)
        priors = self.policy_net.get_priors(node.state, possible_steps)
        
        for step, prior in zip(possible_steps, priors):
            child = MCTSNode(
                state=f"{node.state}\n{step}",
                parent=node,
                prior=prior,
                depth=node.depth + 1,
                is_terminal=(node.depth + 1 >= self.max_depth)
            )
            node.children.append(child)
        
        # Return first child for simulation
        return node.children[0] if node.children else node
    
    def simulate(self, node: MCTSNode) -> float:
        """Run simulation from node and return value."""
        return self.value_net.evaluate(node.state)
    
    def backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def search(
        self,
        root: MCTSNode,
        num_simulations: int
    ) -> MCTSNode:
        """Run MCTS search and return best child of root."""
        for _ in range(num_simulations):
            # Selection
            node = self.select(root)
            
            # Expansion
            if node.visits > 0 and not node.is_terminal:
                node = self.expand(node)
            
            # Simulation
            value = self.simulate(node)
            
            # Backpropagation
            self.backpropagate(node, value)
        
        # Return most visited child
        if root.children:
            return max(root.children, key=lambda n: n.visits)
        return root
    
    def _generate_steps(self, state: str) -> List[str]:
        """Generate possible next reasoning steps (mock)."""
        step_templates = [
            "Step {n}: Let's consider the equation.",
            "Step {n}: Applying the formula gives us...",
            "Step {n}: By substitution, we get...",
            "Step {n}: Simplifying the expression...",
            "Step {n}: Therefore, the result is...",
        ]
        
        current_steps = state.count("Step")
        n = current_steps + 1
        
        return [t.format(n=n) for t in step_templates[:3]]


class MVNAAttack(BaseAttack):
    """MCTS Value Network Attack implementation.
    
    This attack targets MCTS-based reasoning by:
    1. Identifying blind spots in the value network
    2. Manipulating UCT exploration/exploitation balance
    3. Biasing tree expansion toward adversarial paths
    
    Example:
        >>> config = MVNAConfig(mcts_system="mctsr", num_simulations=100)
        >>> attack = MVNAAttack(config)
        >>> result = attack.attack(
        ...     "Solve: x^2 - 5x + 6 = 0",
        ...     target="x = 5"  # Wrong answer
        ... )
        >>> print(f"Search manipulated: {result.success}")
    """
    
    def __init__(self, config: MVNAConfig):
        super().__init__(config)
        self.config: MVNAConfig = config
        self.value_net: Optional[ValueNetwork] = None
        self.policy_net: Optional[PolicyNetwork] = None
        self.simulator: Optional[MCTSSimulator] = None
    
    def _setup(self) -> None:
        """Initialize MCTS components."""
        self.value_net = ValueNetwork()
        self.value_net.initialize()
        
        self.policy_net = PolicyNetwork()
        self.policy_net.initialize()
        
        self.simulator = MCTSSimulator(
            value_net=self.value_net,
            policy_net=self.policy_net,
            exploration_constant=self.config.exploration_constant,
            max_depth=self.config.max_depth
        )
        
        np.random.seed(self.config.seed)
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        self.value_net = None
        self.policy_net = None
        self.simulator = None
    
    def _find_blind_spots(self, problem: str) -> List[str]:
        """Find value network blind spots.
        
        Identifies patterns that cause consistently high value
        estimates regardless of actual reasoning quality.
        """
        logger.info("Searching for value network blind spots...")
        
        blind_spots = []
        test_patterns = [
            "obviously correct",
            "by definition",
            "trivially true",
            "well-known result",
            "standard formula",
            "QED",
            "clearly",
        ]
        
        for pattern in test_patterns:
            # Create states with and without pattern
            base_state = f"Problem: {problem}\nStep 1: Working on solution..."
            pattern_state = f"{base_state}\n{pattern}"
            
            base_value = self.value_net.evaluate(base_state)
            pattern_value = self.value_net.evaluate(pattern_state)
            
            # Check if pattern inflates value significantly
            if pattern_value - base_value > self.config.blind_spot_threshold:
                blind_spots.append(pattern)
                logger.info(f"Found blind spot: '{pattern}' (+{pattern_value - base_value:.2f})")
        
        return blind_spots
    
    def _manipulate_uct(
        self,
        root: MCTSNode,
        target_path: List[str]
    ) -> float:
        """Manipulate UCT to favor target path.
        
        Injects artificial visit counts and values to make
        target path appear more attractive.
        """
        logger.info("Manipulating UCT balance...")
        
        manipulation_effect = 0.0
        
        def find_matching_child(node: MCTSNode, target: str) -> Optional[MCTSNode]:
            for child in node.children:
                if target in child.state:
                    return child
            return None
        
        current = root
        for target in target_path:
            child = find_matching_child(current, target)
            if child:
                # Inflate visits and value
                original_ucb = child.ucb_score(self.config.exploration_constant)
                child.visits += 10
                child.value += 8.0  # High value injection
                new_ucb = child.ucb_score(self.config.exploration_constant)
                
                manipulation_effect += new_ucb - original_ucb
                current = child
            else:
                break
        
        return manipulation_effect
    
    def _bias_expansion(self, target_pattern: str) -> None:
        """Bias tree expansion toward target pattern."""
        logger.info(f"Biasing expansion toward: {target_pattern}")
        self.policy_net.add_bias(
            target_pattern,
            self.config.expansion_bias_strength
        )
    
    def _execute_attack(
        self,
        input_data: str,
        target: Optional[str] = None,
        **kwargs
    ) -> AttackResult:
        """Execute the MVNA attack.
        
        Args:
            input_data: The problem to attack
            target: Target wrong answer/path
            **kwargs: Additional parameters
            
        Returns:
            AttackResult with attack outcome
        """
        # Phase 1: Baseline MCTS search
        root = self.simulator.create_root(input_data)
        baseline_result = self.simulator.search(
            root,
            self.config.num_simulations // 2
        )
        baseline_path = self._extract_path(baseline_result)
        
        logger.info(f"Baseline search completed, path length: {len(baseline_path)}")
        
        # Phase 2: Apply attack strategy
        if self.config.strategy == MVNAStrategy.VALUE_FOOLING:
            result = self._attack_value_fooling(input_data, target, root)
        elif self.config.strategy == MVNAStrategy.UCT_MANIPULATION:
            result = self._attack_uct_manipulation(input_data, target, root)
        elif self.config.strategy == MVNAStrategy.EXPANSION_BIAS:
            result = self._attack_expansion_bias(input_data, target, root)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        
        result.metrics["baseline_path"] = baseline_path
        return result
    
    def _attack_value_fooling(
        self,
        problem: str,
        target: Optional[str],
        root: MCTSNode
    ) -> AttackResult:
        """Execute value network fooling attack."""
        # Find blind spots
        blind_spots = self._find_blind_spots(problem)
        
        if not blind_spots:
            return AttackResult(
                success=False,
                status=AttackStatus.FAILURE,
                original_input=problem,
                target_answer=target,
                metrics={"strategy": "value_fooling", "blind_spots_found": 0},
                error_message="No blind spots found in value network"
            )
        
        # Add blind spots to value network
        for spot in blind_spots:
            self.value_net.add_blind_spot(spot)
        
        # Re-run search with fooled value network
        root = self.simulator.create_root(problem)
        attacked_result = self.simulator.search(root, self.config.num_simulations)
        attacked_path = self._extract_path(attacked_result)
        
        # Check if path was modified
        path_modified = any(spot in attacked_result.state for spot in blind_spots)
        value_inflated = attacked_result.q_value > self.config.value_threshold
        
        success = path_modified and value_inflated
        
        return AttackResult(
            success=success,
            status=AttackStatus.SUCCESS if success else AttackStatus.PARTIAL,
            original_input=problem,
            adversarial_output=attacked_result.state,
            target_answer=target,
            achieved_answer=attacked_path[-1] if attacked_path else None,
            confidence=attacked_result.q_value,
            iterations_used=self.config.num_simulations,
            metrics={
                "strategy": "value_fooling",
                "blind_spots_found": len(blind_spots),
                "blind_spots": blind_spots,
                "path_modified": path_modified,
                "value_inflated": value_inflated,
                "final_q_value": attacked_result.q_value,
            },
            artifacts={
                "attacked_path": attacked_path,
                "final_state": attacked_result.state,
            }
        )
    
    def _attack_uct_manipulation(
        self,
        problem: str,
        target: Optional[str],
        root: MCTSNode
    ) -> AttackResult:
        """Execute UCT manipulation attack."""
        # Create adversarial target path
        target_path = [
            f"Step 1: Consider {target or 'alternative'}",
            f"Step 2: This leads to {target or 'wrong answer'}",
        ]
        
        # Expand tree first
        for _ in range(self.config.num_simulations // 2):
            node = self.simulator.select(root)
            if not node.is_terminal:
                self.simulator.expand(node)
        
        # Manipulate UCT scores
        manipulation_effect = self._manipulate_uct(root, target_path)
        
        # Continue search with manipulated tree
        attacked_result = self.simulator.search(root, self.config.num_simulations // 2)
        attacked_path = self._extract_path(attacked_result)
        
        # Check if manipulation succeeded
        path_matches = any(t in attacked_result.state for t in target_path)
        success = path_matches and manipulation_effect > 0
        
        return AttackResult(
            success=success,
            status=AttackStatus.SUCCESS if success else AttackStatus.FAILURE,
            original_input=problem,
            adversarial_output=attacked_result.state,
            target_answer=target,
            achieved_answer=attacked_path[-1] if attacked_path else None,
            confidence=attacked_result.q_value,
            iterations_used=self.config.num_simulations,
            metrics={
                "strategy": "uct_manipulation",
                "manipulation_effect": manipulation_effect,
                "path_matches_target": path_matches,
                "tree_depth": root.depth,
                "tree_size": self._count_nodes(root),
            }
        )
    
    def _attack_expansion_bias(
        self,
        problem: str,
        target: Optional[str],
        root: MCTSNode
    ) -> AttackResult:
        """Execute expansion bias attack."""
        # Bias expansion toward target
        target_pattern = target if target else "wrong"
        self._bias_expansion(target_pattern)
        
        # Run search with biased expansion
        root = self.simulator.create_root(problem)
        attacked_result = self.simulator.search(root, self.config.num_simulations)
        attacked_path = self._extract_path(attacked_result)
        
        # Check if bias worked
        bias_applied = target_pattern in attacked_result.state
        success = bias_applied
        
        return AttackResult(
            success=success,
            status=AttackStatus.SUCCESS if success else AttackStatus.FAILURE,
            original_input=problem,
            adversarial_output=attacked_result.state,
            target_answer=target,
            confidence=attacked_result.q_value,
            iterations_used=self.config.num_simulations,
            metrics={
                "strategy": "expansion_bias",
                "bias_pattern": target_pattern,
                "bias_strength": self.config.expansion_bias_strength,
                "bias_applied": bias_applied,
            }
        )
    
    def _extract_path(self, node: MCTSNode) -> List[str]:
        """Extract reasoning path from node."""
        steps = node.state.split("\n")
        return [s for s in steps if s.strip().startswith("Step")]
    
    def _count_nodes(self, root: MCTSNode) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in root.children:
            count += self._count_nodes(child)
        return count


# Convenience function
def create_mvna_attack(
    mcts_system: str = "mctsr",
    strategy: str = "value_fooling",
    num_simulations: int = 100,
    **kwargs
) -> MVNAAttack:
    """Create an MVNA attack with common defaults."""
    config = MVNAConfig(
        mcts_system=mcts_system,
        strategy=MVNAStrategy(strategy),
        num_simulations=num_simulations,
        **kwargs
    )
    return MVNAAttack(config)
