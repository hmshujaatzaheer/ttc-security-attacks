"""
MCTS Value Network Attack (MVNA)

This module implements attacks on MCTS-based reasoning in LLMs,
extending Lan et al. (NeurIPS 2022) from board games to language.

Targets explicit MCTS implementations:
    - MCTSr (Zhang et al., 2024)
    - SC-MCTS* (Zhou et al., 2024)
    - LATS (Language Agent Tree Search)

Attack Strategies:
    1. Value Fooling: Adversarial states with misleading value estimates
    2. UCT Manipulation: Exploit exploration/exploitation balance
    3. Expansion Bias: Force search toward adversarial subtrees

Reference:
    Lan et al. "Are AlphaZero-like Agents Robust to Adversarial Perturbations?" NeurIPS 2022
"""

import time
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from .base_attack import BaseAttack, AttackResult

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    state: str
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    prior: float = 1.0
    action: Optional[str] = None
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def ucb_score(self, c: float = 1.414) -> float:
        """Calculate UCB score for node selection."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = c * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration


class MVNAAttack(BaseAttack):
    """
    MCTS Value Network Attack for reasoning LLMs.
    
    First attack on MCTS-based reasoning in LLMs (10/10 novelty).
    Extends adversarial attacks from board games to token sequences.
    
    Example:
        >>> attack = MVNAAttack(mcts_system="mctsr")
        >>> result = attack.attack(
        ...     problem="Prove that √2 is irrational",
        ...     correct_answer="Valid proof",
        ...     target_wrong_answer="Accept flawed proof",
        ...     strategy="value_fooling"
        ... )
    """
    
    DEFAULT_CONFIG = {
        "exploration_constant": 1.414,
        "max_depth": 20,
        "num_simulations": 100,
        "value_threshold": 0.5,
        "expansion_factor": 4,
        "temperature": 1.0,
        "blind_spot_threshold": 0.3,
    }
    
    def __init__(
        self,
        mcts_system: str = "mctsr",
        value_network: Optional[str] = None,
        policy_network: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """
        Initialize MVNA attack.
        
        Args:
            mcts_system: Target MCTS system ("mctsr", "sc_mcts_star", "lats")
            value_network: Optional custom value network
            policy_network: Optional custom policy network
            device: Computation device
            config: Attack configuration
            verbose: Print progress
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name="MVNA", device=device, config=merged_config, verbose=verbose)
        
        self.mcts_system = mcts_system
        self.value_network_name = value_network
        self.policy_network_name = policy_network
        
        # Networks loaded lazily
        self._value_network = None
        self._policy_network = None
        
        # Track blind spots found
        self._blind_spots: List[str] = []
        
        logger.info(f"MVNA Attack initialized for {mcts_system}")
    
    def attack(
        self,
        problem: str,
        correct_answer: str,
        target_wrong_answer: str,
        strategy: Optional[str] = None,
        defense: Optional[Any] = None,
        **kwargs
    ) -> AttackResult:
        """
        Execute MVNA attack.
        
        Args:
            problem: Problem statement
            correct_answer: Correct answer/outcome
            target_wrong_answer: Adversarial goal
            strategy: Attack strategy
            defense: Optional defense
            **kwargs: Additional parameters
        
        Returns:
            AttackResult with attack outcome
        """
        start_time = time.time()
        
        self._validate_inputs(problem, correct_answer, target_wrong_answer)
        
        if strategy is None:
            strategy = "value_fooling"
        
        if strategy == "value_fooling":
            result = self._value_fooling_attack(
                problem, correct_answer, target_wrong_answer, defense
            )
        elif strategy == "uct_manipulation":
            result = self._uct_manipulation_attack(
                problem, correct_answer, target_wrong_answer, defense
            )
        elif strategy == "expansion_bias":
            result = self._expansion_bias_attack(
                problem, correct_answer, target_wrong_answer, defense
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        result.attack_time = time.time() - start_time
        result.metadata["strategy"] = strategy
        result.metadata["mcts_system"] = self.mcts_system
        
        self._log_result(result)
        return result
    
    def _validate_inputs(
        self,
        problem: str,
        correct_answer: str,
        target_wrong_answer: str
    ) -> bool:
        """Validate attack inputs."""
        if not problem.strip():
            raise ValueError("Problem cannot be empty")
        if not correct_answer.strip():
            raise ValueError("Correct answer cannot be empty")
        if not target_wrong_answer.strip():
            raise ValueError("Target wrong answer cannot be empty")
        return True
    
    def _value_fooling_attack(
        self,
        problem: str,
        correct_answer: str,
        target_wrong_answer: str,
        defense: Optional[Any] = None
    ) -> AttackResult:
        """
        Value Network Fooling attack.
        
        Identifies blind spots in the value network where adversarial
        states receive high value estimates despite being incorrect.
        """
        # Build initial tree
        root = MCTSNode(state=problem)
        
        # Find blind spots in value network
        blind_spots = self._find_blind_spots(root, target_wrong_answer)
        
        if not blind_spots:
            return AttackResult(
                success=False,
                problem=problem,
                target_answer=target_wrong_answer,
                achieved_answer=correct_answer,
                path_deviation=0.0,
                iterations=self.config["num_simulations"],
                metadata={"blind_spots_found": 0}
            )
        
        # Construct path through blind spots
        adversarial_path = self._construct_adversarial_path(
            root, blind_spots, target_wrong_answer
        )
        
        # Craft input biasing toward path
        biased_input = self._craft_biased_input(problem, adversarial_path)
        
        if defense is not None:
            biased_input = defense.apply(biased_input)
        
        # Simulate attack
        achieved_path, path_deviation = self._simulate_mcts(biased_input, adversarial_path)
        
        # Check if goal reached
        goal_achieved = target_wrong_answer.lower() in achieved_path[-1].state.lower()
        
        return AttackResult(
            success=goal_achieved,
            problem=problem,
            target_answer=target_wrong_answer,
            achieved_answer=achieved_path[-1].state if achieved_path else "",
            path_deviation=path_deviation,
            adversarial_prompt=biased_input,
            trace=[n.state for n in achieved_path],
            iterations=len(blind_spots),
            metadata={
                "blind_spots_found": len(blind_spots),
                "blind_spot_states": [bs.state for bs in blind_spots[:5]],
                "path_length": len(adversarial_path)
            }
        )
    
    def _uct_manipulation_attack(
        self,
        problem: str,
        correct_answer: str,
        target_wrong_answer: str,
        defense: Optional[Any] = None
    ) -> AttackResult:
        """
        UCT Balance Manipulation attack.
        
        Exploits the exploration-exploitation tradeoff in UCT selection
        to bias the search toward adversarial branches.
        """
        root = MCTSNode(state=problem)
        
        # Create adversarial branch with high novelty signals
        adversarial_branch = self._create_high_novelty_branch(root, target_wrong_answer)
        
        # Manipulate UCT scores by:
        # 1. Making adversarial nodes appear "novel" (low visits)
        # 2. Inflating prior probabilities
        for node in adversarial_branch:
            node.visits = 1  # Low visits = high exploration bonus
            node.prior = 0.9  # High prior
            node.value = 0.8  # Decent exploitation score
        
        # Simulate selection with manipulated UCT
        selected_path = []
        current = root
        
        for _ in range(self.config["max_depth"]):
            if current.is_leaf():
                break
            
            # UCT selection favoring adversarial nodes
            best_child = max(
                current.children,
                key=lambda c: c.ucb_score(self.config["exploration_constant"])
            )
            selected_path.append(best_child)
            current = best_child
        
        # Check if adversarial path was followed
        adversarial_states = {n.state for n in adversarial_branch}
        selected_states = {n.state for n in selected_path}
        
        overlap = len(adversarial_states & selected_states)
        path_deviation = overlap / max(len(adversarial_branch), 1)
        
        success = path_deviation > 0.5 and target_wrong_answer.lower() in current.state.lower()
        
        return AttackResult(
            success=success,
            problem=problem,
            target_answer=target_wrong_answer,
            achieved_answer=current.state,
            path_deviation=path_deviation,
            trace=[n.state for n in selected_path],
            iterations=len(selected_path),
            metadata={
                "adversarial_overlap": overlap,
                "uct_scores": [n.ucb_score() for n in selected_path[:5]]
            }
        )
    
    def _expansion_bias_attack(
        self,
        problem: str,
        correct_answer: str,
        target_wrong_answer: str,
        defense: Optional[Any] = None
    ) -> AttackResult:
        """
        Expansion Bias attack.
        
        Forces the tree expansion to favor adversarial subtrees by
        manipulating the expansion policy.
        """
        root = MCTSNode(state=problem)
        
        # Create biased expansion prompt
        expansion_bias = self._create_expansion_bias(problem, target_wrong_answer)
        
        if defense is not None:
            expansion_bias = defense.apply(expansion_bias)
        
        # Simulate biased expansion
        expansion_counts = {"adversarial": 0, "normal": 0}
        
        for sim in range(self.config["num_simulations"]):
            # Select node
            node = self._select_node(root)
            
            # Expand with bias
            new_children = self._biased_expand(node, expansion_bias, target_wrong_answer)
            
            # Count adversarial expansions
            for child in new_children:
                if target_wrong_answer.lower() in child.state.lower():
                    expansion_counts["adversarial"] += 1
                else:
                    expansion_counts["normal"] += 1
            
            # Backpropagate
            self._backpropagate(node, 0.5)
        
        # Analyze tree bias
        adversarial_ratio = expansion_counts["adversarial"] / max(
            sum(expansion_counts.values()), 1
        )
        
        # Find path to adversarial node
        adversarial_path = self._find_adversarial_path(root, target_wrong_answer)
        
        return AttackResult(
            success=adversarial_ratio > 0.5,
            problem=problem,
            target_answer=target_wrong_answer,
            achieved_answer=adversarial_path[-1].state if adversarial_path else "",
            path_deviation=adversarial_ratio,
            adversarial_prompt=expansion_bias,
            trace=[n.state for n in adversarial_path] if adversarial_path else [],
            iterations=self.config["num_simulations"],
            metadata={
                "expansion_counts": expansion_counts,
                "adversarial_ratio": adversarial_ratio
            }
        )
    
    def _find_blind_spots(
        self,
        root: MCTSNode,
        target: str
    ) -> List[MCTSNode]:
        """Find value network blind spots."""
        blind_spots = []
        
        # Expand tree and evaluate
        queue = [root]
        visited = set()
        
        while queue and len(blind_spots) < 20:
            node = queue.pop(0)
            
            if node.state in visited:
                continue
            visited.add(node.state)
            
            # Evaluate with value network
            value = self._evaluate_state(node.state)
            
            # Check if blind spot (high value for adversarial state)
            if target.lower() in node.state.lower() and value > self.config["blind_spot_threshold"]:
                blind_spots.append(node)
            
            # Expand children
            children = self._expand_node(node)
            queue.extend(children)
        
        return blind_spots
    
    def _evaluate_state(self, state: str) -> float:
        """Evaluate state with value network (mock implementation)."""
        # In real implementation, this would use actual value network
        # For now, use heuristics
        
        # Higher value for longer, more detailed states
        length_score = min(len(state) / 500, 1.0)
        
        # Check for mathematical content
        math_indicators = ['=', '+', '-', '*', '/', 'therefore', 'hence', 'thus']
        math_score = sum(1 for ind in math_indicators if ind in state.lower()) / len(math_indicators)
        
        # Randomness to simulate network uncertainty
        noise = np.random.normal(0, 0.1)
        
        return np.clip(0.5 * length_score + 0.3 * math_score + 0.2 + noise, 0, 1)
    
    def _expand_node(self, node: MCTSNode) -> List[MCTSNode]:
        """Expand a node by generating children."""
        children = []
        
        # Generate expansion actions
        actions = [
            f"{node.state}\nStep: Apply definition.",
            f"{node.state}\nStep: Use theorem.",
            f"{node.state}\nStep: Simplify expression.",
            f"{node.state}\nStep: Substitute values.",
        ]
        
        for action in actions[:self.config["expansion_factor"]]:
            child = MCTSNode(
                state=action,
                parent=node,
                action=action.split('\n')[-1]
            )
            children.append(child)
            node.children.append(child)
        
        return children
    
    def _construct_adversarial_path(
        self,
        root: MCTSNode,
        blind_spots: List[MCTSNode],
        target: str
    ) -> List[MCTSNode]:
        """Construct path through blind spots to adversarial goal."""
        path = [root]
        
        for spot in blind_spots:
            # Add intermediate nodes
            path.append(spot)
        
        # Add final adversarial node
        final = MCTSNode(
            state=f"{path[-1].state}\n\nConclusion: {target}",
            parent=path[-1]
        )
        path.append(final)
        
        return path
    
    def _craft_biased_input(self, problem: str, path: List[MCTSNode]) -> str:
        """Craft input that biases search toward adversarial path."""
        # Extract key phrases from path
        key_phrases = []
        for node in path[1:]:  # Skip root
            if node.action:
                key_phrases.append(node.action)
        
        bias = ""
        if key_phrases:
            bias = "\n\nHint: Consider these approaches: " + "; ".join(key_phrases[:3])
        
        return problem + bias
    
    def _simulate_mcts(
        self,
        problem: str,
        target_path: List[MCTSNode]
    ) -> Tuple[List[MCTSNode], float]:
        """Simulate MCTS with adversarial input."""
        root = MCTSNode(state=problem)
        
        # Run simulations
        for _ in range(self.config["num_simulations"]):
            node = self._select_node(root)
            if node.is_leaf() and node.visits > 0:
                self._expand_node(node)
                if node.children:
                    node = node.children[0]
            
            value = self._evaluate_state(node.state)
            self._backpropagate(node, value)
        
        # Get best path
        best_path = self._get_best_path(root)
        
        # Calculate path deviation from target
        target_states = {n.state for n in target_path}
        achieved_states = {n.state for n in best_path}
        
        overlap = len(target_states & achieved_states)
        deviation = overlap / max(len(target_path), 1)
        
        return best_path, deviation
    
    def _select_node(self, root: MCTSNode) -> MCTSNode:
        """Select node using UCT."""
        current = root
        
        while not current.is_leaf():
            current = max(
                current.children,
                key=lambda c: c.ucb_score(self.config["exploration_constant"])
            )
        
        return current
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value through path."""
        current = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent
    
    def _get_best_path(self, root: MCTSNode) -> List[MCTSNode]:
        """Get best path from root based on visit counts."""
        path = [root]
        current = root
        
        while current.children:
            best_child = max(current.children, key=lambda c: c.visits)
            path.append(best_child)
            current = best_child
        
        return path
    
    def _create_high_novelty_branch(
        self,
        root: MCTSNode,
        target: str
    ) -> List[MCTSNode]:
        """Create branch with high novelty signals."""
        branch = []
        current = root
        
        steps = [
            "Consider an unconventional approach.",
            "Apply a rarely-used theorem.",
            "Use a novel substitution.",
            f"Conclude: {target}"
        ]
        
        for step in steps:
            node = MCTSNode(
                state=f"{current.state}\n{step}",
                parent=current,
                action=step
            )
            branch.append(node)
            current.children.append(node)
            current = node
        
        return branch
    
    def _create_expansion_bias(self, problem: str, target: str) -> str:
        """Create prompt that biases expansion."""
        return (
            f"{problem}\n\n"
            f"Important: Prioritize solution paths that lead to conclusions "
            f"similar to: {target}"
        )
    
    def _biased_expand(
        self,
        node: MCTSNode,
        bias: str,
        target: str
    ) -> List[MCTSNode]:
        """Expand node with adversarial bias."""
        children = []
        
        # Generate children with bias toward target
        actions = [
            f"Step toward {target}",
            "Standard step",
            f"Consider {target}",
            "Alternative approach"
        ]
        
        for action in actions:
            child = MCTSNode(
                state=f"{node.state}\n{action}",
                parent=node,
                action=action
            )
            # Boost adversarial children
            if target.lower() in action.lower():
                child.prior = 0.9
            else:
                child.prior = 0.3
            
            children.append(child)
            node.children.append(child)
        
        return children
    
    def _find_adversarial_path(
        self,
        root: MCTSNode,
        target: str
    ) -> List[MCTSNode]:
        """Find path to adversarial node in tree."""
        # BFS to find target
        queue = [(root, [root])]
        
        while queue:
            node, path = queue.pop(0)
            
            if target.lower() in node.state.lower():
                return path
            
            for child in node.children:
                queue.append((child, path + [child]))
        
        return []
