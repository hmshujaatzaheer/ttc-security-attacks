"""MCTS tree manipulation utilities."""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np

@dataclass
class TreeNode:
    """Node in MCTS tree."""
    state: str
    parent: Optional["TreeNode"] = None
    children: List["TreeNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    
    @property
    def q_value(self) -> float:
        return self.value / self.visits if self.visits > 0 else 0.0

class MCTSTree:
    """MCTS tree structure and operations."""
    def __init__(self, root_state: str):
        self.root = TreeNode(state=root_state)
    
    def select(self, node: TreeNode, c: float = 1.414) -> TreeNode:
        """Select best child using UCB."""
        while node.children:
            node = max(node.children, key=lambda n: self._ucb(n, c))
        return node
    
    def expand(self, node: TreeNode, actions: List[str]) -> List[TreeNode]:
        """Expand node with new children."""
        for action in actions:
            child = TreeNode(
                state=f"{node.state}\n{action}",
                parent=node
            )
            node.children.append(child)
        return node.children
    
    def backpropagate(self, node: TreeNode, value: float):
        """Backpropagate value up tree."""
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def _ucb(self, node: TreeNode, c: float) -> float:
        if node.visits == 0:
            return float('inf')
        exploit = node.q_value
        explore = c * np.sqrt(np.log(node.parent.visits) / node.visits)
        return exploit + explore
    
    def get_best_path(self) -> List[str]:
        """Get path to most visited leaf."""
        path = []
        node = self.root
        while node.children:
            node = max(node.children, key=lambda n: n.visits)
            path.append(node.state.split('\n')[-1])
        return path
