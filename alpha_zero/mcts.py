from __future__ import annotations

from typing import Any, Callable, Iterable, Optional
from typing_extensions import Self
import numpy as np
# import logging

from .utils import sample_next_move
from .model import AlphaZeroNet
from .game_adapter import GameAdapter
# from .logger_config import setup_logger


class Node:
    def __init__(self, state: Any, associated_move: str) -> None:
        """Create a node for a board state and its incoming move."""
        self.state = state
        self.children: list[Node] = []
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.associated_move: str = associated_move
        self.policy: Optional[np.ndarray] = None
        self.prior: float = 0.0 
    
    def get_policy_dict(self, normalized: bool = False) -> dict[str, float]:
        """Return a move-to-visits dict, optionally normalized to probabilities."""
        # Map move UCI strings to visit counts for the child nodes.
        policy = {child.associated_move: child.visits for child in self.children}
        if normalized and policy:
            total_visits = sum(policy.values())
            if total_visits > 0:
                policy = {move: visits / total_visits for move, visits in policy.items()}
        return policy
    
    def apply_move_from_dist(
        self,
        next_move_probs: np.ndarray,
        legal_moves: Iterable[Any],
        temperature: float = 1.0,
        encode_move: Optional[Callable[[str], Optional[int]]] = None,
        decode_move: Optional[Callable[[int], Optional[str]]] = None,
        child_factory: Optional[Callable[[Node, Any], Node]] = None,
    ) -> Self:
        """Sample a move from a policy vector and return the resulting child node."""
        if legal_moves is None:
            raise ValueError("legal_moves must be provided for sampling.")
        next_move_uci = sample_next_move(
            next_move_probs,
            legal_moves,
            temperature=temperature,
            encode_move=encode_move,
            decode_move=decode_move,
        )
        if child_factory is not None:
            return child_factory(self, next_move_uci)
        return self._get_child_from_move(next_move_uci)  # Make the move
    
    def _get_child_from_move(self, move_uci: str) -> Self:
        """Create a child node by applying a move to a copy of this state."""
        state_copy = self.state.copy()
        if hasattr(state_copy, 'push_uci'):
            state_copy.push_uci(move_uci)
        else:
            state_copy.push(move_uci)
        child = Node(state_copy, move_uci)
        return child
        
    # def generate_moves(self) -> list:
    #     """Return the legal moves from this position."""
    #     # return self.state.legal_moves
    #     self.adapter.legal_moves(self.state)
    
    # def is_game_over(self) -> bool:
    #     """Return True if the current state is terminal."""
    #     return self.state.is_game_over() #TODO: either doesnt have this logic or needs adapter
    
    def result(self) -> str:
        """Return the game result string for the current state."""
        return self.state.result()
        
    def expand_children(
        self,
        moves: Iterable[Any],
        child_factory: Optional[Callable[[Node, Any], Node]] = None,
    ) -> None:
        """Generate child nodes for all legal moves."""
        if moves is None:
            raise ValueError("moves must be provided when expanding children.")
        for move in moves:
            if child_factory is not None:
                child = child_factory(self, move)
            else:
                move_str = move.uci() if hasattr(move, 'uci') else str(move)
                child = self._get_child_from_move(move_str)
            self.children.append(child)
    
    def is_leaf(self) -> bool:
        """Return True if the node has no children."""
        return not self.children
    
    # def get_terminal_value(self) -> int:
    #     """Return the terminal value from the current player's perspective."""
    #     # TODO: have gpt implement
    #     result_str = self.state.result()
    #     # 1. Determine the global winner
    #     global_winner = None
    #     if result_str == "1-0":
    #         global_winner = chess.WHITE
    #     elif result_str == "0-1":
    #         global_winner = chess.BLACK
    #     # else "1/2-1/2" implies global_winner is None (Draw)
    #     # Case A: Draw
    #     if global_winner is None:
    #         z = 0.0
            
    #     # Case B: The current player matches the winner
    #     elif self.state.turn == global_winner:
    #         z = 1.0
            
    #     # Case C: The current player lost
    #     else:
    #         z = -1.0
    #     return z
    
    
    # def is_terminal(self):
    #     """Return True if the game is over for this node."""
    #     return self.state.is_game_over()
    


class MCTS:
    def __init__(self, root: Node, model: AlphaZeroNet, adapter: GameAdapter) -> None:
        """Initialize the MCTS runner with a root node and policy/value model."""
        self.root = root
        self.model = model
        self.adapter = adapter
        self.c_puct = 1.0
    
    def run(self, steps: int) -> None:
        """Run a fixed number of MCTS simulations."""
        for _ in range(steps):
            self._search(self.root)
            

    def _search(self, node: Node) -> float:
        """Perform one recursive MCTS search and return the backed-up value."""
        # stop condition: node is a leaf
        if not node.is_leaf():
            best_child = self.select_best_child(node)
            value_from_child = self._search(best_child)
            
            node.visits += 1
            node.value_sum += value_from_child
            
            return -value_from_child
        
        # node is leaf
        if self.is_terminal(node):
            value = self.get_terminal_value(node)
            node.visits += 1
            node.value_sum += value # do i really need to increment this?
            return -value
        # not terminal state, simply leaf
        self.expand_children(node)
        policy, value = self.model.predict_value(
            node.state,
            board_to_tensor_fn=self.adapter.board_to_tensor,
        )
        
        # assign priors
        for child in node.children:
            move_idx = self.adapter.encode_move(child.associated_move)
            # move_idx = utils.converter.encode(child.associated_move) #TODO: change to adapter
            if move_idx is not None:
                child.prior = policy[move_idx]
            else:
                child.prior = 0.0
        node.policy = policy
        node.visits += 1
        node.value_sum += value
        return -value # return it to the parent (to the parent, since theyre on opp team, our val is neg)
        
        
    
    def select_best_child(self, parent: Node) -> Node: 
        """Select the child with the highest UCB score."""
        best_child = max(parent.children, key=lambda child: self._ucb_score(parent, child))
        # print(type(best_child))
        return best_child
    
    def _ucb_score(self, parent: Node, child: Node) -> float:
        """Compute the PUCT UCB score for a child node."""
        # 1. Calculate Q (Exploitation)
        # "Value" from the perspective of the parent. 
        # Since child stores value for the opponent, we flip it.
        if child.visits == 0:
            q_value = 0
        else:
            q_value = -child.value_sum / child.visits
            
        # 2. Calculate U (Exploration)
        # PUCT Formula: c * P(s,a) * sqrt(N_parent) / (1 + N_child)
        u_score = self.c_puct * child.prior * (np.sqrt(parent.visits) / (1 + child.visits))
        
        return q_value + u_score

    def get_terminal_value(self, node: Node) -> float:
        """Return the terminal value from the current player's perspective."""
        return self.adapter.terminal_value(node.state)

    def _create_child(self, parent: Node, move: Any) -> Node:
        """Create a child node by applying a move to a copied board."""
        state_copy = self.adapter.copy_board(parent.state)
        self.adapter.push(state_copy, move)
        move_uci = move.uci() if hasattr(move, 'uci') else str(move)
        return Node(state_copy, move_uci)

    def generate_moves(self, node: Node) -> list[Any]:
        """Return the legal moves from this position."""
        return list(self.adapter.legal_moves(node.state))

    def expand_children(self, node: Node) -> None:
        """Expand a node using adapter-generated legal moves."""
        moves = self.generate_moves(node)
        node.expand_children(moves, child_factory=self._create_child)
    
    # def is_game_over(self, state) -> bool:
    #     """Return True if the current state is terminal."""
    #     return self.state.is_game_over() #TODO: either doesnt have this logic or needs adapter
    
    def is_terminal(self, node: Node) -> bool:
        """Return True if the game is over for this node."""
        return self.adapter.is_terminal(node.state)

    def print_child_visits(self) -> None:
        """Print visit counts for each child of the root."""
        for child in self.root.children:
            print(f"{child.associated_move}: {child.visits}. ")
