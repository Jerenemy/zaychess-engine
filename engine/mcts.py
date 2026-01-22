import chess
import utils
from model import AlphaZeroNet
from typing_extensions import Self
import numpy as np

from utils import converter

class Node:
    def __init__(self, state: chess.Board, associated_move: str):
        self.state = state
        self.children: list = []
        self.visits = 0
        self.value_sum = 0
        self.associated_move = associated_move
        self.policy = None
        self.prior = 0.0 
        # self.player_color = player_color
    
    def get_policy_dict(self, normalized: bool = False) -> dict:
        # Map move UCI strings to visit counts for the child nodes.
        policy = {child.associated_move: child.visits for child in self.children}
        if normalized and policy:
            total_visits = sum(policy.values())
            if total_visits > 0:
                policy = {move: visits / total_visits for move, visits in policy.items()}
        return policy
    
    def apply_move_from_dist(self, next_move_probs) -> Self:
        next_move_uci = utils.sample_next_move(next_move_probs, self.generate_moves())
        return self._get_child_from_move(next_move_uci)  # Make the move
    
    def _get_child_from_move(self, move_uci: str) -> Self:
        move = chess.Move.from_uci(move_uci)
        state_copy = self.state.copy()
        state_copy.push(move)
        child = Node(state_copy, move_uci)
        return child
        
    def generate_moves(self) -> list:
        return self.state.legal_moves
    
    def is_game_over(self) -> bool:
        return self.state.is_game_over()
    
    def result(self) -> str:
        return self.state.result()
        
    def expand_children(self) -> None:
        moves = self.generate_moves()
        # print(moves, type(moves))
        for move in moves:
            # print(move)
            move_str = str(move)
            child = self._get_child_from_move(move_str)
            self.children.append(child)
    
    def is_leaf(self) -> bool:
        return not self.children
    
    def get_terminal_value(self) -> int:
        # TODO: have gpt implement
        result_str = self.state.result()
        # 1. Determine the global winner
        global_winner = None
        if result_str == "1-0":
            global_winner = chess.WHITE
        elif result_str == "0-1":
            global_winner = chess.BLACK
        # else "1/2-1/2" implies global_winner is None (Draw)
        # Case A: Draw
        if global_winner is None:
            z = 0.0
            
        # Case B: The current player matches the winner
        elif self.state.turn == global_winner:
            z = 1.0
            
        # Case C: The current player lost
        else:
            z = -1.0
        return z
    
    
    def is_terminal(self):
        return self.state.is_game_over()
    


class MCTS:
    def __init__(self, root: Node, model: AlphaZeroNet):
        self.root = root
        self.model = model
        self.c_puct = 1.0
    
    def run(self, steps):
        for _ in range(steps):
            self.search(self.root)
            

    def search(self, node: Node) -> float:
        # stop condition: node is a leaf
        if node.is_leaf():
            # here do i want to expand the children? otherwise it's gonna hit this every time. do i expand all of the possible vals?
            if node.is_terminal():
                value = node.get_terminal_value()
                node.visits += 1
                node.value_sum += value
                return -value
            # not terminal state, simply leaf
            node.expand_children()
            policy, value = self.model.predict_value(node.state)
            
            # assign priors
            for child in node.children:
                move_idx = converter.encode(child.associated_move)
                if move_idx is not None:
                    child.prior = policy[move_idx]
                else:
                    child.prior = 0.0
            node.policy = policy
            node.visits += 1
            node.value_sum += value
            return -value # return it to the parent (to the parent, since theyre on opp team, our val is neg)
        
        best_child = self.select_best_child(node)
        value_from_child = self.search(best_child)
        
        node.visits += 1
        node.value_sum += value_from_child
        
        return -value_from_child
    
    def select_best_child(self, parent: Node) -> Node: 
        best_child = max(parent.children, key=lambda child: self.ucb_score(parent, child))
        # print(type(best_child))
        return best_child
    
    def ucb_score(self, parent: Node, child: Node):
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

    def print_child_visits(self):
        for child in self.root.children:
            print(f"{child.associated_move}: {child.visits}. ")