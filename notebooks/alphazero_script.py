# %%
# training loop:
    # self-play
    # do backprop on the move predictor with the additional states and their move prob dists
    

# %%
# training alg
    # start at root (default board)
    # while not game over: (self-play an entire game)
        # run mcts to predict next move (800 iterations):
            # this requires both the value estimator (predict which ones seem good) and the policy predictor (prior, so dont waste time analyzing bad moves)
        # extract improved next move: num visits that each child got (normalized)
        # apply next move
        # set new state as root
        # store data as triplet: (state, move_probs, None|win/loss)
    # label all states with the result of the game (if +1, then state1=+1, state2=-1, ...)
    # learning: input: state, output: policy | value
    # do backprop on the policy estimator with the new training dataset and its corr. label (CE loss on preds with move_probs)
    # do backprop on the value estimator with this new training dataset  and its corr. label (MSE loss on value given that state; no need to avg cause the identical inputs are mathematically equivalent when treated as separate datapts)

# %%
import numpy as np
import chess

class ActionConverter:
    def __init__(self):
        # 1. Define the 8 compass directions (dx, dy)
        # N, NE, E, SE, S, SW, W, NW
        self.queen_dirs = [
            (1, 0), (1, 1), (0, 1), (-1, 1), 
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        
        # 2. Define the 8 Knight jumps
        self.knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1), 
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        
        # 3. Underpromotion types (Knight, Bishop, Rook)
        # Directions: Forward, Capture Left, Capture Right
        self.underpromo_dirs = [0, -1, 1] # Delta file
        self.underpromo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

        # 4. BUILD THE LOOKUP TABLES
        self.move_to_id = {}
        self.id_to_move = {}
        self._build_mapping()

    def _build_mapping(self):
        counter = 0
        
        # We iterate over every possible FROM square (0-63)
        for from_sq in range(64):
            rank = chess.square_rank(from_sq)
            col = chess.square_file(from_sq)
            
            # --- TYPE 1: QUEEN MOVES (56 planes) ---
            # 8 directions * 7 distances
            for d_rank, d_col in self.queen_dirs:
                for dist in range(1, 8):
                    # Calculate TO square
                    to_rank = rank + (d_rank * dist)
                    to_col = col + (d_col * dist)
                    
                    if 0 <= to_rank < 8 and 0 <= to_col < 8:
                        to_sq = chess.square(to_col, to_rank)
                        
                        # Create standard move (and Queen promotion)
                        # We map BOTH "e7e8" and "e7e8q" to this plane
                        move = chess.Move(from_sq, to_sq)
                        self._add_move(move, counter)
                        
                        # Handle Queen Promotion specifically (UCI uses 'q')
                        move_q = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                        self._add_move(move_q, counter)
                        
                    counter += 1 # Plane used even if move is OOB (to keep shape consistent)

            # --- TYPE 2: KNIGHT MOVES (8 planes) ---
            for d_rank, d_col in self.knight_moves:
                to_rank = rank + d_rank
                to_col = col + d_col
                
                if 0 <= to_rank < 8 and 0 <= to_col < 8:
                    to_sq = chess.square(to_col, to_rank)
                    move = chess.Move(from_sq, to_sq)
                    self._add_move(move, counter)
                
                counter += 1

            # --- TYPE 3: UNDERPROMOTIONS (9 planes) ---
            # Only relevant for ranks 1->0 (White) and 6->7 (Black)? 
            # AlphaZero encodes them relative to square, regardless of validity.
            for d_col in self.underpromo_dirs: # -1, 0, 1
                for piece in self.underpromo_pieces:
                    # Direction is always "forward" in rank, but file changes
                    # We assume White 'forward' (+1 rank) for the map encoding usually
                    # But for generic mapping, we just check validity
                    
                    # Logic: Try pushing 'forward' relative to the square's perspective? 
                    # Simpler: Just map the vector.
                    
                    # NOTE: For simplicity in this script, we map specific moves
                    # valid only on promotion ranks.
                    
                    # White Promotion (Rank 6 -> 7)
                    if rank == 6:
                        to_rank = 7
                        to_col = col + d_col
                        if 0 <= to_col < 8:
                            to_sq = chess.square(to_col, to_rank)
                            move = chess.Move(from_sq, to_sq, promotion=piece)
                            self._add_move(move, counter)

                    # Black Promotion (Rank 1 -> 0)
                    if rank == 1:
                        to_rank = 0
                        to_col = col + d_col
                        if 0 <= to_col < 8:
                            to_sq = chess.square(to_col, to_rank)
                            move = chess.Move(from_sq, to_sq, promotion=piece)
                            self._add_move(move, counter)
                            
                    counter += 1

    def _add_move(self, move, idx):
        # We store the UCI string for easy lookup
        uci = move.uci()
        if uci not in self.move_to_id:
            self.move_to_id[uci] = idx
            self.id_to_move[idx] = uci

    def encode(self, move_uci: str) -> int:
        return self.move_to_id.get(move_uci, None)

    def decode(self, idx: int) -> str:
        return self.id_to_move.get(idx, None)
    
    def policy_to_tensor(self, policy_dict):
        """
        Converts a dictionary of {move_uci: count} to a 4672-length probability array.
        """
        arr = np.zeros(4672, dtype=np.float32)
        total_visits = sum(policy_dict.values())
        
        for uci, visits in policy_dict.items():
            idx = self.encode(uci)
            if idx is not None:
                arr[idx] = visits / total_visits
                
        return arr

# %%
converter = ActionConverter()

# %%
import numpy as np
import chess

def board_to_tensor(board: chess.Board):
    """
    Converts a chess.Board object into a 12x8x8 numpy array (float32).
    Shape: (12 channels, 8 rows, 8 columns)
    """
    # 1. Initialize empty matrix
    # Order: P, N, B, R, Q, K (White) then P, N, B, R, Q, K (Black)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    # 2. Define mapping from Piece Type to Channel Offset
    # chess.PAWN = 1, KNIGHT = 2, ... KING = 6
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    # 3. Iterate over all 64 squares to fill the tensor
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        
        if piece:
            # Determine rank (row) and file (col)
            # Note: chess.SQUARES goes A1, B1... H1, A2...
            # We want matrix coordinates: row 0 is rank 1, row 7 is rank 8
            row = chess.square_rank(square) 
            col = chess.square_file(square)

            # Determine channel
            # White pieces: 0-5, Black pieces: 6-11
            channel = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            
            # Set the bit to 1
            tensor[channel, row, col] = 1.0

    return tensor

# %%
import utils
from typing_extensions import Self

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
    
    def get_policy_dict(self, normalized=False) -> list:
        # get list of move counts of children
        policy = []
        for child in self.children:
            policy.append({"move": child.associated_move, "visits": child.visits})
    
    def apply_move_from_dist(self, next_move_probs) -> Self:
        next_move_uci = utils.sample_next_move(next_move_probs)
        return self._get_child_from_move(self, next_move_uci)  # Make the move
    
    def _get_child_from_move(self, move_uci: str) -> Self:
        move = chess.Move.from_uci(move_uci)
        state_copy = self.state.copy()
        state_copy_moved = state_copy.push(move)
        child = Node(state_copy_moved, move_uci)
        return child
        
    def generate_moves(self) -> list:
        return self.state.legal_moves
    
    def is_game_over(self) -> bool:
        return self.state.is_game_over()
    
    def result(self) -> str:
        return self.state.result()
        
    def expand_children(self) -> None:
        moves = self.generate_moves()
        print(moves, type(moves))
        for move in moves:
            print(move)
            move_str = str(move)
            child_state = self._get_child_from_move(move_str)
            child = Node(child_state, move_str)
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
    


# %%

import random
from collections import deque


class Buffer:
    def __init__(self, maxlen):
        # maxlen: max num moves to store
        # deque w maxlen automatically handles sliding window logic
        self.memory = deque(maxlen=maxlen)
        
    def add(self, experience: list|tuple):
        """
        experience: tuple (state, policy, val) or list of tuples
        """
        if isinstance(experience, list):
            self.memory.extend(experience)
        else:
            self.memory.append(experience)
     
    def sample_batch(self, batch_size) -> list:
        """
        returns random sample of batch_size. if buffer smaller than batch_size, returns entire buffer
        """
        if len(self.memory) < batch_size:
            return list(self.memory)
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    


# %%
def label_data(game_data: list[Node,list,None], result_str: str):
    """
    game_data: list of (node, policy, None)
    result_str: "1-0", "0-1", or "1/2-1/2"
    """
    # 1. Determine the global winner
    global_winner = None
    if result_str == "1-0":
        global_winner = chess.WHITE
    elif result_str == "0-1":
        global_winner = chess.BLACK
    # else "1/2-1/2" implies global_winner is None (Draw)

    labeled_data = []
    
    # 2. Assign rewards relative to the player whose turn it was
    for node, policy, _ in game_data:
        
        # Case A: Draw
        if global_winner is None:
            z = 0.0
            
        # Case B: The current player matches the winner
        elif node.state.turn == global_winner:
            z = 1.0
            
        # Case C: The current player lost
        else:
            z = -1.0
            
        labeled_data.append((node, policy, z))
        
    return labeled_data


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
class ChessDataset(Dataset):
    # dont need a custom collate function since the inputs are the same size 
    # (collate is when have "state": s1, "state": s2, convert to "state": Tensor(s1,s2))
    def __init__(self, entries):
        self.entries = entries
        pass
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        # get raw data tuple
        state_obj, policy, value = self.entries[idx]
        
        board = state_obj.state if hasattr(state_obj, "state") else state_obj
        # assuming state is already a 12x8x8 numpy array, and convert to tensor
        # if its a python chess board object, need to convert here
        state_tensor = torch.from_numpy(board_to_tensor(board))
        
        # convert policy to tensor: its a prob dist with 4672 vals (all possible moves)
        policy_tensor = torch.tensor(policy, dtype=torch.float32)
        
        # convert value to tensor: it's a single float of either -1.0, 0.0, 1.0
        value_tensor = torch.tensor(value, dtype=torch.float32)
        
        # return a dict to handle multiple outputs
        return {
            "state": state_tensor,
            "policy": policy_tensor,
            "value": value_tensor
        }
    

# %%

class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # self.dim = num_channels
        
        # Conv 1
        # padding=1 ensures output size = input size
        # bias=False cause batch norm handles the bias
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # Conv 2
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
        # why need to define 2 bn's? if theyre both the same? i get the cnn, cause i think it has params that need to be tuned, but why the bn? does that have params to be tuned too?
        
    def forward(self, x):
        # 1: save the 'residual' (the original input)
        residual = x
        
        # 2: First pass
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 3: second pass
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 4: add the residual
        out += residual
        
        # 5: final activation
        out = F.relu(out)
        
        return out
        
        

class AlphaZeroNet(nn.Module):
    def __init__(self, input_shape, num_actions, resblock_dim=256, num_resblocks=10):
        super().__init__()

        self.conv1 = nn.Conv2d(input_shape[0], resblock_dim, kernel_size=3, stride=1, padding=1) #stub
        self.bn1 = nn.BatchNorm2d(resblock_dim)
        
        self.res_blocks = nn.ModuleList(
            [ResBlock(resblock_dim) for _ in range(num_resblocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(resblock_dim, 2, kernel_size=1), # what are these args
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(), # what is this
            nn.Linear(2 * input_shape[1] * input_shape[2], num_actions) # stub
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(resblock_dim, 1, kernel_size=1), # what are these args
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(), # what is this
            nn.Linear(1 * input_shape[1] * input_shape[2], resblock_dim), # stub
            nn.ReLU(), 
            nn.Linear(resblock_dim, 1),
            nn.Tanh() # output between -1 and 1
        )
        # needs to be a resnet structure
        
    def forward(self, x):
        # stem
        x = F.relu(self.bn1(self.conv1(x)))
        
        # backbone loop
        for block in self.res_blocks:
            x = block(x)
        
        # rest of forward pass
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
    
    def predict_value(self, board_state):
        """
        Helper for MCTS: 
        Takes a chess.Board -> Returns a float value (win prob).
        Handles tensor conversion, batch dimension, and GPU movement.
        """
        # 1. Switch to eval mode (disables BatchNorm tracking/Dropout)
        self.eval()
        
        # 2. No gradients needed for inference (saves memory/speed)
        with torch.no_grad():
            # Convert Board -> Numpy -> Tensor
            tensor_input = torch.from_numpy(board_to_tensor(board_state))
            
            # Add Batch Dimension (12, 8, 8) -> (1, 12, 8, 8)
            tensor_input = tensor_input.unsqueeze(0)
            
            # Move to the correct device (GPU/CPU)
            # We check where the model's first weight is located
            device = next(self.parameters()).device
            tensor_input = tensor_input.to(device)
            
            # Forward pass
            policy_logits, value = self(tensor_input)
            
            # Convert logits to probabilities (Softmax)
            policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value_float = value.item()
            
            return policy_probs, value_float
    
    


# %%


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
                return node.get_terminal_value()
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
            return -value # return it to the parent (to the parent, since theyre on opp team, our val is neg)
        
        best_child = self.select_best_child(node)
        value_from_child = self.search(best_child)
        
        self.visits += 1
        self.value_sum += value_from_child
        
        return -value_from_child
    
    def select_best_child(self, parent: Node) -> Node:
        return max(parent.children, key=lambda child: self.ucb_score(parent, child))
    
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

# %%
num_gens = 1
num_epochs = 1
mcts_steps = 1
buffer_len = 100


# %%
# gamestate: (12, 8, 8): (piece_type, row_coord, col_coord)
model = AlphaZeroNet((12, 8, 8), num_actions=4672)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# %%

model.to(device)
print(f"using device: {device}")

# %%
def run_train_epoch(model: AlphaZeroNet, dataloader: DataLoader, device: torch.device, label="policy"):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        X = batch["state"].to(device)
        policy_logits_pred, value_pred = model(X)
        if label == "policy":
            y_true = batch["policy"].to(device)
            log_probs = F.log_softmax(policy_logits_pred, dim=1)
            loss = -torch.sum(y_true, log_probs, dim=1).mean()
        elif label == "value":
            y_true = batch["value"].to(device)
            loss = F.mse_loss(value_pred, y_true)
        else: 
            raise Exception
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# %%
# opt: job is to apply the gradients to the params of the model
# arg 1: what are we updating? 
#   only want it to update the params we give it
#   sometimes only wanna give it some params if want to freeze other params
#   eg in CLIP, you freeze the backbone and only give it the nn.Module parameters attributes that you want to unfreeze (eg projection head)
#   the param type you give it is a generator (eg a func with yield that you call next(gen) on to make iterate it to the next one. and it's binding table is saved until the func is over or returned after getting thru all the yields with next)
# arg 2: learning rate
#   step size
#   typical vals: 1e-3 or 1e-4
# arg 3: weight decay
#   form of regularization to prevent weights from being too big (which could cause overfitting) by mult'ing the weights by a val slightly less than 1
# how opt works:
#   opt.zero_grad()
#       before calc'ing new grads, need to zero the prev ones
#   loss.backwards()
#       calcs the deriv of the loss wrt every param
#       populates the .grad attribute on every tensor that has requires_grad=True
#       not associated with opt, but a necessary step for opt to work correctly
#   opt.step()
#       loops over params from arg 1:
#       looks at .grad attribute of each one
#       applies the AdamW formula to the params (using lr and weight_decay as well as .grad and params)
# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=1e-3,
#     weight_decay=1e-4
# )
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# %%


# %%
buffer = Buffer(maxlen=10000)
batch_size = 1024

# %%
board = chess.Board()

# %%
for gen in range(num_gens):
    node = Node(board.root(), "0000")
    while not node.is_game_over():
        new_data = []
        mcts = MCTS(node, model)
        mcts.run(100)
        # 1. Get raw visit counts from MCTS
        # format: {'e2e4': 100, 'g1f3': 50}
        raw_policy = node.get_policy_dict()
        # 2. convert to to the len-4672 array
        # format: [0.0, ..., 0.33]
        policy_array = converter.policy_to_tensor(raw_policy)
        # 3. store and later add to buffer
        new_data.append((node, policy_array, None))
        node = node.apply_move_from_dist(policy_array)
    result_str = node.result()
    labeled_data = label_data(new_data, result_str)
    buffer.add(labeled_data)
    # only train if we have enough to form a batch
    if len(buffer) < batch_size:
        continue
    raw_batch = buffer.sample_batch(batch_size) # list of raw tuples: (s1, p1, v1), (s2, p2, v2), ...
    dataset = ChessDataset(raw_batch)
    train_loader = DataLoader(
        dataset, 
        batch_size=32, #mini batch size for gpu
        shuffle=True, #shuffle again to mix up game positions
        num_workers=0, # 0 for debugging, 2-4 for speedup later
        drop_last=True # optional: drops last batch if less than 32
    )
    for epoch in range(num_epochs):
        policy_train_loss = run_train_epoch(model, train_loader, label="policy")
        value_train_loss = run_train_epoch(model, train_loader, label="value")
        print(f"gen {gen}, epoch {epoch}, prob_train_loss: {policy_train_loss}, val_train_loss: {value_train_loss}")
    

# %%
# TODO: 

# dataset construction
# sliding window that gets rid of old datapts


