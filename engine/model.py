import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import board_to_tensor

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