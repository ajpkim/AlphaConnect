import math
import torch
# loss function for alpha_net

# l2 regularization of c is set to 10**-4 in alphagoZero  
#    L2 weight regularizatiojn is c * theta**2 (sum of squares of all weights)

### L = (z - v)**2 - pi * log(p) + c * theta**2

# optimazation 
# SGD  with momentum optimizer, momentum = 0.9
    # LR = 10**-2 -> 10**-4 after 600k steps

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()
        pass

    def forward(self, z, p, v, pi):
        """
        Args:
            - z: value of state caclucated by MCTS
            - p: actual probabilities calculated by MCTS
            - v: value predicated by network
            - pi: policy predicated by the network 
        """
        value_loss = (z - v) ** 2
        pass
        # policy_loss = 
        
        # pi * math.log(p)  # cross entropy 
        # loss = value_loss - policy_loss + c * theta**2
        # return loss





