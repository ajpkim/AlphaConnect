import torch
import torch.nn as nn


class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, Z, Pi, V, P):
        """
        Loss function is a combination of MSE and cross entropy.

        (Weight decay is used for regularization via optimizer)

        Args:
            - Z: game outcome
            - Pi: action probabilities derived from MCTS 
            - V: outcome value predicated by network
            - P: action probabilities predicted by NN
        """
        value_loss = ((Z - V) ** 2).mean()
        policy_loss = torch.sum(-Pi * P.log(), axis=1).mean()
        return value_loss + policy_loss

