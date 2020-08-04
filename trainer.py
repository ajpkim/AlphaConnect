import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_loss import AlphaLoss
from alpha_net import AlphaNet
from game.connect4 import Connect4
from mcts import mcts_self_play
from replay_buffer import ReplayBuffer


class Trainer:
    def __init__(self, config):
        self.config = config
        self.game = globals()[config.game]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AlphaNet().to(self.device)
        self.replay_buffer = ReplayBuffer(capacity=config.memory_capacity, seed=config.random_seed)
        self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size, config.lr_gamma, -1)
        self.loss_fn = AlphaLoss()
        self.training_step_count = 0


    ### MAIN SCRIPT SO ITS EASIER TO WRITE DATA OUT LATER AND STORE GAME DATA ETC.?
    def self_play(self):
        """Execute one game of self play and store training data"""
        states, Pis, Zs = mcts_self_play(self.net, self.game(), self.config.n_simulations, self.config.C_puct)
        
        for state, Pi, Z in zip(states, Pis, Zs):
            self.replay_buffer.push(state, Pi, Z)
        
        return states, Pis, Zs

    def learn(self):
        """Perform one learning step with batch sampled from replay buffer"""
        batch = self.replay_buffer.sample(self.config.batch_size)
        states = torch.stack([x.state for x in batch]).to(self.device)  # game states are already torch tensors
        Pi = torch.tensor([x.Pi for x in batch], dtype=torch.float32).to(self.device)
        Z = torch.tensor([x.Z for x in batch], dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        V, P = self.net(states)
        loss = self.loss_fn(Z, Pi, V, P)
        print(loss)
        loss.backward()
        self.optimizer.step()

        self.training_step_count += 1
        if self.training_step_count < 10_000: 
            self.scheduler.step()

    def save_checkpoint(self, checkpoint_file):        
        torch.save({
                    'model_state_dict': self.net.state_dict(),
                    'replay_buffer_memory': self.replay_buffer.memory,
                    'training_step_count': self.training_step_count
                    }, checkpoint_file)
    
    def load_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        for state, Pi, Z in checkpoint['replay_buffer_memory']:
            self.replay_buffer.push(state, Pi, Z)
        self.training_step_count = checkpoint['training_step_count']

    def __repr__(self):
        return '<Trainer obj>'



