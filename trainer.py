import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_loss import AlphaLoss
from alpha_net import AlphaNet
from game.connect4 import Connect4
from mcts import mcts_self_play
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, config):
        self.config = config
        self.game = globals()[config.game]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AlphaNet().to(self.device)
        self.replay_buffer = ReplayBuffer(capacity=config.start_memory_cap, seed=config.random_seed)
        self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size, config.lr_gamma, -1)
        self.loss_fn = AlphaLoss()
        self.training_step_count = 0
        self.self_play_count = 0


    def self_play(self):
        """Execute one game of self play and store training data in replay buffer."""
        game = self.game()
        states, Pis, Zs = mcts_self_play(self.net, game, self.config.n_simulations, self.config.C_puct, self.config.dirichlet_alpha)
        
        for state, Pi, Z in zip(states, Pis, Zs):
            if self.config.horizontal_flip:
                if (flip := random.choice((True, False))):
                    state = state.flip(2)
            self.replay_buffer.push(state, Pi, Z)
        
        self.self_play_count += 1
        if self.self_play_count == self.config.memory_step:
            self.replay_buffer.capacity = self.config.end_memory_cap
        
        return game.history

    def learn(self):
        """Perform one learning step with batch sampled from replay buffer"""
        batch = self.replay_buffer.sample(self.config.batch_size)
        states = torch.stack([x.state for x in batch]).to(self.device)  # game states are already torch tensors
        Pi = torch.tensor([x.Pi for x in batch], dtype=torch.float32).to(self.device)
        Z = torch.tensor([x.Z for x in batch], dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        V, P = self.net(states)
        loss = self.loss_fn(Z, Pi, V, P)
        loss.backward()
        self.optimizer.step()

        self.training_step_count += 1
        if self.training_step_count <= self.config.lr_scheduler_last_step: 
            self.scheduler.step()
        
    
    def save_model(self, model_file):
        torch.save({'model_state_dict': self.net.state_dict()}, model_file)
    
    def load_model(self, model_file):
        data = torch.load(mdoel_file, map_location=self.device)
        state_dict = data['state_dict']
        self.net.load_state_dict(state_dict)

    def save_checkpoint(self, checkpoint_file):        
        
        torch.save({
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'replay_buffer_position': self.replay_buffer.position,
                    'training_step_count': self.training_step_count,
                    'self_play_count': self.self_play_count
                    }, checkpoint_file)
    
    def load_checkpoint(self, checkpoint_file):

        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.replay_buffer.position = checkpoint['replay_buffer_position']
        self.training_step_count = checkpoint['training_step_count']
        self.self_play_count = checkpoint['self_play_count']

    def save_replay_memory(self, memory_file):
        torch.save(self.replay_buffer.memory, memory_file)
        
    def load_replay_memory(self, memory_file):
        memory = torch.load(memory_file, map_location='cpu')  # get sent to gpu during learn step. Otherwise some memories are on cpu, others gpu.
        for state, Pi, Z in memory:
            self.replay_buffer.push(state, Pi, Z)


    def __repr__(self):
        return f'<Trainer. self play count: {self.self_play_count}, training steps: {self.training_step_count}>'



