import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_loss import AlphaLoss
from alpha_net import AlphaNet
from replay_buffer import ReplayBuffer


class Trainer:
    def __init__(self, config):
        self.net = AlphaNet()
        self.replay_buffer = ReplayBuffer(capacity=config.memory_capacity, seed=config.random_seed)
        self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=config.lr, momentum=config.momentum)
        # self.scheduler = torch.optim.lr_scheduler.
        self.loss_fn = AlphaLoss()
        self.batch_size = config.batch_size
        self.training_step_count = 0

    def save_checkpoint(self, checkpoint_file):
        with open(checkpoint_file, 'wb') as f:
            data = {}
            data['model_state_dict'] = self.net.state_dict()
            data['replay_buffer_memory'] = self.replay_buffer.memory
            data['training_step_count'] = self.training_step_count
            pickle.dump(data, f)
    
    def load_checkpoint(self, checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
            model_state_dict = data['model_state_dict']
            self.net.load_state_dict(model_state_dict)
            self.training_step_count = data['training_step_count']
            for state, Pi, Z in data['replay_buffer_memory']:
                self.replay_buffer.push(state, Pi, Z)
        



    def self_play(self):
        game_data = []
        pass

    def learn(self):
        batch = self.replay_buffer.sample(self.batch_size)

        pass

    def __repr__(self):
        return '<Trainer obj>'



