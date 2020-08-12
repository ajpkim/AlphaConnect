from collections import namedtuple
import random

GameData = namedtuple('GameData', field_names=('state', 'Pi', 'Z'))

class ReplayBuffer:
    def __init__(self, capacity=10000, seed=3):
        self.capacity = capacity
        self.memory = [] 
        self.position = 0
        self.seed = random.seed(seed)
    
    def push(self, state, Pi, Z):
        if len(self.memory) <= self.position:
            self.memory.append(None)  # avoid index errors below
        self.memory[self.position] = GameData(state, Pi, Z)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __repr__(self):
        return f'ReplayBuffer storing {len(self.memory)} training data points'
