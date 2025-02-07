from ai.Transformer.TransformerTrainer import TransformerTrainer
from ai.DQN.DQNTrainer import DQNTrainer
import torch

class TrainController:
    def __init__(self, model):
        if model == 'transformer':
            self.trainer = TransformerTrainer()
        elif model == 'DQN':
            self.trainer = DQNTrainer()
        else:
            raise ValueError("Invalid model type. Choose 'transformer' or 'DQN'.")
        
    def train(self):
        self.trainer.train()
        torch.save(self.trainer.agent.model.state_dict(), f"2048_{self.trainer.model_name}.pth")