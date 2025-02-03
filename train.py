from ai.TransformerTrainer import TransformerTrainer
from ai.trainer import Trainer
from ai.TrainingVisualizer import TrainingVisualizer
import torch

if __name__ == "__main__":
    trainer = TransformerTrainer()
    trainer.train()
    torch.save(trainer.agent.model.state_dict(), "2048_transformer.pth")