from ai.trainer import Trainer
from TrainingVisualizer import TrainingVisualizer
import torch

if __name__ == "__main__":
    trainingVisualizer =  TrainingVisualizer()
    trainer = Trainer(trainingVisualizer)
    trainer.train()
    torch.save(trainer.model.state_dict(), "2048_dqn.pth")