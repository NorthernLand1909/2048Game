from ai.trainer import Trainer
import torch

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    torch.save(trainer.model.state_dict(), "2048_dqn.pth")