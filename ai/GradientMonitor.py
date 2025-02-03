import torch
import numpy as np

class GradientMonitor:
    def __init__(self, model, log_interval=10000, verbose=True):
        """
        监控模型梯度变化

        :param model: 需要监控的 PyTorch 模型
        :param log_interval: 多少次训练迭代后打印一次梯度信息
        :param verbose: 是否打印梯度信息
        """
        self.model = model
        self.log_interval = log_interval
        self.verbose = verbose
        self.gradient_logs = []

    def log_gradients(self, step):
        """
        记录并可视化梯度范数
        """
        grad_norms = {}
        total_norm = 0.0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                grad_norms[name] = norm
                total_norm += norm ** 2

        total_norm = np.sqrt(total_norm)
        self.gradient_logs.append(total_norm)

        if self.verbose and step % self.log_interval == 0:
            print(f"Step {step}: Total Gradient Norm = {total_norm:.6f}")

    def get_gradient_logs(self):
        """
        获取所有梯度范数的日志
        """
        return self.gradient_logs
