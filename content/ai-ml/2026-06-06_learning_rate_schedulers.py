"""
Learning Rate Schedulers
Common scheduling strategies for training neural networks.
"""
import numpy as np

class StepDecay:
    def __init__(self, initial_lr=0.01, drop_factor=0.5, drop_every=10):
        self.initial_lr = initial_lr
        self.drop_factor = drop_factor
        self.drop_every = drop_every

    def __call__(self, epoch):
        return self.initial_lr * (self.drop_factor ** (epoch // self.drop_every))

class CosineAnnealing:
    def __init__(self, initial_lr=0.01, T_max=100, eta_min=1e-6):
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min

    def __call__(self, epoch):
        return self.eta_min + (self.initial_lr - self.eta_min) *                (1 + np.cos(np.pi * epoch / self.T_max)) / 2

class WarmupCosine:
    def __init__(self, initial_lr=0.01, warmup_epochs=10, total_epochs=100):
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def __call__(self, epoch):
        if epoch < self.warmup_epochs:
            return self.initial_lr * (epoch + 1) / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        return self.initial_lr * (1 + np.cos(np.pi * progress)) / 2

class ExponentialDecay:
    def __init__(self, initial_lr=0.01, decay_rate=0.95):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def __call__(self, epoch):
        return self.initial_lr * (self.decay_rate ** epoch)


if __name__ == "__main__":
    schedulers = {
        "Step Decay": StepDecay(),
        "Cosine Annealing": CosineAnnealing(),
        "Warmup + Cosine": WarmupCosine(),
        "Exponential Decay": ExponentialDecay(),
    }

    for name, scheduler in schedulers.items():
        lrs = [scheduler(e) for e in range(100)]
        print(f"{name}: start={lrs[0]:.6f}, mid={lrs[50]:.6f}, end={lrs[99]:.6f}")
