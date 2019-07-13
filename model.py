import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, in_size, out_size, **kwargs):
        super(DQN, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.SAVE_MODEL_NAME = kwargs.pop("save_model_name", None)
        self.model = nn.Sequential(
            nn.Linear(in_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, out_size)
        )

    def forward(self, x):
        return self.model(x)

    def __call__(self, x):
        return self.forward(x)
