import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


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


class TestDQN(nn.Module):
    def __init__(self, input_shape, num_actions, **kwargs):
        super(TestDQN, self).__init__()
        self.SAVE_MODEL_NAME = kwargs.pop("save_model_name", None)

        # self.conv1 = nn.Conv2d(input_shape[0], 32, 5, padding=1)
        self.conv1 = nn.Conv2d(3, 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        conv_out_size = self._get_conv_output(input_shape)

        self.lin1 = nn.Linear(conv_out_size, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, num_actions)

    def _get_conv_output(self, shape):
        input = Variable(torch.rand(1, *shape))
        output_feat = self._forward_conv(input)
        # print("Conv out shape: %s" % str(output_feat.size()))
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_conv(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        return x

    def forward(self, states):
        x = self._forward_conv(states)

        # flattening each element in the batch
        x = x.view(states.size(0), -1)

        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        return self.lin3(x)

    def __call__(self, x):
        return self.forward(x)
