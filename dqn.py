import numpy as np
import torch

from gym.envs.classic_control.cartpole import CartPoleEnv

from torch import nn


class DQN(nn.Module):
    def __init__(self, env: CartPoleEnv, optimizer_lr: float):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.optimizer_lr = optimizer_lr
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=env.action_space.n)
        )

        self.optimizer = torch.optim.Adam(self.net.parameters(), self.optimizer_lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_action(self, state: np.ndarray) -> int:
        state_t = torch.FloatTensor(np.expand_dims(state, axis=0))
        q_values = self(state_t)

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action
