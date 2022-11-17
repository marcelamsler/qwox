from typing import Union, Optional, Dict, Any, Tuple

import numpy as np
import torch
from gym.spaces import Sequence
from tianshou.data import Batch
from tianshou.policy import C51Policy
from tianshou.utils.net.discrete import NoisyLinear
from torch import nn


class RainbowNet(C51Policy):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Union[int, Sequence[int]],
            num_atoms: int = 51,
            noisy_std: float = 0.5,
            device: Union[str, int, torch.device] = "cpu",
            is_dueling: bool = True,
            is_noisy: bool = True,
    ) -> None:
        super().__init__(c, h, w)
        self.action_num = np.prod(action_shape)
        self.num_atoms = num_atoms

        def linear(x, y):
            if is_noisy:
                return NoisyLinear(x, y, noisy_std)
            else:
                return nn.Linear(x, y)

        self.Q = nn.Sequential(
            linear(self.output_dim, 512), nn.ReLU(inplace=True),
            linear(512, self.action_num * self.num_atoms)
        )
        self._is_dueling = is_dueling
        if self._is_dueling:
            self.V = nn.Sequential(
                linear(self.output_dim, 512), nn.ReLU(inplace=True),
                linear(512, self.num_atoms)
            )
        self.output_dim = self.action_num * self.num_atoms

    def forward(self, obs: Batch, state: Optional[Any] = None, info: Dict[str, Any] = {},
                **kwargs) -> Tuple[torch.Tensor, Any]:

        obs, state = super().forward(obs)
        q = self.Q(obs)
        q = q.view(-1, self.action_num, self.num_atoms)
        if self._is_dueling:
            v = self.V(obs)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        probs = logits.softmax(dim=2)
        return probs, state
