import os

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net


def get_trained_agent(env, policy_number: int):
    net = Net(
        state_shape=env.observation_spaces[env.agents[0]]["observation"].shape,
        action_shape=env.observation_spaces[env.agents[0]]["action_mask"].shape,
        hidden_sizes=[128, 128, 128, 128, 128, 128],
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    optim = torch.optim.Adam(net.parameters(), lr=1e-5)

    agent = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=0.99,
        estimation_step=1,
        target_update_freq=500
    )

    path = os.path.join(f"trained-agents/policy-{policy_number}.pth")
    trained_agent = agent
    trained_agent.load_state_dict(torch.load(path))

    return trained_agent


def create_batch_from_observation(observation, info):
    return Batch(
        obs=Batch(obs=np.array([observation["observation"]]),
                  mask=np.array(np.array([observation["action_mask"]], dtype=bool))),
        info=info)


def create_batch_from_observation_for_handmade_policy(observation, info):
    return Batch(
        obs=Batch(obs=np.array([observation["observation"].reshape(3, 5, 12)]),
                  mask=np.array([observation["action_mask"].reshape(5, 11)])),
        info=info)
