import os
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import wandb
import torch
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy, RainbowPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import WandbLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter

from agents.tianshou.lowest_value_taker_policy import LowestValueTakerPolicy
from env.wrapped_quox_env import wrapped_quox_env


def _get_agents(
        wandb,
        agent_learn: Optional[BasePolicy] = None,
        agent_opponent: Optional[BasePolicy] = None,
        optim: Optional[torch.optim.Optimizer] = None,
        opponent_path: str = None
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:

    env = _get_env(wandb)
    observation_space = (
        env.observation_space["observation"]
    )
    if agent_learn is None:
        # model
        net = Net(
            state_shape=observation_space.shape or observation_space.n,
            action_shape=env.action_space.shape or env.action_space.n,
            hidden_sizes=[128, 128, 128, 128, 128, 128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-5)

        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.99,
            estimation_step=1,
            target_update_freq=500
        )

        wandb.watch(net)

    if agent_opponent is None:
        if opponent_path:
            agent_opponent = deepcopy(agent_learn)
            agent_opponent.load_state_dict(torch.load(opponent_path))
        else:
            agent_opponent = RandomPolicy()

    agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents

def _get_env(wandb):
    """This function is needed to provide callables for DummyVectorEnv."""
    env = wrapped_quox_env()
    env.unwrapped.wandb = wandb
    return PettingZooEnv(env)


if __name__ == "__main__":
    log_path = os.path.join("log", "summary.log")
    logger = WandbLogger(project="Tianshou1")

    logger.load(SummaryWriter(log_path))
    # ======== Step 1: Environment setup =========
    _get_env_with_wandb = lambda: _get_env(logger.wandb_run)
    train_envs = DummyVectorEnv([_get_env_with_wandb for _ in range(10)])
    test_envs = DummyVectorEnv([_get_env_with_wandb for _ in range(10)])

    # seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    path = os.path.join("log", "rps", "dqn", "policy-100.pth")
    policy, optim, agents = _get_agents(logger.wandb_run, agent_opponent=LowestValueTakerPolicy())


    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20_000, len(train_envs), alpha=0.001, beta=0.001),
        #PrioritizedVectorReplayBuffer(20_000, len(train_envs), alpha=0.001, beta=0.001),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    train_collector.collect(n_step=64 * 10)  # batch size * training_num




    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("log", "rps", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "rps", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)


    def stop_fn(mean_rewards):
        return mean_rewards >= 70


    def train_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.1)


    def test_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0)


    def reward_metric(rews):
        return rews[:, 1]


    logger = WandbLogger()
    logger.load(SummaryWriter(log_path))

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=750,
        step_per_epoch=1000,
        step_per_collect=100,
        episode_per_test=20,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
        logger=logger
    )


    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")
    logger.wandb_run.finish()
