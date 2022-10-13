import random

import numpy as np

from stable_baselines3 import A2C
import multiprocessing

from env.wrapped_quox_env import ss_wrapped_quox_env

"""
This setup does not work, as the dependencies needed can't be made compatible. Also changes in gym/pettingzoo versions
to go further back would be needed. This would mean, the environment would not support the latest gym api and would
cause problems if other libs should be used

- pettingzoo==1.22.0
- git+https://github.com/carlosluis/stable-baselines3@fix_tests #does not support terminations/truncations
- stable-baselines3==1.6.2 # does only support gym 0.21 
- gym==0.26.2
- numpy==1.23.4
- supersuit==3.6.0

"""

def policy_random_agent(observation):
    if np.count_nonzero(observation["action_mask"][:44]) > 0:
        action = random.choice(np.flatnonzero(observation["action_mask"]))
    else:
        print("Couldn't find possible action, so taking 54")
        action = 54

    return action


def main():
    multiprocessing.set_start_method("fork")
    env = ss_wrapped_quox_env()

    model = A2C("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    env.reset()

    for i in range(2):

        for agent in env.agent_iter():

            observation, reward, done, info = env.last()

            if agent == env.agents[0]:
                action = policy_random_agent(observation)
            else:
                action, _state = model.predict(env, deterministic=True)

            env.step(action)
            if done:
                env.reset()


if __name__ == '__main__':
    main()
