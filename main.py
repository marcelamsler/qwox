import functools

import numpy as np

from gym import spaces
from gym.spaces import Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


def env():
    env = raw_env()
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "qwoxxv1",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self):
        self.possible_agents = ["player_1", "player_2"]

        [1, 2, 3, 4]
        [1, 2, 3, 4]
        [4, 3, 2, 1]
        [4, 3, 2, 1]
        action_space = Box(low=1, high=10, shape=(10, 4), dtype=np.int8)
        self._action_spaces = {agent: action_space for agent in self.agents}

        observation_space = Box(low=1, high=10, shape=(10, 4), dtype=np.int8)
        self._observation_spaces = {agent: observation_space for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return Box(low=1, high=10, shape=(10, 4), dtype=np.int8)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(low=1, high=10, shape=(10, 4), dtype=np.int8)

    def render(self, mode="human"):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        # TODO

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        TODO should return possible actions and the board values of a player
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        # TODO
        # self.state = {agent: NONE for agent in self.agents}
        # self.observations = {agent: NONE for agent in self.agents}
        # self.num_moves = 0

        self._agent_selector.reset()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_done_step(action)

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # TODO set state on playing board
        # stores action of current agent
        self.state[self.agent_selection] = action

        self.rewards[self.agent_selection] = _calculate_reward(self.agent_selection)
        # TODO check when all are done, maybe move to end of step. But check what happens if multiple agents are simultaniously finished with their rows
        # self.dones = {agent: self.num_moves >= NUM_ITERS for agent in self.agents}
        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():

            # observe the current state
            # TODO set observation state
            # TODO set rewards for this agent after the action


        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        def _calculate_reward(agent):
            # TODO calculate reward based on state/observation
            return 1

        def render(self, mode="human"):
            #TODO render it
            pass

if __name__ == '__main__':
    env()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
