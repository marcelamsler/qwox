import functools
from typing import Optional

import numpy as np
from gym import spaces

from gym.spaces import Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env import AgentID

from game_models.board import Board
from game_models.game_card import GameCard


class QwoxEnv(AECEnv):
    WHITE_DICE_ACTION = "white_dice_action"
    COLOR_DICE_ACTION = "color_dice_action"

    def state(self) -> np.ndarray:
        return self.get_state_from(self.board)

    def seed(self, seed: Optional[int] = None) -> int:
        return 42

    metadata = {
        "render_modes": ["human"],
        "name": "qwoxxv1",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self):
        super().__init__()

        self.observations = None
        self.possible_agents: list[AgentID] = [AgentID("player_1"), AgentID("player_2")]
        self.agents = self.possible_agents[:]

        action_space = spaces.Dict({self.WHITE_DICE_ACTION: Box(low=0, high=1, shape=(11, 4), dtype=np.int8),
                                    self.COLOR_DICE_ACTION: Box(low=0, high=1, shape=(11, 4), dtype=np.int8)})
        self._action_spaces = {agent: action_space for agent in self.agents}
        self.agent_selection = self.agents[0]
        self.rewards: {AgentID: int} = {agent: 0 for agent in self.agents}
        self._cumulative_rewards: {AgentID: int} = {agent: 0 for agent in self.agents}

        observation_space = spaces.Dict(
            {
                "observation": Box(low=0, high=1, shape=(11, 4), dtype=np.int8),
                "action_mask": Box(low=0, high=1, shape=(11, 4), dtype=np.int8),
            }
        )

        self._observation_spaces = {agent: observation_space for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.board: Board = Board(self.possible_agents)
        self.dones: {AgentID: bool} = {agent_id: False for agent_id in self.agents}
        self.total_finished_step_count = 0
        self.current_tosser_index = 0

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return Box(low=1, high=10, shape=(10, 4), dtype=np.int8)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(low=0, high=1, shape=(11, 4), dtype=np.int8)

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
        """
        game_card: GameCard = self.board.game_cards[agent]

        return {"observation": game_card.get_state(), "action_mask": game_card.get_allowed_actions_mask()}

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
        self.agents: list[AgentID] = self.possible_agents[:]
        self.board = Board(player_ids=self.possible_agents)
        self.rewards: {AgentID: int} = {agent: 0 for agent in self.agents}
        self._cumulative_rewards: {AgentID: int} = {agent: 0 for agent in self.agents}
        self.dones: {AgentID: bool} = {agent_id: False for agent_id in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.total_finished_step_count = 0
        self.current_tosser_index = 0
        self.agent_selection = self.agents[0]

        self._agent_selector.reset()

    @staticmethod
    def get_state_from(board: Board) -> np.ndarray:
        # Todo flatten State
        return []

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

        current_agent_id: AgentID = self.agent_selection
        is_tossing_agent = self.current_tosser_index == self.agents.index(current_agent_id)
        current_game_card: GameCard = self.board.game_cards[current_agent_id]
        starting_points = current_game_card.points

        # General Action for white dices which anyone can do
        current_game_card.cross_value_with_action(action[self.WHITE_DICE_ACTION])

        # Specific Action which can only be done by tossing-agent
        if is_tossing_agent:
            current_game_card.cross_value_with_action(action[self.COLOR_DICE_ACTION])
        elif not is_tossing_agent and np.count_nonzero(action[self.COLOR_DICE_ACTION]) > 0:
            print("Agent is not tossing but tried to set color-dice action")

        self.rewards[self.agent_selection] = self.board.game_cards[current_agent_id].points - starting_points

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            self.dones = {agent: self.board.game_is_finished() for agent in self.agents}

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Check if next agent has to toss the dices and do it before the next agent takes its step, as the dice
        # state need to be known by the agent that takes a step
        self.total_finished_step_count += 1
        self.update_tossing_agent()
        next_agent_tossing_agent = self.current_tosser_index == self.agents.index(current_agent_id)
        if next_agent_tossing_agent:
            self.board.roll_dices()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


    def render(self, mode="human"):
        # TODO render it
        pass

    def update_tossing_agent(self) -> None:
        if self.total_finished_step_count % self.num_agents == 0:
            if self.current_tosser_index + 1 >= self.num_agents:
                self.current_tosser_index = 0
            else:
                self.current_tosser_index += 1
