import functools
from typing import Optional

import numpy as np
from gym import spaces

from gym.spaces import Box
from numpy import int64
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
        self.current_round = 1
        self.observations = None
        self.possible_agents: list[AgentID] = [AgentID("player_1"), AgentID("player_2")]
        self.agents = self.possible_agents[:]

        self._action_spaces = {agent: self.action_space(agent) for agent in self.agents}
        self.agent_selection = self.agents[0]
        self.rewards: {AgentID: int} = {agent: 0 for agent in self.agents}
        self._cumulative_rewards: {AgentID: int} = {agent: 0 for agent in self.agents}

        self._observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}
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
        return spaces.Dict(
            {
                "observation": Box(low=0, high=1, shape=(4, 11), dtype=np.int8),
                "action_mask": Box(low=0, high=1, shape=(4, 11), dtype=np.int8)
            })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(low=0, high=1, shape=(4, 11), dtype=np.int8)

    def render(self, mode="human"):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if mode == "human":
            print("########")
            print("Dices", self.board.dices)
            for agent_index, agent in enumerate(self.agents):
                is_tossing_agent = self.get_tossing_agent_index(self.current_round) == agent_index
                print("---------------------------------------")
                print(agent, "round", self.current_round)
                if is_tossing_agent:
                    print("Is Tossing Agent")
                print(self.observe(agent)["observation"][0])
                print(self.observe(agent)["observation"][1])
                print(self.observe(agent)["observation"][2])
                print(self.observe(agent)["observation"][3])
                print("---------------------------------------")

    def observe(self, agent: AgentID):
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

        self.current_round = QwoxEnv.get_round(self.total_finished_step_count, self.num_agents)
        current_agent_id: AgentID = self.agent_selection
        is_tossing_agent = self.get_tossing_agent_index(self.current_round) == self.agents.index(current_agent_id)
        all_common_actions_done_in_round = self.are_all_common_actions_done_in_round()
        if all_common_actions_done_in_round and not is_tossing_agent:
            print("skip agent ", current_agent_id, "with action", action)
            return

        current_game_card: GameCard = self.board.game_cards[current_agent_id]
        starting_points = current_game_card.points

        # DO ACTION
        if isinstance(action, int64):
            current_game_card.cross_value_with_flattened_action(action)
        else:
            current_game_card.cross_value_with_action(action)

        self.rewards[self.agent_selection] = self.board.game_cards[current_agent_id].points - starting_points

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            self.dones = {agent: self.board.game_is_finished() for agent in self.agents}

        self.render()
        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Check if next agent has to toss the dices and do it before the next agent takes its step, as the dice
        # state need to be known by the agent that takes a step
        self.total_finished_step_count += 1
        next_agent_tossing_agent = self.get_tossing_agent_index(self.current_round) == self.agents.index(
            current_agent_id)
        if next_agent_tossing_agent:
            self.board.roll_dices()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    def are_all_common_actions_done_in_round(self):
        steps_in_one_round = self.num_agents + 1
        actions_done_in_this_round = self.total_finished_step_count % steps_in_one_round
        all_common_actions_done_in_round = actions_done_in_this_round > steps_in_one_round - 1
        return all_common_actions_done_in_round

    def get_tossing_agent_index(self, round):
        return (round - 1) % self.num_agents

    @staticmethod
    def get_round(total_finished_step_count, agent_count) -> int:
        # round number always consists of an action of every agent + one action of the tossing agent
        # and we want to start
        steps_in_one_round = agent_count + 1
        # We want to start with round 1 not 0
        initial_offset = 1
        if total_finished_step_count < steps_in_one_round:
            return initial_offset
        else:
            offset_because_count_is_after_step = 1
            return total_finished_step_count // steps_in_one_round + initial_offset
