import datetime
import functools
import logging
from typing import Optional

import numpy as np
from gym import spaces
from gym.spaces import Box, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AgentID

from game_models.board import Board
from game_models.game_card import GameCard


class QwoxEnv(AECEnv):
    WHITE_DICE_ACTION = "white_dice_action"
    COLOR_DICE_ACTION = "color_dice_action"
    ACTION_SPACE_SIZE = 55

    def state(self) -> np.ndarray:
        # Not needed as long as I don't use training methods as described in super
        return np.array([])

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
        self.total_started_step_count = 1
        self.wandb = None

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return spaces.Dict(
            {
                "observation": Box(low=0, high=6, shape=(
                    self.num_agents + 1, GameCard.OBSERVATION_SHAPE_ROWS, GameCard.OBSERVATION_SHAPE_COLUMNS),
                                   dtype=np.int8),
                "action_mask": Box(low=0, high=1, shape=(self.ACTION_SPACE_SIZE,), dtype=np.int8)
            })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.ACTION_SPACE_SIZE)

    def render(self, mode: str = "human"):
        print("")
        print("")
        print("Dices", self.board.dices)
        tossing_agent = "player" + str(self.get_tossing_agent_index(self.current_round) + 1)
        part_of_round = 2 if self.is_second_part_of_round(self.total_started_step_count, self.num_agents) else 1
        print("Round", self.current_round,
              "part", part_of_round,
              "| Tossing Agent: ",
              tossing_agent, "| Closed Rows",
              self.board.get_closed_row_indexes())
        print("\n")
        for agent in self.agents:
            print(agent, "| Passes used", self.board.game_cards[agent].get_pass_count(), "| Current Points:",
                  self.board.game_cards[agent].get_points(), end="                        ")
        print("")
        for agent in self.agents:
            observation = self.get_observation(agent)
            print("RED   ", observation[0], "           ", end='')
        print("")
        for agent in self.agents:
            observation = self.get_observation(agent)
            print("YELLOW", observation[1], "           ", end='')
        print("")
        for agent in self.agents:
            observation = self.get_observation(agent)
            print("GREEN ", observation[2], "           ", end='')
        print("")
        for agent in self.agents:
            observation = self.get_observation(agent)
            print("BLUE  ", observation[3], "           ", end='')
        print("\n")

    def get_observation(self, agent):
        obs = self.observe(agent)["observation"][0].reshape(5, 12).astype(str)
        np.putmask(obs, obs == "1", "X")
        np.putmask(obs, obs == "0", "-")
        return obs

    def render_for_one_agent(self, agent_id, mode="human", action="unknown"):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if mode == "human":
            print("#################################################")
            print("started steps", self.total_started_step_count)
            print("Dices", self.board.dices)
            is_tossing_agent = self.get_tossing_agent_index(self.current_round) == self.agents.index(agent_id)
            print("---------------------------------------")
            part_of_round = 2 if self.is_second_part_of_round(self.total_started_step_count, self.num_agents) else 1
            print(agent_id, "| Round", self.current_round,
                  "part", part_of_round,
                  "| Tossing Agent: ",
                  is_tossing_agent, "| Reward", self.rewards[agent_id], "| Action: ", action, "| Passes used",
                  self.board.game_cards[agent_id].get_pass_count(), "| Closed Rows",
                  self.board.get_closed_row_indexes())
            observation = self.get_observation(agent_id)
            print(observation[0])
            print(observation[1])
            print(observation[2])
            print(observation[3])
            print(observation[4])
            print("ACTION MASK:")
            action_mask = self.observe(agent_id)["action_mask"].reshape(5, 11)
            print(action_mask[0])
            print(action_mask[1])
            print(action_mask[2])
            print(action_mask[3])
            print(action_mask[4])
            print("---------------------------------------")

    def observe(self, agent: AgentID):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        is_tossing_agent = self.get_tossing_agent_index(self.current_round) == self.agents.index(agent)
        is_second_part_of_round = QwoxEnv.is_second_part_of_round(self.total_started_step_count, self.num_agents)
        return {"observation": self.board.get_observation(player_id=agent,
                                                          is_tossing_player=is_tossing_agent,
                                                          is_second_part_of_round=is_second_part_of_round),
                "action_mask": self.board.get_allowed_actions_mask(agent,
                                                                   is_tossing_player=is_tossing_agent,
                                                                   is_second_part_of_round=is_second_part_of_round).flatten()}

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
        self.board.roll_dices()
        self.rewards: {AgentID: int} = {agent: 0 for agent in self.agents}
        self._cumulative_rewards: {AgentID: int} = {agent: 0 for agent in self.agents}
        self.dones: {AgentID: bool} = {agent_id: False for agent_id in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.total_started_step_count = 1
        self.agent_selection = self.agents[0]
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
            return self._was_done_step(None)

        current_agent_id: AgentID = self.agent_selection
        is_tossing_agent = self.get_tossing_agent_index(self.current_round) == self.agents.index(current_agent_id)
        is_second_part_of_round = QwoxEnv.is_second_part_of_round(self.total_started_step_count, self.num_agents)
        current_game_card: GameCard = self.board.game_cards[current_agent_id]
        starting_points = current_game_card.get_points()

        if action is None:
            print("Agent chose no action ", current_agent_id)

        if is_second_part_of_round and not is_tossing_agent:
            logging.debug("skip agent ", current_agent_id, "with action", action)
            self.rewards = {agent: 0 for agent in self.agents}
        else:
            if action not in np.flatnonzero(self.observe(current_agent_id)["action_mask"]):
                raise Exception("Wrong action", action, self.observe(current_agent_id)["action_mask"].reshape(5, 11))

            # DO ACTION
            current_game_card.cross_value_with_flattened_action(action)

            for agent in self.agents:
                if agent == current_agent_id:
                    self.rewards[current_agent_id] = current_game_card.get_points() - starting_points
                else:
                    self.rewards[agent] = 0

            self.dones = {agent: self.board.is_game_finished() for agent in self.agents}
            if self.board.is_game_finished():
                learned_player_points = self.board.game_cards[self.agents[1]].get_points()
                opponent_player_points = self.board.game_cards[self.agents[0]].get_points()
                print("----------------------- > Total Rewards: Random Player:",
                      opponent_player_points,
                      "Learned Player: ", learned_player_points)

                self.render()
                self.log_to_wandb_if_possible_or_file(learned_player_points, opponent_player_points)

        self.set_state_for_next_step(current_game_card, is_second_part_of_round)

    def log_to_wandb_if_possible_or_file(self, learned_player_points, opponent_player_points):
        are_closed_rows_reason_for_finish = len(self.board.get_closed_row_indexes()) >= 2
        finish_reason = 1 if are_closed_rows_reason_for_finish else 0
        if self.wandb:

            self.wandb.log({"player_1_points": opponent_player_points,
                            "player_2_points": learned_player_points,
                            "player_1_passes": self.board.game_cards[self.agents[0]].get_pass_count(),
                            "player_2_passes": self.board.game_cards[self.agents[1]].get_pass_count(),
                            "closed_rows": len(self.board.get_closed_row_indexes()),
                            "point_difference": learned_player_points - opponent_player_points,
                            "finish_reason": finish_reason,
                            "agent1_winning": (1 if learned_player_points > opponent_player_points else 0)})
        else:
            with open('test-log.csv', 'a') as f:

                f.write(
                    "\n"
                    f"{datetime.datetime.now().now().strftime('%Y-%m-%d %H:%M:%S')},"
                    f"{opponent_player_points}, "
                    f"{learned_player_points}, "
                    f"{self.board.game_cards[self.agents[0]].get_pass_count()}, "
                    f"{self.board.game_cards[self.agents[1]].get_pass_count()}, "
                    f"{len(self.board.get_closed_row_indexes())}, "
                    f"{finish_reason}")

    def set_state_for_next_step(self, current_game_card, is_second_part_of_round):
        # Reset for next round
        if is_second_part_of_round:
            current_game_card.crossed_something_in_current_round = False
        self._cumulative_rewards[self.agent_selection] = 0
        # selects the next agent
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()
        if self.total_started_step_count % (self.num_agents * 2) == 0:
            self.board.roll_dices()
        self.total_started_step_count += 1
        self.current_round = QwoxEnv.get_round(self.total_started_step_count, self.num_agents)

    def get_tossing_agent_index(self, current_round):
        return (current_round - 1) % self.num_agents

    @staticmethod
    def get_round(total_started_step_count, agent_count) -> int:
        # round number always consists of an action of every agent + one action of the tossing agent
        steps_in_one_round = agent_count * 2
        # We want to start with round 1 not 0
        initial_offset = 1
        if total_started_step_count <= steps_in_one_round:
            return initial_offset
        else:
            return (total_started_step_count - 1) // steps_in_one_round + initial_offset

    @staticmethod
    def is_second_part_of_round(total_started_step_count, num_agents):
        steps_in_one_round = num_agents * 2

        steps_in_this_round = total_started_step_count % steps_in_one_round
        return steps_in_this_round > num_agents or steps_in_this_round == 0
