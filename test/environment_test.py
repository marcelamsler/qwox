import unittest
from pettingzoo.test import api_test

from env.qwox_env import QwoxEnv


class MyTestCase(unittest.TestCase):

    def test_env(self):
        env = QwoxEnv()
        api_test(env, num_cycles=10, verbose_progress=False)

    def test_tossing_agent_calculation(self):
        env = QwoxEnv()
        self.assertEqual(0, env.current_tosser_index)
        # Simulate one round, where every agent played once
        env.total_finished_step_count = env.num_agents
        # Update as done every step
        env.update_tossing_agent()

        # next round the agent with index one should toss the dices
        self.assertEqual(1, env.current_tosser_index)

    def test_tossing_agent_calculation_in_env(self):
        env = QwoxEnv()
        self.assertEqual(0, env.current_tosser_index)
        # Simulate one round, where every agent played once
        for agent in env.agents:
            env.step([])

        # next round the agent with index one should toss the dices
        self.assertEqual(1, env.current_tosser_index)

    def test_get_round(self):
        first_round = QwoxEnv.get_round(total_finished_step_count=4, agent_count=4)
        self.assertEqual(1, first_round)

        after_last_action_in_round = QwoxEnv.get_round(total_finished_step_count=5, agent_count=4)
        self.assertEqual(2, after_last_action_in_round)

        some_round = QwoxEnv.get_round(total_finished_step_count=13, agent_count=2)
        self.assertEqual(5, some_round)

if __name__ == '__main__':
    unittest.main()
