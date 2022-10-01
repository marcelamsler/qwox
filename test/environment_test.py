import unittest
from pettingzoo.test import api_test

from env.qwox_env import QwoxEnv


class MyTestCase(unittest.TestCase):

    def test_env(self):
        env = QwoxEnv()
        api_test(env, num_cycles=100, verbose_progress=False)

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

if __name__ == '__main__':
    unittest.main()
