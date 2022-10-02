import unittest
from pettingzoo.test import api_test

from env.qwox_env import QwoxEnv
from env.wrapped_quox_env import wrapped_quox_env


class MyTestCase(unittest.TestCase):

    def test_env(self):
        env = QwoxEnv()
        api_test(env, num_cycles=10, verbose_progress=True)

    # Test does not work, as api_test sends ints as actions instead of
    # proper array based on action space, which causes IllegalActionWrapper to
    # exit, as the int is not part of the unflattened action_mask
    #def test_wrapped_env(self):
    #    env = wrapped_quox_env()
    #    api_test(env, num_cycles=20, verbose_progress=True)

    def test_get_round(self):
        first_round = QwoxEnv.get_round(total_finished_step_count=4, agent_count=4)
        self.assertEqual(1, first_round)

        after_last_action_in_round = QwoxEnv.get_round(total_finished_step_count=5, agent_count=4)
        self.assertEqual(2, after_last_action_in_round)

        some_round = QwoxEnv.get_round(total_finished_step_count=13, agent_count=2)
        self.assertEqual(5, some_round)

if __name__ == '__main__':
    unittest.main()
