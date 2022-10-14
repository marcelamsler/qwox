import unittest

from tianshou.data import Batch

from agents.tianshou.lowest_value_taker_policy import LowestValueTakerPolicy


class LowestValueTakerPolicyTest(unittest.TestCase):

    def test_batch(self):
        policy = LowestValueTakerPolicy()
        action_array = policy.forward(self.get_example_batch())
        self.assertEqual(6, action_array[0]["act"])

    def get_example_batch(self):
        batch = Batch()
        batch["obs"] = Batch()
        batch["obs"]["obs"] = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                               [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
        batch["obs"]["mask"] = [[False, False, False, False, False, False, True, False, False,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, False,
                                 False, True, False, False, False, False, False, False, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True],
                                [False, False, False, False, False, False, True, False, False,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, False,
                                 False, True, False, False, False, False, False, False, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True],
                                [False, False, False, False, False, False, True, False, False,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, False,
                                 False, True, False, False, False, False, False, False, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True],
                                [False, False, False, False, False, False, True, False, False,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, False,
                                 False, True, False, False, False, False, False, False, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True],
                                [False, False, False, False, False, False, True, False, False,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, False,
                                 False, True, False, False, False, False, False, False, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True],
                                [False, False, False, False, False, False, True, False, False,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, False,
                                 False, True, False, False, False, False, False, False, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True],
                                [False, False, False, False, False, False, True, False, False,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, False,
                                 False, True, False, False, False, False, False, False, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True],
                                [False, False, False, False, False, False, True, False, False,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, False,
                                 False, True, False, False, False, False, False, False, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True],
                                [False, False, False, False, False, False, True, False, False,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, False,
                                 False, True, False, False, False, False, False, False, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True],
                                [False, False, False, False, False, False, True, False, False,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, True,
                                 False, False, False, False, False, False, False, False, False,
                                 False, True, False, False, False, False, False, False, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True]]

        return batch
