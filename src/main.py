from pettingzoo.utils import wrappers

from env.qwox_env import QwoxEnv


def env():
    env = QwoxEnv()
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env





if __name__ == '__main__':
    env()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
