from env.qwox_env import QwoxEnv
from pettingzoo.utils import wrappers

from env.qwox_env import QwoxEnv


def wrapped_quox_env():
    env = QwoxEnv()
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


if __name__ == '__main__':
    wrapped_quox_env()
