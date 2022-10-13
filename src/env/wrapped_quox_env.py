from pettingzoo.utils.conversions import turn_based_aec_to_parallel_wrapper

from env.qwox_env import QwoxEnv
from pettingzoo.utils import wrappers
from env.qwox_env import QwoxEnv


def wrapped_quox_env():
    env = QwoxEnv()
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

# def ss_wrapped_quox_env():
#     env_ = QwoxEnv()
#     #env_ = wrappers.CaptureStdoutWrapper(env_)
#     #env_ = wrappers.TerminateIllegalWrapper(env_, illegal_reward=-1)
#     #env_ = wrappers.OrderEnforcingWrapper(env_)
#     env_ = turn_based_aec_to_parallel_wrapper(env_)
#     env_ = ss.pettingzoo_env_to_vec_env_v1(env_)
#     env_ = ss.concat_vec_envs_v1(env_, 1, base_class="stable_baselines3")
#     return env_


if __name__ == '__main__':
    wrapped_quox_env()
