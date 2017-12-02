import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='CartPoleMod-v0',
    entry_point='gym_cartpolemod.envs:CartPoleModEnv',
    kwargs={'case':1},
)
register(
    id='CartPoleMod-v1',
    entry_point='gym_cartpolemod.envs:CartPoleModEnv',
    kwargs={'case':2},
)
register(
    id='CartPoleMod-v2',
    entry_point='gym_cartpolemod.envs:CartPoleModEnv',
    kwargs={'case':3},
)
register(
    id='CartPoleMod-v3',
    entry_point='gym_cartpolemod.envs:CartPoleModEnv',
    kwargs={'case':4},
)
register(
    id='CartPoleMod-v4',
    entry_point='gym_cartpolemod.envs:CartPoleModEnv',
    kwargs={'case':5},
)
register(
    id='CartPoleMod-v5',
    entry_point='gym_cartpolemod.envs:CartPoleModEnv',
    kwargs={'case':6},
)
register(
    id='CartPoleMod-v6',
    entry_point='gym_cartpolemod.envs:CartPoleModEnv',
    kwargs={'case':7},
)