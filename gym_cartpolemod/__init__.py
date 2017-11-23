import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='CartPoleMod-v0',
    entry_point='gym_cartpolemod.envs:CartPoleModEnv',
)