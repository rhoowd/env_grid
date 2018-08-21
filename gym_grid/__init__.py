from gym.envs.registration import register

register(
id='grid-dsdp-v0',
entry_point='gym_grid.envs:DsdpEnv',
)

register(
id='grid-dsdp-v1',
entry_point='gym_grid.envs:DsdpEnvOneHot',
)