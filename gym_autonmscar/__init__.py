from gym.envs.registration import register

register(
    id='autonmscar-v0',
    entry_point='gym_autonmscar.envs:AutonomousCarEnv',
)
