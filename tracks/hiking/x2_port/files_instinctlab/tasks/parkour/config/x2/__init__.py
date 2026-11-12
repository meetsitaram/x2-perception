import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.parkour.config.x2"

gym.register(
    id="Instinct-Parkour-Target-Amp-X2-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.x2_parkour_target_amp_cfg:X2ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:X2ParkourPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Parkour-Target-Amp-X2-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.x2_parkour_target_amp_cfg:X2ParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:X2ParkourPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Parkour-TorsoCam-Amp-X2-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.x2_parkour_target_amp_cfg:X2ParkourTorsoCamEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:X2ParkourPPORunnerCfg",
    },
)

# Extended task (new terrain columns, velocity-mode commands, stumble and clearance terms). Written on the CPU, not yet run in Isaac.
gym.register(
    id="Instinct-Parkour-Extended-Amp-X2-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.extended:X2ParkourExtendedEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:X2ParkourPPORunnerCfg",
    },
)

# Candidate deviations for getting off the stand (HIKING_X2_OPTS); identical to the run-2 task when no option is set. Not yet run in Isaac.
gym.register(
    id="Instinct-Parkour-Staged-Amp-X2-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.staged:X2ParkourStagedEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:X2ParkourPPORunnerCfg",
    },
)
