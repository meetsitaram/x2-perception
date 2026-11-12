from isaaclab.utils import configclass

from instinctlab.tasks.parkour.config.g1.agents.instinct_rl_amp_cfg import G1ParkourPPORunnerCfg


@configclass
class X2ParkourPPORunnerCfg(G1ParkourPPORunnerCfg):
    """Runner, network and algorithm settings of the paper, unchanged; only the experiment name differs."""

    experiment_name = "x2_parkour"
