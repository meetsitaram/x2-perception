"""v5 option j (user decision 2026-11-13, docs/USER_RULES.md): the single discriminator (the paper's) sees joints weighted by
body part: legs, pelvis (base terms) and waist 1.0, shoulders and elbows 0.3, wrists and head excluded. Until v4k it saw the 12
leg joints only (fixed.py item a). The input grows from 10 x 33 = 330 to 10 x 55 = 550, so a checkpoint's discriminator cannot be
loaded: `install_discriminator_reset` makes WasabiPPO.load_state_dict drop a discriminator (and its optimizer) whose shapes differ,
which starts a fresh discriminator on the new motion reference; the policy, value and their optimizer load as before.
Pure torch; tested on CPU by v5/precheck_v5.py."""
from __future__ import annotations

import torch

SIDES = ("left", "right")
LEGS = [f"{s}_{j}_joint" for s in SIDES for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
WAIST = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
ARMS = [f"{s}_{j}_joint" for s in SIDES for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
STYLE_JOINTS = LEGS + WAIST + ARMS  # 23; wrists and head are not in the input at all


def joint_weight(name: str) -> float:
    if "wrist" in name or "head" in name:
        return 0.0
    if "shoulder" in name or "elbow" in name:
        return 0.3
    return 1.0


_cache: dict = {}


def _weights(asset_cfg, n: int, device) -> torch.Tensor:
    names = list(asset_cfg.joint_names) if isinstance(asset_cfg.joint_names, (list, tuple)) else []
    key = (asset_cfg.name, tuple(names), str(device))
    w = _cache.get(key)
    if w is None:
        if len(names) != n:
            raise RuntimeError(f"style_weights: {asset_cfg.name} gives {n} joints but its cfg names {len(names)}: {names}")
        w = torch.tensor([joint_weight(x) for x in names], device=device, dtype=torch.float)
        _cache[key] = w
    return w


def _apply(out: torch.Tensor, asset_cfg) -> torch.Tensor:
    return out * _weights(asset_cfg, out.shape[-1], out.device)


# 2026-11-13 fix: Isaac Lab's term check (manager_base._resolve_common_term_cfg) reads the signature and counted `**kw` as a
# mandatory parameter "kw" (v5 construction check 01:32). Explicit keyword parameters with defaults, like the wrapped functions.
def _kw(**k):
    return {a: b for a, b in k.items() if b is not None}


def joint_pos_rel_w(env, asset_cfg=None):
    import instinctlab.tasks.parkour.mdp as mdp
    return _apply(mdp.joint_pos_rel(env, asset_cfg=asset_cfg), asset_cfg)


def joint_vel_rel_w(env, asset_cfg=None):
    import instinctlab.tasks.parkour.mdp as mdp
    return _apply(mdp.joint_vel_rel(env, asset_cfg=asset_cfg), asset_cfg)


def joint_pos_rel_reference_w(env, asset_cfg=None, robot_cfg=None):
    import instinctlab.tasks.parkour.mdp as mdp
    return _apply(mdp.joint_pos_rel_reference_as_state(env, **_kw(asset_cfg=asset_cfg, robot_cfg=robot_cfg)), asset_cfg)


def joint_vel_rel_reference_w(env, asset_cfg=None, robot_cfg=None, mask=True):
    import instinctlab.tasks.parkour.mdp as mdp
    return _apply(mdp.joint_vel_rel_reference_as_state(env, **_kw(asset_cfg=asset_cfg, robot_cfg=robot_cfg), mask=mask), asset_cfg)


WRAPPED = {  # original func name -> weighted twin
    "joint_pos_rel": joint_pos_rel_w,
    "joint_vel_rel": joint_vel_rel_w,
    "joint_pos_rel_reference_as_state": joint_pos_rel_reference_w,
    "joint_vel_rel_reference_as_state": joint_vel_rel_reference_w,
}


def apply_to_cfg(cfg, SceneEntityCfg):
    """Called at the end of X2ParkourFixedEnvCfg.__post_init__ when option j is on (replaces the 12-leg-joint selection)."""
    for group in (cfg.observations.amp_policy, cfg.observations.amp_reference):
        for term in ("joint_pos_rel", "joint_vel"):
            t = getattr(group, term)
            old = t.params["asset_cfg"]
            t.params["asset_cfg"] = SceneEntityCfg(old.name, joint_names=list(STYLE_JOINTS), preserve_order=True)
            fname = getattr(t.func, "__name__", "")
            if fname not in WRAPPED:
                raise RuntimeError(f"style_weights: unexpected func {fname} in {term}")
            t.func = WRAPPED[fname]


def drop_mismatched_discriminator(own: dict, state_dict: dict) -> list:
    """Remove `discriminator` / `discriminator_optimizer` from state_dict when any tensor shape differs from `own`
    (the discriminator's current state dict). Returns the list of mismatches (empty = kept)."""
    sd = state_dict.get("discriminator")
    if sd is None:
        return []
    bad = [(k, tuple(v.shape), tuple(own[k].shape) if k in own else None) for k, v in sd.items()
           if k not in own or tuple(own[k].shape) != tuple(v.shape)]
    bad += [(k, None, tuple(v.shape)) for k, v in own.items() if k not in sd]
    if bad:
        state_dict.pop("discriminator", None)
        state_dict.pop("discriminator_optimizer", None)
    return bad


def install_discriminator_reset(WasabiPPO):
    if getattr(WasabiPPO, "_x2_disc_reset", False):
        return
    orig = WasabiPPO.load_state_dict

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        bad = drop_mismatched_discriminator(self.discriminator.state_dict(), state_dict)
        if bad:
            print(f"[X2 v5 style] discriminator shapes changed ({bad[0]}): fresh discriminator, policy loaded", flush=True)
        return orig(self, state_dict)

    WasabiPPO.load_state_dict = load_state_dict
    WasabiPPO._x2_disc_reset = True
