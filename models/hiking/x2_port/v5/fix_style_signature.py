import sys, re
p = sys.argv[1]
s = open(p).read()
old_new = [
 ("def joint_pos_rel_w(env, asset_cfg, **kw):", "def joint_pos_rel_w(env, asset_cfg=None):"),
 ("mdp.joint_pos_rel(env, asset_cfg=asset_cfg, **kw)", "mdp.joint_pos_rel(env, asset_cfg=asset_cfg)"),
 ("def joint_vel_rel_w(env, asset_cfg, **kw):", "def joint_vel_rel_w(env, asset_cfg=None):"),
 ("mdp.joint_vel_rel(env, asset_cfg=asset_cfg, **kw)", "mdp.joint_vel_rel(env, asset_cfg=asset_cfg)"),
 ("def joint_pos_rel_reference_w(env, asset_cfg, **kw):", "def joint_pos_rel_reference_w(env, asset_cfg=None, robot_cfg=None):"),
 ("mdp.joint_pos_rel_reference_as_state(env, asset_cfg=asset_cfg, **kw)", "mdp.joint_pos_rel_reference_as_state(env, **_kw(asset_cfg=asset_cfg, robot_cfg=robot_cfg))"),
 ("def joint_vel_rel_reference_w(env, asset_cfg, **kw):", "def joint_vel_rel_reference_w(env, asset_cfg=None, robot_cfg=None, mask=True):"),
 ("mdp.joint_vel_rel_reference_as_state(env, asset_cfg=asset_cfg, **kw)", "mdp.joint_vel_rel_reference_as_state(env, **_kw(asset_cfg=asset_cfg, robot_cfg=robot_cfg), mask=mask)"),
]
for o, n in old_new:
    assert s.count(o) == 1, o
    s = s.replace(o, n)
helper = '''

# 2026-11-13 fix: Isaac Lab's term check (manager_base._resolve_common_term_cfg) reads the signature and counted `**kw` as a
# mandatory parameter "kw" (v5 construction check 01:32). Explicit keyword parameters with defaults, like the wrapped functions.
def _kw(**k):
    return {a: b for a, b in k.items() if b is not None}


def joint_pos_rel_w('''
s = s.replace("\n\ndef joint_pos_rel_w(", helper, 1)
open(p, "w").write(s)
print("patched", p)
