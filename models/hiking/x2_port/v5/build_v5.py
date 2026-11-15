"""Build the v5 files (2026-11-13 00:50). Writes only new files (.v5 copies, new modules, a new data dir); the live v4j files
are untouched. continue_v5.sh swaps them in at the restart.
 staged.py.v5 = staged.py + option j in the allowed set
 fixed.py.v5  = fixed.py; with j the discriminator input is the weighted 23 joints (style_weights.py) instead of the 12 legs,
                and a checkpoint discriminator of another shape is dropped; simcheck terminations (simfault.py) always
 supervise_x2_v5.py = supervise_x2.py + the simcheck heartbeat hunk (tools/simcheck/patches/hiking.diff, second file)
 data/motion_paper_x2/ = yaml for the paper's reference retargeted to X2 (npz copied in by continue_v5.sh once it exists,
                renamed *_retargetted.npz: the loader dispatches on that suffix)"""
import os, re, shutil, subprocess

T = "/mnt/filesystem-g4/tracks/hiking"
X = f"{T}/code/x2_port"
D = f"{X}/files_instinctlab/tasks/parkour/config/x2"
V5 = f"{X}/v5"

s = open(f"{D}/staged.py").read()
old = 'unknown = opts - set("abcdefgrstkyzowv")'
assert old in s
s = s.replace(old, 'unknown = opts - set("abcdefgrstkyzowvj")  # j: v5 weighted style input (fixed.py.v5, 2026-11-13)', 1)
open(f"{D}/staged.py.v5", "w").write(s)

f = open(f"{D}/fixed.py").read()
old = """        for group in (self.observations.amp_policy, self.observations.amp_reference):
            for term in ("joint_pos_rel", "joint_vel"):
                old = getattr(group, term).params["asset_cfg"]  # a new object, so no other group that shares it changes
                getattr(group, term).params["asset_cfg"] = SceneEntityCfg(old.name, joint_names=list(LEG_JOINTS))
"""
assert f.endswith(old), "fixed.py changed since the v5 build was written"
new = """        import os as _os
        if "j" in {o.strip() for o in _os.environ.get("HIKING_X2_OPTS", "").split(",")}:
            # v5 (user 2026-11-13): legs, pelvis, waist 1.0; shoulders, elbows 0.3; wrists, head out (style_weights.py)
            from instinct_rl.algorithms.wasabi import WasabiPPO
            from .style_weights import apply_to_cfg, install_discriminator_reset
            apply_to_cfg(self, SceneEntityCfg)
            install_discriminator_reset(WasabiPPO)
        else:
            for group in (self.observations.amp_policy, self.observations.amp_reference):
                for term in ("joint_pos_rel", "joint_vel"):
                    old = getattr(group, term).params["asset_cfg"]  # a new object, so no other group that shares it changes
                    getattr(group, term).params["asset_cfg"] = SceneEntityCfg(old.name, joint_names=list(LEG_JOINTS))
        # simcheck (TRACK_RULES "Simulator checks"): simulator faults end as time-outs, never failures (simfault.py)
        from isaaclab.managers import TerminationTermCfg as _DoneTerm
        from .simfault import install as _install_simfault
        print("[X2 simfault] wrapped terminations:", _install_simfault(self, _DoneTerm), flush=True)
"""
open(f"{D}/fixed.py.v5", "w").write(f[: -len(old)] + new)

# supervisor with the simcheck heartbeat hunk
patch = open("/mnt/filesystem-g4/tools/simcheck/patches/hiking.diff").read()
sup = patch[patch.index("--- a/x2_port/supervise_x2.py"):]
tmp = f"{V5}/supervise_x2_v5.py"
shutil.copy(f"{X}/supervise_x2.py", tmp)
open(f"{V5}/supervise.diff", "w").write(sup)
r = subprocess.run(["patch", "--no-backup-if-mismatch", tmp, f"{V5}/supervise.diff"], capture_output=True, text=True)
print("patch supervise_x2_v5.py:", r.returncode, r.stdout.strip()[-200:], r.stderr.strip()[-200:])
assert r.returncode == 0

M = f"{T}/data/motion_paper_x2"
os.makedirs(M, exist_ok=True)
open(f"{M}/parkour_motion_without_run_x2.yaml", "w").write("selected_files:\n- parkour_motion_without_run_x2_retargetted.npz\n")
open(f"{M}/README.txt", "w").write(
    "The paper's own motion reference (380 s G1 file) retargeted to X2 by another agent (user decision 2026-11-13).\n"
    "Source: /mnt/filesystem-g4/data/motions/hiking_release_x2/parkour_motion_without_run_x2.npz; continue_v5.sh copies it here as\n"
    "parkour_motion_without_run_x2_retargetted.npz (AmassMotion reads only *retargetted.npz / *retargeted.npz as retargeted data).\n")
print("built:", ", ".join(sorted(x for x in os.listdir(D) if x.endswith(".v5"))), "| supervise_x2_v5.py | data/motion_paper_x2/")
