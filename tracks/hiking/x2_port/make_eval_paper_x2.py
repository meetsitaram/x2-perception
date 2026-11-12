"""Builds eval_paper_x2.py from the G1 reproduction's scripts/eval_paper.py (the paper's Table 1 and 2 protocol), changing only
what is robot- or platform-specific. Run again whenever the original changes:  python3 make_eval_paper_x2.py

X2 differences from the G1 evaluation (also in state/hiking.md):
- task Instinct-Parkour-Target-Amp-X2-v0 (X2 model, pelvis root, head camera);
- landing-area rays cover the X2 sole: 0.18 x 0.10 m centred 0.037 m ahead of the ankle (G1 shoe: 0.18 x 0.06 m at 0.039 m);
  sole height = ankle z - 0.073 m (from the task's feet_at_plane offset, as for the G1);
- the 1 x 1 m "interaction range" grid hangs under the pelvis (the X2 root) instead of the G1's torso root;
- "reaching" distance is measured from the pelvis; terminations are the X2 task's (root height 0.39 m);
- terrain geometry (Table 8), trial length, success radius, robot count: unchanged, i.e. the G1's obstacle sizes on a smaller robot.
Optional: HIKING_OBS_CHECK=<file.npz> records the first 50 control steps of robot 0 (observation, action, state) for the eval hand-over.
"""
import pathlib

T = pathlib.Path("/mnt/filesystem-g4/tracks/hiking/code")
s = (T / "hiking-in-the-wild/scripts/eval_paper.py").read_text()


def rep(a, b, count=1):
    global s
    assert s.count(a) >= 1, a
    s = s.replace(a, b) if count == 0 else s.replace(a, b, count)


rep('ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))', 'ROOT = "%s"' % (T / "hiking-in-the-wild"))
rep('PAPER_TXT = os.path.join(ROOT, "docs", "paper", "v2.txt")', 'PAPER_TXT = "%s"' % (T / "x2_port/paper_v2.txt"))
rep('TASK = "Instinct-Parkour-Target-Amp-G1-v0"', 'TASK = "Instinct-Parkour-Target-Amp-X2-v0"')
rep('subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=60)', 'proc.kill()')
rep('offset=RayCasterCfg.OffsetCfg(pos=(0.039, 0.0, 20.0)),', 'offset=RayCasterCfg.OffsetCfg(pos=(0.037, 0.0, 20.0)),  # X2 sole: x -0.070..0.144, y +-0.065')
rep('pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=[0.18, 0.06]),', 'pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=[0.18, 0.10]),')
rep('prim_path="{ENV_REGEX_NS}/Robot/torso_link",\n        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),', 'prim_path="{ENV_REGEX_NS}/Robot/pelvis",\n        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),')
rep('''            obs, _ = env.reset()
            start = distances()
            for _ in range(num_steps):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
''', '''            obs, _ = env.reset()
            start = distances()
            rec = [] if (os.environ.get("HIKING_OBS_CHECK") and not getattr(run, "recorded", False)) else None
            for _ in range(num_steps):
                actions = policy(obs)
                if rec is not None and len(rec) < 50:
                    rec.append(dict(obs=obs[0].cpu().numpy(), actions=actions[0].cpu().numpy(),
                                    root_pos_w=robot.data.root_pos_w[0].cpu().numpy(), root_quat_w=robot.data.root_quat_w[0].cpu().numpy(),
                                    root_lin_vel_b=robot.data.root_lin_vel_b[0].cpu().numpy(), root_ang_vel_b=robot.data.root_ang_vel_b[0].cpu().numpy(),
                                    projected_gravity_b=robot.data.projected_gravity_b[0].cpu().numpy(),
                                    joint_pos=robot.data.joint_pos[0].cpu().numpy(), joint_vel=robot.data.joint_vel[0].cpu().numpy(),
                                    command=base_env.command_manager.get_command("base_velocity")[0].cpu().numpy()))
                    if len(rec) == 50:
                        np.savez(os.environ["HIKING_OBS_CHECK"], joint_names=np.array(robot.joint_names),
                                 default_joint_pos=robot.data.default_joint_pos[0].cpu().numpy(),
                                 **{k: np.stack([r[k] for r in rec]) for k in rec[0]})
                        run.recorded = True
                        print("[eval] wrote obs check", os.environ["HIKING_OBS_CHECK"], flush=True)
                obs, _, dones, _ = env.step(actions)
''')
(T / "x2_port/eval_paper_x2.py").write_text(s)
print("wrote eval_paper_x2.py", len(s))
