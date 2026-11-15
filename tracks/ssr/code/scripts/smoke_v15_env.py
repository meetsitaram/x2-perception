"""CPU unit smoke of the v15 env items: terrain set (discrete + free column), omni only after the top level, promotion at half
the tile, slope backward commands, no vy clamp, foothold base frame, route robots' home type. Prints PASS/FAIL lines, then DONE."""
import os
os.environ.setdefault("X2_TERRAIN_PAPER", "1")   # v15 terrain layout (code/ since 2026-11-13)
import math
import sys
import time

import torch

sys.path.insert(0, "/mnt/filesystem-g4/tracks/ssr/code")
from ssr import terrain as T
from ssr.env import SSREnv

torch.set_grad_enabled(False)
t0 = time.time()
ok_all = True


def check(name, cond, info=""):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, info, flush=True)


r = T.add_extended(T.generate(0), types=("tight_spots", "log_corridors", "homes", "street_front", "stairs", "wedges", "single_beams", "beam_fields", "blocks"),
                   stairs_inset=True, prop_routes=True, stations=1, slope_turns=2)
check("types", T.TYPES == ("ramp", "stairs", "beam", "platform", "discrete"), str(T.TYPES))
bn, tx, ty = T._i(T.BORDER), T._i(T.TILE_X), T._i(T.TILE_Y)
dz = r["z"][bn + 4 * tx: bn + 5 * tx, bn + 19 * ty: bn + 20 * ty]
check("discrete L19 blocks 0.05-0.20 m", 0.15 <= float(dz.max()) <= 0.2001 and float((dz > 0.04).mean()) > 0.05, "max %.3f cover %.2f" % (dz.max(), (dz > 0.04).mean()))
fz = r["z"][bn + 5 * tx: bn + 6 * tx, bn: bn + ty]
fv = r["vision"]["z"][bn + 5 * tx: bn + 6 * tx, bn: bn + ty]
check("free column flat, no walls", float(abs(fz).max()) == 0.0 and float(abs(fv).max()) == 0.0)
n = 40
env = SSREnv(n, "cpu", raster=r, tiles=True, train=True, camera="chest_rs", seed=5, ext_phys=24)
env.omni_top, env.promote_half, env.vy_clamp, env.slope_back_ban, env.q_noise, env.foot_base_frame = True, True, False, False, 0.0, True
env.back_scale, env.ext_frac, env.omni_frac = 0.5, 0.0, 0.3
env.home_type = env.type.clone()
ids = torch.arange(n)
env.ep[:] = 1
env.type[:] = ids % 5
env.level[:] = 19
env.ep[:] = 1; env.reset(ids); env.refresh(); env._contacts(); env._update_obs(ids)
check("no omni robots before anyone clears the top level (omni_frac ignored)", int(env.free.sum()) == 0, int(env.free.sum()))
# all clear the top level -> next episode on the free area, omnidirectional
env.curriculum(ids, torch.ones(n, dtype=torch.bool))
check("cleared top level -> random level, flagged", bool(env.grad.all()) and int(env.level.max()) < 20, env.level.tolist()[:8])
env.ep[:] = 1; env.reset(ids); env.refresh(); env._contacts(); env._update_obs(ids)
x_rel = env.qpos[:, 0] - env.origin[:, 0]
check("graduates are omni on the free column", bool(env.free.all()) and bool((env.origin[:, 0] == T.FREE * T.TILE_X).all()), env.origin[:3, 0].tolist())
c = env.cmd
check("Table 5 omni ranges (no turn / slow overrides)", float(c[:, 0].abs().max()) <= 1.0 and float(c[:, 1].abs().max()) <= 0.4 and float(c[:, 2].abs().max()) <= 1.2
      and float((c[:, 0].abs() > 0.4).float().mean()) > 0.3, "vx range %.2f..%.2f" % (c[:, 0].min(), c[:, 0].max()))
check("omni robots keep their terrain discriminator", bool((env.type == ids % 5).all()))
env.ep[:] = 1; env.reset(ids); env.refresh(); env._contacts(); env._update_obs(ids)
check("one omni episode, then back on terrain", int(env.free.sum()) == 0)
# promotion at half the tile
env.level[:] = 5
sx = env.spawn_xrel.clone()
fwd = env.speed >= 0
env.qpos[:, 0] = env.origin[:, 0] + torch.where(fwd, sx + 4.2, sx - 4.2)
env.ep[:] = 500
env.curriculum(ids, torch.zeros(n, dtype=torch.bool))
check("walked > 4 m without reaching the end -> level up", bool((env.level == 6).all()), env.level.tolist()[:8])
env.qpos[:, 0] = env.origin[:, 0] + torch.where(fwd, sx + 1.0, sx - 1.0)
env.speed[:] = torch.where(fwd, torch.full_like(env.speed, 1.0), torch.full_like(env.speed, -1.0))
env.curriculum(ids, torch.zeros(n, dtype=torch.bool))
check("walked 1 m of a 1 m/s x 10 s command -> level down", bool((env.level == 5).all()), env.level.tolist()[:8])
# slope commands: backward allowed, heading table
sl = ids[env.type == T.SLOPE]
neg = 0
for _ in range(20):
    env._command_reset(sl)
    neg += int((env.speed[sl] < 0).sum())
check("slope backward commands drawn (back_scale 0.5, no ban)", neg > 0, neg)
hm = {}
for k in range(5):
    tt = ids[env.type == k]
    mx = 0.0
    for _ in range(30):
        env._command_reset(tt)
        mx = max(mx, float(env.heading[tt].abs().max()))
    hm[T.TYPES[k]] = round(math.degrees(mx))
check("heading ranges 80/30/15/15/30", hm["ramp"] > 70 and 25 < hm["stairs"] <= 30 and hm["beam"] <= 15 and hm["platform"] <= 15 and 25 < hm["discrete"] <= 30, hm)
# vy clamp off
env.free[:] = False
env.speed[:] = 1.0
env.heading[:] = env.s["yaw"] + 1.4
env._update_cmd()
check("projected vy not clamped", float(env.cmd[:, 1].abs().max()) > 0.6, round(float(env.cmd[:, 1].abs().max()), 2))
# foothold support in the base frame: on flat ground a level base gives the same world point as the heading frame
mu, sg = torch.zeros(n, 2, 2), torch.full((n, 2), 0.05)
a_ = env.support_imagined(mu, sg)
env.foot_base_frame = False
b_ = env.support_imagined(mu, sg)
check("foothold support: base frame == heading frame at the base origin", bool(torch.allclose(a_, b_)))
# route robots: home type restored
env.foot_base_frame = True
env.ext_frac = 1.0
env.type[:] = ids % 5
env.home_type[:] = env.type
env.ep[:] = 1
env.reset(ids)
ext_ids = ids[env.ext]
env.ext_frac = 0.0
env.ep[:] = 1
env.reset(ids)
check("route robots go back to their home tile type", bool((env.type == ids % 5).all()) and len(ext_ids) > 0, "routes %d" % len(ext_ids))
# a short rollout with zero actions: nothing explodes
for t in range(25):
    out = env.step(torch.zeros(n, 21), torch.zeros(n, 2, 2), torch.full((n, 2), 0.05))
check("rollout finite", bool(torch.isfinite(out["r_l"]).all()) and bool(torch.isfinite(env.qpos).all()))
print("sim %.0f s" % (time.time() - t0))
print("DONE" if ok_all else "FAILED")
