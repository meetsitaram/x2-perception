"""Plant selection (mc_20261024 by default), per-camera optics, the wide torso camera. Run in tracks/ssr/code."""
import pathlib


def sub(p, pairs):
    f = pathlib.Path(p)
    s = f.read_text()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:70], s.count(a))
        s = s.replace(a, b)
    f.write_text(s)


sub("ssr/x2.py", [
 ('    for n_ in ("waist_pitch_joint", "waist_roll_joint"):\n        i_ = names.index(n_)\n        kp[i_], kd[i_] = 120.0, 6.0\n',
  '    if not plant.stiffness:                                              # armature-rule plants only; a plant with measured gains (mc_20261024) is used as it is\n'
  '        for n_ in ("waist_pitch_joint", "waist_roll_joint"):\n            i_ = names.index(n_)\n            kp[i_], kd[i_] = 120.0, 6.0\n'),
 ('    "chest85": dict(body="torso_link", pos=(0.155, 0.0, 0.20), pitch=float(np.deg2rad(85.0))),\n',
  '    "chest85": dict(body="torso_link", pos=(0.155, 0.0, 0.20), pitch=float(np.deg2rad(85.0))),\n'
  '    # User spec 2026-11-11: see the ground from about 10 cm behind the toe tips to 2 m ahead, feet in view, range 3 m. Mount point PROVISIONAL (the G1 torso camera\n'
  '    # offset of the Hiking track put on the X2 torso link, y centred) until the Hiking track fixes the X2 offset; fov = square field of view of the 36 x 36 image.\n'
  '    "torso58": dict(body="torso_link", pos=(0.0488, 0.0, 0.4378), pitch=float(np.deg2rad(58.5)), fov=float(np.deg2rad(62.0)), dmax=3.0, march=38),\n'),
])
sub("ssr/env.py", [
 ('        cam = X.CAMERAS[self.camera]\n',
  '        cam = X.CAMERAS[self.camera]\n'
  '        self.cam_fov0, self.depth_max, self.march = cam.get("fov", X.CAM_FOV), cam.get("dmax", DEPTH_MAX), cam.get("march", 26)\n'),
 ('        rays = T.camera_rays(DEPTH_N, X.CAM_FOV, dev)\n', '        rays = T.camera_rays(DEPTH_N, self.cam_fov0, dev)\n'),
 ('        self.half_fov = math.tan(X.CAM_FOV / 2)\n', '        self.half_fov = math.tan(self.cam_fov0 / 2)\n'),
 ('        sc = torch.stack((e, 1 + self.cam_fov[:, 0] / X.CAM_FOV, 1 + self.cam_fov[:, 1] / X.CAM_FOV), -1)\n',
  '        sc = torch.stack((e, 1 + self.cam_fov[:, 0] / self.cam_fov0, 1 + self.cam_fov[:, 1] / self.cam_fov0), -1)\n'),
 ('        tmax = DEPTH_MAX / 0.4\n        t = self.hmap.depth(o, d, tmax=tmax)\n',
  '        tmax = self.depth_max / 0.4\n        t = self.hmap.depth(o, d, tmax=tmax, iters=self.march)\n'),
 ('        self.depth = ((z_depth.clamp(DEPTH_MIN, DEPTH_MAX) - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)).reshape(-1, DEPTH_N, DEPTH_N)\n',
  '        self.depth = ((z_depth.clamp(DEPTH_MIN, self.depth_max) - DEPTH_MIN) / (self.depth_max - DEPTH_MIN)).reshape(-1, DEPTH_N, DEPTH_N)\n'),
])
sub("ssr/train.py", [
 ('    dev = a.device\n', '    dev = a.device\n    os.environ["X2_PLANT"] = a.plant                 # read by ssr.plant_config.load_plant when the model is built\n'),
 ('    ap.add_argument("--camera", default="head", choices=("head", "chest", "chest62", "chest85"))\n',
  '    ap.add_argument("--camera", default="head", choices=("head", "chest", "chest62", "chest85", "torso58"))\n'
  '    ap.add_argument("--plant", default="mc_20261024", help="user decision 2026-11-11: the measured motion-controller gains; vendor_20260823 = runs r1-r4, d1, d2")\n'),
])
print("patched3")
