"""Prior windows for slow walking and turning (2026-11-12): --prior_turn_min W keeps windows whose mean |yaw rate| >= W even
when their planar speed is under --prior_min_speed. Default 0 = as before. Backups ../_bak_20261112/*.pre_priorturn"""
import pathlib
import shutil

C = pathlib.Path(__file__).resolve().parents[1]
p = C / "ssr/train.py"
s = p.read_text()
shutil.copy(p, C.parent / "_bak_20261112/train.py.pre_priorturn")
rep = [
    ("    def __init__(self, device, min_speed=0.0, file=\"data/amp.npz\", balance=False, q_shift=None):\n",
     "    def __init__(self, device, min_speed=0.0, file=\"data/amp.npz\", balance=False, q_shift=None, turn_min=0.0):\n"),
    ("                ok &= win >= min_speed\n",
     "                keep = win >= min_speed\n"
     "                if turn_min > 0:                            # deviation (2026-11-12): turning windows (|yaw rate| >= turn_min) stay even when slow\n"
     "                    wz_ = np.convolve(np.abs(z[name][:, 8]), np.ones(AMP_FRAMES) / AMP_FRAMES)[AMP_FRAMES - 1:]\n"
     "                    wz_ = np.concatenate((wz_, np.zeros(n - len(wz_))))[:n]\n"
     "                    keep |= wz_ >= turn_min\n"
     "                ok &= keep\n"),
    ("    amp = AMPData(dev, a.prior_min_speed, a.amp_file, a.prior_balance, q_shift=_shift)\n",
     "    amp = AMPData(dev, a.prior_min_speed, a.amp_file, a.prior_balance, q_shift=_shift, turn_min=a.prior_turn_min)\n"),
    ("    ap.add_argument(\"--edge_stop\",",
     "    ap.add_argument(\"--prior_turn_min\", type=float, default=0.0, help=\"deviation: keep prior windows with mean |yaw rate| >= this even under --prior_min_speed; 0 = off\")\n"
     "    ap.add_argument(\"--edge_stop\","),
]
for o, n in rep:
    assert s.count(o) == 1, o
    s = s.replace(o, n)
p.write_text(s)
print("patched")
