"""--flat_until N (coordinator, 2026-11-12 02:50; deviation): every robot on the open flat tile until iteration N, then the terrain
types are drawn again (uniform over the five), all levels start at 0, every robot is reset and the terrain curriculum runs.
A run resumed after the switch (types already mixed) keeps its loaded levels. Default 0 = off. Apply after patch_tf.py."""
import pathlib
R = pathlib.Path(__file__).resolve().parents[1]


def sub(path, old, new, count=1):
    p = R / path
    s = p.read_text()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    p.write_text(s.replace(old, new))


Tr = "ssr/train.py"
sub(Tr, "    env.flat_only = a.flat_only\n", "    env.flat_only = a.flat_only or a.flat_until > 0\n")
sub(Tr, "    if a.flat_only:\n        env.reset(torch.arange(a.num_envs, device=dev))",
    "    if a.flat_only or a.flat_until > 0:\n        env.reset(torch.arange(a.num_envs, device=dev))")
sub(Tr, "        print(\"resumed\", a.resume, start, flush=True)\n",
    "        print(\"resumed\", a.resume, start, flush=True)\n"
    "    if a.flat_until > 0:\n"
    "        env.flat_only = not (start >= a.flat_until and bool((env.type != T.FLAT).any()))   # resumed after the switch: keep the loaded levels\n")
sub(Tr, "        env.boot = max(0.0, 1.0 - it / a.boot_iters)",
    "        if a.flat_until > 0 and env.flat_only and it >= a.flat_until:      # flat phase over: terrain curriculum from level 0\n"
    "            env.flat_only = False\n"
    "            allr = torch.arange(N_, device=dev)\n"
    "            env.type[:] = torch.randint(0, 5, (N_,), device=dev)\n"
    "            env.level[:] = 0\n"
    "            env.reset(allr)\n"
    "            env.refresh()\n"
    "            env._contacts()\n"
    "            env._update_obs(allr)\n"
    "            h = h * 0.0\n"
    "            ep_r.zero_(); ep_len.zero_()\n"
    "            print(\"flat phase over at\", it, \"; terrain curriculum on, types\", torch.bincount(env.type, minlength=5).tolist(), flush=True)\n"
    "        env.boot = max(0.0, 1.0 - it / a.boot_iters)")
sub(Tr, "    ap.add_argument(\"--style_low\",",
    "    ap.add_argument(\"--flat_until\", type=int, default=0, help=\"deviation: all robots on the flat tile until this iteration, then terrain types redrawn at level 0 and the curriculum on; 0 = off\")\n"
    "    ap.add_argument(\"--style_low\",")
print("patched")
# applied afterwards (02:52): env.ep[:] = 1 before the switch reset, so reset() keeps the drawn types and level 0 on a fresh resume
