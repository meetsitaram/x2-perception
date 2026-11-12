"""--disc_mask_arms (2026-11-12 03:25, after the port audit; deviation): the style discriminators no longer see the arms. In every
63-d AMP frame the 8 arm joint positions (cols 22-29), their velocities (43-50) and the two wrist positions in the base frame
(51-56) are set to 0 for reference and policy samples alike, at the discriminator input. Reason: in the reference the shoulder
roll sits past the X2's hard limit in 62-66 % of moving frames and the wrists are inside the hip/thigh links in 46 % of frames, so
the arms were a style cue the robot can only copy by pressing into its limits and its own legs (seen in r3: both shoulder rolls
on the limit 94 % of the time). The network size is unchanged, so checkpoints resume. Default off. Apply after patch_air.py."""
import pathlib
R = pathlib.Path(__file__).resolve().parents[1]


def sub(path, old, new, count=1):
    p = R / path
    s = p.read_text()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    p.write_text(s.replace(old, new))


N = "ssr/networks.py"
sub(N, "        self.norm = RunningNorm(AMP_DIM * AMP_FRAMES)\n\n    def forward(self, x, i):\n        return self.nets[i](self.norm(x))",
    "        self.norm = RunningNorm(AMP_DIM * AMP_FRAMES)\n        self.mask = None                        # optional fixed input mask (not saved; set by train --disc_mask_arms)\n\n"
    "    def forward(self, x, i):\n        if self.mask is not None:\n            x = x * self.mask\n        return self.nets[i](self.norm(x))")
Tr = "ssr/train.py"
sub(Tr, "    lr_ppo = 5e-4\n",
    "    if a.disc_mask_arms:\n"
    "        mk = torch.ones(AMP_DIM, device=dev)\n"
    "        mk[22:30], mk[43:51], mk[51:57] = 0.0, 0.0, 0.0           # arm q, arm qd, wrist positions\n"
    "        disc.mask = mk.repeat(AMP_FRAMES)\n"
    "    lr_ppo = 5e-4\n")
sub(Tr, "    ap.add_argument(\"--style_from\",",
    "    ap.add_argument(\"--disc_mask_arms\", action=\"store_true\", help=\"deviation: the style discriminators do not see arm joints or wrist positions\")\n"
    "    ap.add_argument(\"--style_from\",")
print("patched")
