"""t_f patch (coordinator, 2026-11-12; not in the paper): style weight held at --style_low x w_style until mean touchdown air time
exceeds --gate_air for 5 iterations (the same latch as the progress gate), then ramped to full over --gate_ramp iterations.
Default off (--style_low -1). Apply after patch_te.py: python scripts/patch_tf.py (from tracks/ssr/code)."""
import pathlib
R = pathlib.Path(__file__).resolve().parents[1]


def sub(path, old, new, count=1):
    p = R / path
    s = p.read_text()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    p.write_text(s.replace(old, new))


Tr = "ssr/train.py"
sub(Tr, "        w[2] = a.w_style * ramp\n",
    "        w[2] = a.w_style * ramp\n"
    "        if a.style_low >= 0:                         # t_f: style held low until real swing appears, then ramped to full\n"
    "            w[2] = w[2] * (a.style_low if gate_start is None else min(1.0, a.style_low + (1.0 - a.style_low) * (it - gate_start) / max(a.gate_ramp, 1)))\n")
sub(Tr, "        if a.prog_gate > 0 and gate_start is None and gate_hits >= 5:\n",
    "        if (a.prog_gate > 0 or a.style_low >= 0) and gate_start is None and gate_hits >= 5:\n")
sub(Tr, "prog_gate=round(env.prog_gate, 3), gate_start=gate_start,",
    "prog_gate=round(env.prog_gate, 3), gate_start=gate_start, w_style_now=round(float(w[2]), 4),")
sub(Tr, "    ap.add_argument(\"--prog_gate\",",
    "    ap.add_argument(\"--style_low\", type=float, default=-1.0, help=\"t_f (deviation): style weight x this until touchdown air time > --gate_air for 5 iterations, then ramped to full over --gate_ramp; -1 = off\")\n"
    "    ap.add_argument(\"--prog_gate\",")
print("patched")
