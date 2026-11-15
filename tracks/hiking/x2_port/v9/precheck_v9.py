"""precheck_v9.py <continue_v9.sh> (2026-11-14): v9's launcher against v8's live start line, variable by variable.
Accepted differences, and only these (user decisions "Hiking v9 decisions"): the paper command mix (stand 0.05, turn 0, slow fwd 0,
slow yaw 0), option t removed (OPTS without t, AIR_TD_PROGRESS 0, AIR_TD_WEIGHT 0), the v9 switches (V9 1, PAPER_STAND 1) and the
contact scales at their neutral values (1.0, scope paper). Also: live staged.py = variants/staged.py.v6, live fair.py = v8/fair.py,
live extended.py = v8/extended.py.v8 (v8 installed, nothing else), v9/extended.py.v9 = v8's + the 3-line hook only.
Prints PRECHECK_V9 PASS; exit 0 = pass."""
import filecmp, difflib, re, sys
X = "/mnt/filesystem-g4/tracks/hiking/code/x2_port"; T = "/mnt/filesystem-g4/tracks/hiking"; S = T + "/runs/x2_parkour_v4/_supervisor"
D = X + "/files_instinctlab/tasks/parkour/config/x2"
fail = []
src = open(sys.argv[1]).read()
envv = re.search(r'^ENVV="([^"]*)"', src, re.M).group(1).replace("$T", T).replace("$S", S)
v9 = dict(kv.split("=", 1) for kv in envv.split())
st8 = open(X + "/start_x2_parkour_v8.sh").read()
line = re.search(r'tmux new -d -s hiking_x2 "(.*?) \$HOME/hiking-env', st8).group(1)
v8 = dict(kv.split("=", 1) for kv in line.split())
want = {"HIKING_X2_V9": "1", "HIKING_X2_PAPER_STAND": "1", "HIKING_X2_STUMBLE_SCALE": "1.0", "HIKING_X2_CONTACT_SCALE": "1.0",
        "HIKING_X2_SCALE_SCOPE": "paper", "HIKING_X2_AIR_TD_PROGRESS": "0", "HIKING_X2_OPTS": "a,c,d,f,r,s,k,y,o,z,w,g,v,j",
        "HIKING_X2_SLOW_FWD_SHARE": "0", "HIKING_X2_SLOW_TURN_SHARE": "0", "HIKING_X2_AIR_TD_WEIGHT": "0", "HIKING_X2_TURN_SHARE": "0",
        "HIKING_X2_STAND_SHARE": "0.05"}
for k in sorted(set(v8) | set(v9)):
    if k in want:
        if v9.get(k) != want[k]:
            fail.append("%s is %s, want %s" % (k, v9.get(k), want[k]))
    elif v8.get(k) != v9.get(k):
        fail.append("%s differs from v8: v8 %s, v9 %s" % (k, v8.get(k), v9.get(k)))
if set(v8["HIKING_X2_OPTS"].split(",")) - set(v9.get("HIKING_X2_OPTS", "").split(",")) != {"t"}:
    fail.append("OPTS: only t may be removed")
for a, b in ((D + "/staged.py", X + "/variants/staged.py.v6"), (D + "/fair.py", X + "/v8/fair.py"), (D + "/extended.py", X + "/v8/extended.py.v8")):
    if not filecmp.cmp(a, b, shallow=False):
        fail.append("%s != %s" % (a, b))
d = [l for l in difflib.unified_diff(open(X + "/v8/extended.py.v8").read().splitlines(), open(X + "/v9/extended.py.v9").read().splitlines(), lineterm="", n=0)
     if l[:1] in "+-" and not l.startswith(("+++", "---"))]
if len(d) != 3 or not all(l.startswith("+") for l in d) or "apply_v9" not in "".join(d):
    fail.append("extended.py.v9 is not v8's + the 3-line hook: %s" % d)
print("PRECHECK_V9 PASS" if not fail else "\n".join("FAIL " + f for f in fail))
sys.exit(1 if fail else 0)
