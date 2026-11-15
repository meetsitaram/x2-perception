#!/usr/bin/env python3
"""Run on the hiking node: fix continue_v8.sh (ENVV used $T/$S before they were set), build v8b (v8 + fewer non-walking
commands), chain v8 -> v8b -> v7. Writes only into code/x2_port/v8/ (old copies go to _trash/<date>/)."""
import os, re, shutil, time
T = "/mnt/filesystem-g4/tracks/hiking"; X = T + "/code/x2_port"; V = X + "/v8"
TR = T + "/_trash/" + time.strftime("%F"); os.makedirs(TR, exist_ok=True)
stamp = time.strftime("%H%M%S")
shutil.copy(V + "/continue_v8.sh", TR + "/continue_v8.sh.envbug_" + stamp)
with open(TR + "/NOTES.txt", "a") as f:
    f.write(time.strftime("%F %T") + " %s/continue_v8.sh.envbug_%s: continue_v8.sh before the ENVV-order fix\n" % (TR, stamp))

src = open(V + "/continue_v8.sh").read()
DEF = "T=/mnt/filesystem-g4/tracks/hiking; S=$T/runs/x2_parkour_v4/_supervisor   # defined BEFORE ENVV (bug of 11-14 08:50)\n"
PATHCHECK = r'''# ENVV paths must exist (11-14 08:50: ENVV used $T/$S before they were set -> /data/extended_v4k, check failed)
for kv in $ENVV; do case "$kv" in HIKING_EXTENDED_TERRAINS=*|HIKING_X2_MOTION_DIR=*) [ -d "${kv#*=}" ] || { echo "$(date -u '+%F %T') @TAG@ NOT started: ENVV path ${kv#*=} missing" >> $S/launch.txt; exit 1; };; esac; done
'''
assert src.count('\nENVV="') == 1 and DEF not in src
src = src.replace('\nENVV="', "\n" + DEF + 'ENVV="', 1)
# path check right after the line that defines NAME/RUN/S
src = src.replace("TR=$T/_trash/$(date +%F); mkdir -p $TR\n", "TR=$T/_trash/$(date +%F); mkdir -p $TR\n" + PATHCHECK, 1)

# v8 only: any exit after GO without "is training" arms v8b (user 11-14: try fewer turn/slow episodes before v7)
TRAP = '''V8_TRAINING=0
v8b_next() { [ $V8_TRAINING = 1 ] && return; echo now > $S/GO_v8b; say "did not pass -> GO_v8b written (v8b next, then v7)"; }
trap v8b_next EXIT
'''
v8 = src.replace('say "GO seen; restart from model_$WAIT_FOR"\n', 'say "GO seen; restart from model_$WAIT_FOR"\n' + TRAP, 1)
v8 = v8.replace('  say "is training, GPU idle', '  V8_TRAINING=1\n  say "is training, GPU idle', 1)
v8 = v8.replace("@TAG@", "v8")
assert "trap v8b_next EXIT" in v8 and "V8_TRAINING=1" in v8
open(V + "/continue_v8.sh", "w").write(v8)

# v8b = fixed v8 with the three shares lowered and its own names; no trap (its failure -> v7, final)
b = src.replace("@TAG@", "v8b")
for a, c in [("HIKING_X2_SLOW_FWD_SHARE=0.25", "HIKING_X2_SLOW_FWD_SHARE=0.12"),
             (" HIKING_X2_TURN_SHARE=0.15", " HIKING_X2_TURN_SHARE=0.10"),
             ("HIKING_X2_STAND_SHARE=0.10", "HIKING_X2_STAND_SHARE=0.05")]:
    assert b.count(a) == 1, a
    b = b.replace(a, c)
reps = [("$S/GO_v8", "$S/GO_v8b"), (' v8 $*"', ' v8b $*"'), ("_check_v8_", "_check_v8b_"),
        ("terrain_levels_prestop_v8.pt", "terrain_levels_prestop_v8b.pt"), ("precheck_v8.log", "precheck_v8b.log"),
        ("$X/v8/precheck_v8.sh $X/v8/continue_v8.sh", "$X/v8/precheck_v8b.sh $X/v8/continue_v8b.sh"),
        ("$X/v8/cfgscan_v8.sh $S/cfgscan_v8.log", "$X/v8/cfgscan_v8b.sh $S/cfgscan_v8b.log"),
        ("test_fair_v8.log", "test_fair_v8b.log"), ("start_x2_parkour_v8.sh", "start_x2_parkour_v8b.sh"),
        ("(v8):", "(v8b):"), ("guard8 \"bash $X/v8/crashguard_v8.sh\"", "guard8b \"bash $X/v8/crashguard_v8b.sh\""),
        ("v7_files_v8_", "v7_files_v8b_"), ("used by continue_v8.sh", "used by continue_v8b.sh"),
        ("failed v8 continuation", "failed v8b continuation"),
        ("STOP_${NAME}", "STOP_${NAME}"),
        ("#continue_v8.sh: v7 + paper curriculum signal + touchdown pay for progress#",
         "#continue_v8b.sh: v8 + fewer non-walking commands (stand 0.05, turn 0.10, slow fwd 0.12)#")]
for a, c in reps:
    assert a in b, a
    b = b.replace(a, c)
# the generated start line must carry the lowered shares too
b = b.replace('s#^tmux new -d -s hiking_x2 \\"#tmux new -d -s hiking_x2 \\"HIKING_X2_FAIR=1 HIKING_X2_AIR_TD_PROGRESS=1 #;',
              's#^tmux new -d -s hiking_x2 \\"#tmux new -d -s hiking_x2 \\"HIKING_X2_FAIR=1 HIKING_X2_AIR_TD_PROGRESS=1 #; '
              's#HIKING_X2_SLOW_FWD_SHARE=0.25#HIKING_X2_SLOW_FWD_SHARE=0.12#; s# HIKING_X2_TURN_SHARE=0.15# HIKING_X2_TURN_SHARE=0.10#; '
              's#HIKING_X2_STAND_SHARE=0.10#HIKING_X2_STAND_SHARE=0.05#;', 1)
assert "SLOW_FWD_SHARE=0.12#" in b
b = b.replace("# continue_v8.sh (2026-11-14", "# continue_v8b.sh (2026-11-14 09:20: v8 + stand 0.10->0.05, turn 0.15->0.10, slow fwd 0.25->0.12;"
              " armed by continue_v8.sh when v8 does not pass; failure -> v7). Based on continue_v8.sh (2026-11-14", 1)
open(V + "/continue_v8b.sh", "w").write(b)

# cfgscan / precheck / crashguard for v8b
c = open(V + "/cfgscan_v8.sh").read().replace("$X/v8/continue_v8.sh", "$X/v8/continue_v8b.sh")
open(V + "/cfgscan_v8b.sh", "w").write(c)
g = open(V + "/crashguard_v8.sh").read().replace("start_x2_parkour_v8.sh", "start_x2_parkour_v8b.sh").replace("crashguard_v8", "crashguard_v8b")
open(V + "/crashguard_v8b.sh", "w").write(g)
p = open(V + "/precheck_v8.sh").read()
p = p.replace("\\['HIKING_X2_AIR_TD_PROGRESS', 'HIKING_X2_FAIR'\\]", "@LIST@")
p = p.replace('grep -q "^ENVV=\\"HIKING_X2_FAIR=1 HIKING_X2_AIR_TD_PROGRESS=1 " "$1" || OUT="$OUT FAIL ENVV lacks the v8 vars"',
              'grep -q "^ENVV=\\"HIKING_X2_FAIR=1 HIKING_X2_AIR_TD_PROGRESS=1 " "$1" || OUT="$OUT FAIL ENVV lacks the v8 vars"\n'
              'for kv in HIKING_X2_STAND_SHARE=0.05 HIKING_X2_TURN_SHARE=0.10 HIKING_X2_SLOW_FWD_SHARE=0.12 HIKING_X2_SLOW_TURN_SHARE=0.4; do '
              'grep "^ENVV=" "$1" | grep -q " $kv " || OUT="$OUT FAIL ENVV lacks $kv"; done')
p = p.replace("PRECHECK_V8 PASS", "PRECHECK_V8B PASS").replace("# precheck_v8.sh <continue_v8.sh>", "# precheck_v8b.sh <continue_v8b.sh> (= precheck_v8.sh + the three v8b shares)")
open(V + "/precheck_v8b.sh", "w").write(p)
print("setup ok")
