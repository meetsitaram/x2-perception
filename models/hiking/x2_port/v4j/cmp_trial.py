"""Compare v4k trial iterations (after byte OFF) with the last 32 v4j iterations before OFF. Prints selected keys."""
import re, sys, collections
LOG = "/mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/_supervisor/train.log"
OFF = 95100216
pat = re.compile(r"^\s*(?:Mean episode )?([A-Za-z][^:]{2,120}):\s+(-?[0-9.]+(?:e[-+]?\d+)?)\s*$")
it_pat = re.compile(r"Learning iteration (\d+)/")

def blocks(text):
    out, cur, it = [], None, None
    for line in text.splitlines():
        m = it_pat.search(line)
        if m:
            if cur is not None: out.append((it, cur))
            it, cur = int(m.group(1)), {}
            continue
        if cur is not None:
            m = pat.match(line)
            if m:
                try: cur[m.group(1).strip()] = float(m.group(2))
                except ValueError: pass
    if cur is not None: out.append((it, cur))
    return out

with open(LOG, "rb") as f:
    f.seek(OFF - 60_000_000 if OFF > 60_000_000 else 0)
    before = f.read(OFF - f.tell()).decode("utf8", "replace")
    after = f.read(20_000_000).decode("utf8", "replace")
B = blocks(before)[-32:]
A = []
for it, d in blocks(after):
    if A and it < A[-1][0]: break  # the fall-back restarts the counter
    A.append((it, d))
print("v4j its", B[0][0], "-", B[-1][0], "| v4k its", A[0][0], "-", A[-1][0], len(A))
def mean(bl, k):
    v = [d[k] for _, d in bl if k in d]
    return sum(v) / len(v) if v else float("nan")
def first_last(bl, k, n=8):
    v = [d[k] for _, d in bl if k in d]
    return (sum(v[:n]) / max(1, len(v[:n])), sum(v[-n:]) / max(1, len(v[-n:]))) if v else (float("nan"),) * 2
keys = sorted(set(k for _, d in A for k in d) | set(k for _, d in B for k in d))
sel = re.compile(sys.argv[1] if len(sys.argv) > 1 else ".")
for k in keys:
    if not sel.search(k): continue
    a, b = mean(A, k), mean(B, k)
    f, l = first_last(A, k)
    print("%-72s v4j %10.4f | v4k %10.4f (first8 %9.4f last8 %9.4f)" % (k[-72:], b, a, f, l))
