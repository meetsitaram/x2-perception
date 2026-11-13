# Runs ON A NODE (piped over ssh stdin by plots.py; never copied there). Read-only, stdlib only, CPU only.
# argv[1] = base64(json {"globs": [...], "offsets": {path: byte_offset}}), paths relative to /mnt/filesystem-g4/tracks.
# Writes to stdout one gzip stream of framed text:
#   "##FILE <path> <old_offset> <new_offset> <size> <kind>\n" then the new data lines.
# jsonl files: raw new complete lines. tfevents files: one JSON row per step {"it": step, tag: value, ...} (scalars only).
import base64, glob, gzip, json, os, struct, sys

ROOT = "/mnt/filesystem-g4/tracks"
req = json.loads(base64.b64decode(sys.argv[1]).decode())
os.chdir(ROOT)
out = gzip.GzipFile(fileobj=sys.stdout.buffer, mode="wb", compresslevel=6)


# Train/time/* is logged against wall-clock seconds, not iterations; the per-term episode-length copies are redundant.
SKIP_PRE = ("Train/time/",)
SKIP_SUF = ("/max_episode_len_s", "/timestep")


def varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7F) << s; s += 7
        if c < 0x80:
            return r, i


def fields(b):
    i, n = 0, len(b)
    while i < n:
        k, i = varint(b, i); f, w = k >> 3, k & 7
        if w == 0:
            v, i = varint(b, i)
        elif w == 1:
            v = b[i:i + 8]; i += 8
        elif w == 5:
            v = b[i:i + 4]; i += 4
        elif w == 2:
            l, i = varint(b, i); v = b[i:i + l]; i += l
        else:
            return
        yield f, w, v


def tf_rows(d):
    """Parse complete records of a tfevents chunk; return (rows, bytes consumed)."""
    i = 0; rows = []; cur = None
    while i + 12 <= len(d):
        l = struct.unpack("<Q", d[i:i + 8])[0]
        if i + 12 + l + 4 > len(d):
            break
        rec = d[i + 12:i + 12 + l]; i += 12 + l + 4
        step = 0; summ = None
        try:
            for f, w, v in fields(rec):
                if f == 2 and w == 0:
                    step = v
                elif f == 5 and w == 2:
                    summ = v
        except Exception:
            continue
        if summ is None:
            continue
        for f, w, v in fields(summ):
            if f != 1 or w != 2:
                continue
            tag = val = None
            for g, ww, x in fields(v):
                if g == 1 and ww == 2:
                    tag = x.decode("utf-8", "replace")
                elif g == 2 and ww == 5:
                    val = struct.unpack("<f", x)[0]
            if tag is None or val is None or tag.startswith(SKIP_PRE) or tag.endswith(SKIP_SUF):
                continue
            if cur is None or cur["it"] != step:
                if cur is not None:
                    rows.append(cur)
                cur = {"it": step}
            cur[tag] = round(val, 6) if val == val and abs(val) != float("inf") else None
    if cur is not None:
        rows.append(cur)
    return rows, i


paths = []
for g in req["globs"]:
    paths += sorted(glob.glob(g))
for p in paths:
    try:
        size = os.path.getsize(p)
    except OSError:
        continue
    off = int(req["offsets"].get(p, 0))
    if off > size:          # file replaced or truncated: start again
        off = 0
    kind = "tf" if "tfevents" in p else "jsonl"
    if off == size:
        out.write(("##FILE %s %d %d %d %s\n" % (p, off, off, size, kind)).encode()); continue
    with open(p, "rb") as fh:
        fh.seek(off); d = fh.read(size - off)
    if kind == "jsonl":
        cut = d.rfind(b"\n") + 1
        body = d[:cut]
    else:
        rows, cut = tf_rows(d)
        body = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows).encode()
    out.write(("##FILE %s %d %d %d %s\n" % (p, off, off + cut, size, kind)).encode())
    out.write(body)
out.close()
