"""Contact labels for any SONIC-format clip set on its own props: the nine-body hindsight contacts
(pipeline.contacts.label_clip, ACP's labels), exactly as pipeline.chain_contacts does for the chains.

    python -m pipeline.clip_contacts build/train/pico_2026-10-29_2x2_train.pkl build/train/pico_2026-10-29_2x2_test.pkl \
        --props build/train/pico_2026-10-29_2x2_props.json --out build/train/pico_2026-10-29_2x2_contacts.pkl

Clips without props are labelled on the flat floor. Writes one sidecar for all the given clip files (clip -> next,
next_dt, occ, contact; the chains' contact format).

Run (from sim).
"""
import argparse
import json

import joblib
import numpy as np

from pipeline import clipdb as C
from pipeline import contacts as K
from pipeline.chain_contacts import qpos_of, terrain_of


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('clips', nargs='+')
    ap.add_argument('--props', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    props = json.load(open(a.props)) if a.props else {}
    bp = K.BodyPoints()
    out, n, con, occ = {}, 0, 0, 0
    for path in a.clips:
        for name, clip in joblib.load(path).items():
            assert clip['fps'] == 50, clip['fps']
            course = terrain_of(props[name]) if props.get(name) else None
            lab = K.label_clip(C.Clip(name, 'clip', qpos_of(clip)), bp, course)
            out[name] = dict(next=lab['next'].astype(np.float32), next_dt=lab['next_dt'].astype(np.float32),
                             occ=lab['occ'], contact=lab['contact'])
            n += len(lab['occ'])
            con = con + lab['contact'].sum(0)
            occ = occ + lab['occ'].sum(0)
    joblib.dump(out, a.out)
    print('%d clips, %d frames -> %s' % (len(out), n, a.out))
    for b, name in enumerate(K.BODIES):
        print('  %-12s in contact %5.1f%%  contact coming %5.1f%%' % (name, 100 * con[b] / n, 100 * occ[b] / n))


if __name__ == '__main__':
    main()
