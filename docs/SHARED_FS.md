# Shared filesystem layout

Mounted at `/mnt/filesystem-g4` on all three nodes (500 GB, confirmed 2026-11-11). Log in as the login in `config/nodes.local.json` with the desktop key `~/.ssh/nebius_corpus`.

```
/mnt/filesystem-g4/
  data/                  read-only inputs, changed only through the coordinator
    motions/             human motion: Bones-SEED X2 retargets, Pico captures (pico_tapes, terrain-park), the Hiking release file
    apparatus/           CAD height rasters and wall boxes (from ssr-x2/assets/route), route waypoints
    terrains/            terrain generators and generated fields (from hiking-in-the-wild/review/new_terrains once ported)
    robots/              X2 URDF/MJCF/USD
  tracks/
    acp/     code/  env/  runs/  eval/     one tree per track; only that track's agent writes here
    ssr/     code/  env/  runs/  eval/
    hiking/  code/  env/  runs/  eval/
  exchange/              copies handed from one track to another, each with a README naming source, date and commit
  eval/                  the common eval: code, results per track and checkpoint, renders for the user's review
  status/                heartbeats: <track>.json written by each supervisor; eval.json by the eval monitor
```

Copying from the desktop: with `scp`/`rsync` under a rate cap (`--bwlimit`), into `/mnt/filesystem-g4/data/...`, recorded in
`state/STATUS.md` (what, from where, size). Motion data keeps its licence terms: Bones Studio credit on anything shown; no
dataset clip names in renders.
