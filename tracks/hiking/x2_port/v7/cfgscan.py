# v6 copy of v5b/cfgscan.py + reward listing (2026-11-13)
"""CPU scan of the FINAL env cfg (2026-11-13, after v5 and v5b died in Isaac Lab's term check): builds the task's env cfg with the
launcher's env vars, with the simulator-only modules (omni, pxr, carb, isaacsim, ...) stubbed, then applies Isaac Lab's own
signature rule (manager_base._resolve_common_term_cfg) to EVERY term of every manager: observations (all groups), rewards,
terminations, events, curriculum. No GPU.
usage: HIKING_X2_...=... python cfgscan.py <module:Class>    exit 1 on any mismatch."""
import importlib, importlib.abc, importlib.machinery, inspect, os, sys, types

STUB = ("omni", "pxr", "carb", "isaacsim", "usdrt", "warp", "nvidia.srl", "Semantics", "pinocchio", "isaaclab_contrib", "isaaclab_rl", "isaaclab_mimic")


class _Stub:
    """Stand-in for any simulator object: attribute access, calls, subclassing, indexing, `|` in annotations."""
    def __init__(self, name="stub"):
        object.__setattr__(self, "__name__", name); object.__setattr__(self, "__qualname__", name)

    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        return _Stub(f"{self.__name__}.{k}")

    def __call__(self, *a, **k):
        return _Stub(self.__name__ + "()")

    def __mro_entries__(self, bases):
        return (object,)

    def __getitem__(self, k):
        return self

    def __or__(self, o):
        return self

    __ror__ = __or__

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class _StubModule(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        return _Stub(f"{self.__name__}.{k}")


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in STUB or name in STUB:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _StubModule(spec.name); m.__path__ = []; m.__file__ = "<stub>"
        return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Finder())
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
modname, cls = sys.argv[1].split(":")
cfg = getattr(importlib.import_module(modname), cls)()

MIN = {"observations": 1, "rewards": 1, "terminations": 1, "events": 2, "curriculum": 2}


def check(name, t, min_argc):
    f = t.func
    if isinstance(f, str):
        mod, fn = f.split(":"); f = getattr(importlib.import_module(mod), fn)
    if inspect.isclass(f):
        f, min_argc = f.__call__, min_argc + 1
    params = list((getattr(t, "params", None) or {}).keys())
    a = inspect.signature(f).parameters
    wd = [x for x in a if a[x].default is not inspect.Parameter.empty]
    nd = [x for x in a if a[x].default is inspect.Parameter.empty]
    args = nd + wd
    if len(args) > min_argc and set(args[min_argc:]) != set(params + wd):
        return f"{name}: func {getattr(f, '__qualname__', f)} expects mandatory {nd[min_argc:]}, optional {wd}, given {params}"
    return None


def terms(obj):
    for k in dir(obj):
        if k.startswith("_"):
            continue
        v = getattr(obj, k, None)
        if v is not None and hasattr(v, "func") and not isinstance(v, (types.FunctionType, type)):
            yield k, v


bad, n = [], 0
for mgr, mn in MIN.items():
    root = getattr(cfg, mgr, None)
    if root is None:
        continue
    groups = [(g, v) for g, v in vars(root).items() if not g.startswith("_") and v is not None and not hasattr(v, "func")] \
        if mgr == "observations" else [("", root)]
    for g, obj in groups:
        for k, t in terms(obj):
            n += 1
            r = check(f"{mgr}/{g + '/' if g else ''}{k}", t, mn)
            if r:
                bad.append(r)
for _k, _t in [kv for g in [cfg.rewards] + [getattr(cfg.rewards, a) for a in dir(cfg.rewards) if not a.startswith("_") and not callable(getattr(cfg.rewards, a))] for kv in terms(g)]:
    if hasattr(_t, "func"):
        print("reward", _k, getattr(_t.func, "__name__", _t.func), float(_t.weight))
print(f"cfgscan: {n} terms checked in {sorted(MIN)}")
for b in bad:
    print("FAIL " + b)
sys.exit(1 if bad or n < 20 else 0)
