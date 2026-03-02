"""Memory system registry. Drop a .py file here to add a system."""

import importlib
import pkgutil

SYSTEMS: dict[str, dict] = {}

for _, name, _ in pkgutil.iter_modules(__path__):
    if name.startswith("_"):
        continue
    mod = importlib.import_module(f".{name}", __package__)
    if hasattr(mod, "SYSTEM_INFO") and hasattr(mod, "run"):
        info = mod.SYSTEM_INFO.copy()
        info["fn"] = mod.run
        SYSTEMS[name] = info
