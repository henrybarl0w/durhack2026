#the reader's debug layer: finds out WHICH step of the reader did something wrong. everything here is optional,
#delete this file and reader/probe.py's fallbacks take over.
#   probe    records a number or note a step wants to show (what it measured, how many things it changed)
#   expect   an invariant that must hold if the step works; a failure is recorded against that step
#   guarded  runs a step's new code, or its old code when the step is switched off (features.py), and if the
#            new code crashes it falls back to the old code and records the crash against that step
#everything lands in trace["steps"][step] and is printed in the dump's STEP HEALTH section
import traceback

from . import features


def record(trace, step):
    if trace is None:
        return {"new": 0, "legacy": 0, "crashed": 0, "error": None, "violations": [], "probes": {}}
    return trace.setdefault("steps", {}).setdefault(
        step, {"new": 0, "legacy": 0, "crashed": 0, "error": None, "violations": [], "probes": {}})


def probe(trace, step, name, **data):
    if trace is None:
        return
    entries = record(trace, step)["probes"].setdefault(name, [])
    if len(entries) < 200:
        entries.append(data)


def expect(trace, step, ok, message, **data):
    if trace is None or ok:
        return
    violations = record(trace, step)["violations"]
    if len(violations) < 50:
        violations.append({"message": message, **data})


def guarded(trace, step, new, legacy, *args, **kwargs):
    entry = record(trace, step)
    if not features.on(step):
        entry["legacy"] += 1
        return legacy(*args, **kwargs)
    try:
        result = new(*args, **kwargs)
        entry["new"] += 1
        return result
    except Exception:
        entry["crashed"] += 1
        entry["legacy"] += 1
        entry["error"] = entry["error"] or traceback.format_exc(limit=8)
        return legacy(*args, **kwargs)
