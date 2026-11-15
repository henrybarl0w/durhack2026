#the reader's only link to the debug layer. delete reader/debug.py (and webapp/debug*.py) to strip the debugging out:
#the fallbacks below do nothing and the reader runs exactly as before, with no further edits
try:
    from .debug import expect, guarded, probe
except ImportError:
    def probe(trace, step, name, **data):
        pass

    def expect(trace, step, ok, message, **data):
        pass

    def guarded(trace, step, new, legacy, *args, **kwargs):
        return new(*args, **kwargs)
