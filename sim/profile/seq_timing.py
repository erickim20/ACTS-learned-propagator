"""Run ColliderML's `digi_and_reco.py` with the Sequencer writing `timing.tsv`.

That script builds its Sequencer without an `outputDir`, so
the Geant4 arm has stage wall times and no per-algorithm breakdown. Patching the
one constructor call and then running the script as it stands keeps the chain
and its cuts ColliderML's. A copy of the script would answer the same question
today and drift from it silently afterwards.

Runs inside the pinned image, and is driven by environment because a container
entrypoint is a poor place to pass arguments:

    RECO_SCRIPT   the `digi_and_reco.py` to run
    TIMING_DIR    where the Sequencer writes `timing.tsv`

Everything on the command line goes through to `digi_and_reco.py` unchanged.
"""
import os
import runpy
import sys
from pathlib import Path

import acts.examples

RECO_SCRIPT = Path(os.environ["RECO_SCRIPT"])
TIMING_DIR = os.environ["TIMING_DIR"]

_Sequencer = acts.examples.Sequencer


def Sequencer(*args, **kwargs):
    """`acts.examples.Sequencer` with the timing output filled in."""
    kwargs.setdefault("outputDir", TIMING_DIR)
    kwargs.setdefault("outputTimingFile", "timing.tsv")
    return _Sequencer(*args, **kwargs)


# `digi_and_reco.py` does `from acts.examples import Sequencer` at import time,
# so the attribute has to be replaced before runpy reads the file.
acts.examples.Sequencer = Sequencer

# sys.path[0] is this file's directory rather than the script's, and `utils`
# sits beside `digi_and_reco.py` rather than on PYTHONPATH.
sys.path.insert(0, str(RECO_SCRIPT.parent))
sys.argv = [str(RECO_SCRIPT)] + sys.argv[1:]

runpy.run_path(str(RECO_SCRIPT), run_name="__main__")
