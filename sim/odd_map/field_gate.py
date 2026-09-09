"""Read the detector's own field back, before anything is generated with it.

The check that the geometry changed. Three independent paths to the same
points, because a field that is right in one of them and wrong in another is
the failure mode that costs a month:

    --via dd4hep   DD4hep's own `OverlayedField`, straight off the compact.
                   This is what Geant4 flies the muons through in `ddsim`.
    --via acts     `getOpenDataDetector(...).field`, the `DD4hepFieldAdapter`
                   that `digi_and_reco.py:167` hands to the CKF.
    --via map      `acts.MagneticFieldMapXyz` on the csv the ODD ships, read
                   directly. The reference: it is the same numbers, through
                   none of the new code.

Each writes a csv. `--report` reads all three and prints the table.

Why three processes and not one. `dd4hep::Detector` is a singleton and both
the DD4hep path and the ACTS path load a compact into it, so doing both in one
interpreter loads the geometry twice.

    python3 field_gate.py --via dd4hep --out /tmp/g_dd4hep.csv
    python3 field_gate.py --via acts   --out /tmp/g_acts.csv
    python3 field_gate.py --via map    --out /tmp/g_map.csv --mapfile ...
    python3 field_gate.py --report /tmp/g_dd4hep.csv /tmp/g_acts.csv /tmp/g_map.csv

The verdict is in the exit code. Nonzero means the analytic solenoid is still
in the geometry, and nothing downstream is worth running.
"""
import argparse
import csv
import math
import os
import sys

# (r, z) in mm, with y = 0 so r is x. The first four are the tabulated points;
# the rest are there because a field that is right on the axis and
# wrong off it would pass a four-point check.
POINTS = [
    (0.0, 0.0),
    (500.0, 1500.0),
    (900.0, 2500.0),
    (0.0, 2900.0),
    (100.0, 0.0),
    (500.0, 0.0),
    (900.0, 0.0),
    (1100.0, 0.0),
    (0.0, 500.0),
    (0.0, 1500.0),
    (0.0, 2500.0),
    (200.0, 2900.0),
    (800.0, 2900.0),
]

# The analytic solenoid the ODD ships answers exactly this, everywhere a track
# goes, with no transverse component at all.
UNIFORM_BZ = 2.0
FLAT_TOL = 1e-4


def rows_dd4hep(compact):
    """DD4hep's own field, off the compact. Units are the trap; see below."""
    import dd4hep

    description = dd4hep.Detector.getInstance()
    description.fromXML(compact)
    fld = description.field()

    # DD4hep's default unit system is not Geant4's: `Evaluator/DD4hepUnits.h`
    # sets millimeter = 0.1, so a position of 1.0 means one centimetre, and
    # `tesla` is derived rather than 1. Both constants are read out of the
    # build rather than assumed, and printed, because assuming them is exactly
    # how a field ends up off by a factor nobody sees.
    import cppyy
    mm = float(cppyy.gbl.dd4hep.mm)
    tesla = float(cppyy.gbl.dd4hep.tesla)
    print(f"[dd4hep] 1 mm = {mm:g}, 1 tesla = {tesla:g} in this build",
          file=sys.stderr)

    from array import array
    out = []
    for r, z in POINTS:
        pos = array("d", [r * mm, 0.0, z * mm])
        b = array("d", [0.0, 0.0, 0.0])
        fld.magneticField(pos, b)
        out.append((r, z, b[0] / tesla, b[1] / tesla, b[2] / tesla))
    return out


def rows_acts(_compact):
    """What the reconstruction gets: `detector.field`, built as digi_and_reco
    builds it. The compact is not passed here on purpose: this path has to
    resolve the geometry the way the production script does, through ODD_PATH,
    or it is not testing the thing that will run."""
    import acts
    from acts.examples.odd import (getOpenDataDetector,
                                   getOpenDataDetectorDirectory)

    u = acts.UnitConstants
    geoDir = getOpenDataDetectorDirectory()
    print(f"[acts] ODD_PATH={os.environ.get('ODD_PATH')} geoDir={geoDir}",
          file=sys.stderr)
    deco = acts.IMaterialDecorator.fromFile(
        geoDir / "data/odd-material-maps.root")
    detector = getOpenDataDetector(odd_dir=geoDir, materialDecorator=deco)
    field = detector.field
    print(f"[acts] type(detector.field) = {type(field)}", file=sys.stderr)

    cache = field.makeCache(acts.MagneticFieldContext())
    out = []
    for r, z in POINTS:
        b = field.getField(acts.Vector3(r * u.mm, 0.0, z * u.mm), cache)
        out.append((r, z, b[0] / u.T, b[1] / u.T, b[2] / u.T))
    return out


def rows_map(mapfile):
    """The reference: the shipped map, through ACTS, through none of the new
    code."""
    import acts

    u = acts.UnitConstants
    m = acts.MagneticFieldMapXyz(mapfile)
    cache = m.makeCache(acts.MagneticFieldContext())
    out = []
    for r, z in POINTS:
        b = m.getField(acts.Vector3(r * u.mm, 0.0, z * u.mm), cache)
        out.append((r, z, b[0] / u.T, b[1] / u.T, b[2] / u.T))
    return out


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r_mm", "z_mm", "bx_T", "by_T", "bz_T"])
        for row in rows:
            w.writerow([f"{v:.6g}" for v in row])
    print(f"wrote {path}", file=sys.stderr)


def read(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    # drop the header before converting, not after
    return [tuple(float(v) for v in row) for row in rows[1:]]


def report(paths):
    named = {}
    for p in paths:
        for key in ("dd4hep", "acts", "map"):
            if key in os.path.basename(p):
                named[key] = read(p)
    missing = {"dd4hep", "acts", "map"} - set(named)
    if missing:
        raise SystemExit(f"no csv for {sorted(missing)} among {paths}")

    # The header names the detector it actually read rather than asserting
    # which one it is. Run as a control against the unmodified ODD, a fixed
    # caption would be a false statement in the one place a reader trusts.
    odd = os.environ.get("ODD_PATH", "unset")
    element = "not found"
    try:
        with open(os.path.join(odd, "xml/OpenDataDetector.xml")) as f:
            for line in f:
                s = line.strip()
                if s.startswith("<field ") and not s.startswith("<!--"):
                    element = s
                    break
    except OSError as e:
        element = f"unreadable: {e}"

    print()
    print("Field read back from the detector.")
    print(f"  ODD_PATH       {odd}")
    print(f"  field element  {element}")
    if "ODDFieldMapXyz" in element:
        print("  This detector is NOT the ODD: its field is the shipped map, "
              "not the analytic solenoid the release defines.")
    else:
        print("  This is the ODD as shipped.")
    print()
    print(f"{'r':>6}{'z':>7} | {'DD4hep Bz':>10}{'|Bt|':>8} | "
          f"{'ACTS Bz':>9}{'|Bt|':>8} | {'map Bz':>9}{'|Bt|':>8}")
    print("-" * 74)

    flat = {"dd4hep": True, "acts": True}
    worst = 0.0
    for i in range(len(named["map"])):
        r, z = named["map"][i][0], named["map"][i][1]
        cells = []
        for key in ("dd4hep", "acts", "map"):
            _, _, bx, by, bz = named[key][i]
            bt = math.hypot(bx, by)
            cells.append((bz, bt))
            if key in flat:
                if (abs(abs(bz) - UNIFORM_BZ) > FLAT_TOL or bt > FLAT_TOL):
                    flat[key] = False
        for key, (bz, bt) in zip(("dd4hep", "acts"), cells):
            worst = max(worst, abs(bz - cells[2][0]), abs(bt - cells[2][1]))
        print(f"{r:6.0f}{z:7.0f} | "
              f"{cells[0][0]:10.4f}{cells[0][1]:8.4f} | "
              f"{cells[1][0]:9.4f}{cells[1][1]:8.4f} | "
              f"{cells[2][0]:9.4f}{cells[2][1]:8.4f}")
    print()
    print(f"largest disagreement with the shipped map, either path: "
          f"{worst:.4f} T")

    bad = [k for k, v in flat.items() if v]
    if bad:
        print()
        for k in bad:
            print(f"GATE FAILED: the {k} path still reads a flat "
                  f"{UNIFORM_BZ:.4f} T with no transverse component. "
                  f"The geometry did not change on that path.")
        return 1

    # The map's own values are the reference, so a path that disagrees with it
    # is reading something else even if it is not flat.
    if worst > 1e-3:
        print()
        print(f"GATE FAILED: a path disagrees with the shipped map by "
              f"{worst:.4f} T, which is more than the map's own 0.001 T "
              f"quantisation.")
        return 1

    print()
    print("GATE PASSED: both paths read the map, and agree with it.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--via", choices=["dd4hep", "acts", "map"])
    ap.add_argument("--out")
    ap.add_argument("--compact", default="")
    ap.add_argument("--mapfile", default="")
    ap.add_argument("--report", nargs="*", default=None)
    a = ap.parse_args()

    if a.report is not None:
        raise SystemExit(report(a.report))

    if not a.via or not a.out:
        raise SystemExit("--via and --out, or --report")

    if a.via == "dd4hep":
        compact = a.compact or (
            os.path.join(os.environ["ODD_PATH"], "xml/OpenDataDetector.xml"))
        rows = rows_dd4hep(compact)
    elif a.via == "acts":
        rows = rows_acts(a.compact)
    else:
        if not a.mapfile:
            raise SystemExit("--via map needs --mapfile")
        rows = rows_map(a.mapfile)
    write(a.out, rows)


if __name__ == "__main__":
    main()
