"""The pT by |eta| efficiency grid, from sim/arm_grid.py's JSON.

    python -m prop.arm_grid_table ~/partA/arms.json --baseline stock

`sim/seam_table.py` pools |eta| and `sim/eta_table.py` pools pT, and the loss
the moved seam's transport carries is a function of the two together, so
neither of those tables shows it. This prints the grid, because a figure is not
a number anyone can put in a table.

Efficiency is passed over total in the cell, summed the way `sim/eta_table.py`
sums: `trackeff_vs_eta`'s two histograms folded to |eta|, per pT bin, no
pooling. A cell with no truth particles prints as a dash.
"""
import argparse
import json

PT_LABEL = {1: "1 to 2", 2: "2 to 4", 3: "4 to 8", 4: "8 to 20",
            5: "20 to 50"}


def cells(rows):
    """{bin: [efficiency per |eta| band]} and the truth totals beside it."""
    eff, tot = {}, {}
    for r in rows:
        e, t = [], []
        for p, n in zip(r["eta_passed"], r["eta_total"]):
            e.append(100.0 * p / n if n > 0 else None)
            t.append(n)
        eff[r["bin"]] = e
        tot[r["bin"]] = t
    return eff, tot


def band_labels(edges):
    return [f"{edges[k]:.1f}-{edges[k + 1]:.1f}" for k in range(len(edges) - 1)]


def show(title, grid, labels, bins, width=12):
    print(title)
    print()
    print(f"{'pT [GeV]':<10}" + "".join(f"{b:>{width}}" for b in labels))
    for b in bins:
        row = f"{PT_LABEL.get(b, str(b)):<10}"
        for v in grid[b]:
            row += f"{'-':>{width}}" if v is None else f"{v:>{width}.3f}"
        print(row)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--baseline", default="stock")
    a = ap.parse_args()

    with open(a.json) as f:
        d = json.load(f)
    labels = band_labels(d["eta_edges"])
    bins = d["bins"]
    base, tot = cells(d["arms"][a.baseline])

    _, t0 = cells(d["arms"][a.baseline])
    show("truth particles, |eta| across, pT down",
         {b: [float(x) for x in t0[b]] for b in bins}, labels, bins)

    for name, rows in d["arms"].items():
        eff, _ = cells(rows)
        show(f"efficiency [%], {name}", eff, labels, bins)

    for name, rows in d["arms"].items():
        if name == a.baseline:
            continue
        eff, _ = cells(rows)
        diff = {b: [None if (x is None or y is None) else x - y
                    for x, y in zip(eff[b], base[b])] for b in bins}
        show(f"efficiency points against {a.baseline}, {name}",
             diff, labels, bins)


if __name__ == "__main__":
    raise SystemExit(main())
