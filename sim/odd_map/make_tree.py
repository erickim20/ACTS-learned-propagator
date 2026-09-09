"""Build the ODD tree whose compact carries the field map instead of the
analytic solenoid.

What this produces is not the ODD. The output tree differs from ODD v4.0.4 in
the field element of `xml/OpenDataDetector.xml`, in one added factory source,
and in one added data file. Nothing else: same modules, same material maps,
same alignment, same identifiers. The diff is printed at the end so that claim
is checked rather than asserted.

The field element it writes reads `cpp/ODDFieldMapXyz.cpp`'s plugin, which
DD4hep does not ship; see that file for why one had to be written.

    python sim/odd_map/make_tree.py \\
        --src  ~/baseline0/cache/odd-v4 \\
        --dst  ~/baseline0/cache/odd-map \\
        --map  ~/exp/odd-bfield-xyz.bin \\
        --map-container-path /cache/odd-map/data/odd-bfield-xyz.bin

`--map-container-path` is what goes into the compact, and it is a path inside
the image, not on this filesystem. It is passed rather than derived because
nothing here can see where the tree will be mounted.
"""
import argparse
import difflib
import pathlib
import re
import shutil
import sys

PLUGIN_SRC = "field/ODDFieldMapXyz.cpp"

# The element as ODD v4.0.4 ships it, matched rather than assumed: if the
# upstream text ever differs, this fails instead of silently leaving a
# uniform-field detector behind.
SOLENOID_RE = re.compile(
    r'<field\s+type="solenoid"\s+name="GlobalSolenoid"'
    r'[^>]*?/>',
    re.DOTALL)


def patch_compact(path, map_path):
    text = path.read_text()
    hits = SOLENOID_RE.findall(text)
    if len(hits) != 1:
        raise SystemExit(
            f"{path}: expected exactly one <field type=\"solenoid\"> element, "
            f"found {len(hits)}. The compact is not the one this was written "
            f"against and the field must not be replaced blind.")
    new = (f'<field type="ODDFieldMapXyz" name="GlobalFieldMap"\n'
           f'                        file="{map_path}" />')
    return text, SOLENOID_RE.sub(new, text)


def patch_cmake(path):
    text = path.read_text()
    if PLUGIN_SRC in text:
        return text, text
    anchor = "    calorimeter/ODDPolyhedraBarrelCalorimeter_geo.cpp)"
    if anchor not in text:
        raise SystemExit(
            f"{path}: source list does not end where this expects. Add "
            f"{PLUGIN_SRC} by hand rather than letting a patch guess.")
    return text, text.replace(anchor, anchor[:-1] + f"\n    {PLUGIN_SRC})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unmodified ODD checkout")
    ap.add_argument("--dst", required=True, help="tree to write")
    ap.add_argument("--map", required=True,
                    help="the 3-component binary from export_fieldbin.py")
    ap.add_argument("--map-container-path", required=True,
                    help="where that file will be readable from inside the "
                         "image; this is what the compact records")
    ap.add_argument("--plugin", default="cpp/ODDFieldMapXyz.cpp")
    args = ap.parse_args()

    src = pathlib.Path(args.src).expanduser()
    dst = pathlib.Path(args.dst).expanduser()
    mapbin = pathlib.Path(args.map).expanduser()
    plugin = pathlib.Path(args.plugin).expanduser()

    for p in (src / "xml/OpenDataDetector.xml", src / "factory/CMakeLists.txt",
              mapbin, plugin):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    if dst.exists():
        shutil.rmtree(dst)
    # copy2 keeps mtimes, so a rebuild does not redo unchanged sources
    shutil.copytree(src, dst, symlinks=True)

    compact = dst / "xml/OpenDataDetector.xml"
    before, after = patch_compact(compact, args.map_container_path)
    compact.write_text(after)

    cml = dst / "factory/CMakeLists.txt"
    cbefore, cafter = patch_cmake(cml)
    cml.write_text(cafter)

    (dst / "factory/field").mkdir(parents=True, exist_ok=True)
    shutil.copy2(plugin, dst / "factory" / PLUGIN_SRC)

    data = dst / "data/odd-bfield-xyz.bin"
    shutil.copy2(mapbin, data)

    print(f"tree: {dst}")
    print(f"  map: {data} ({data.stat().st_size / 1e6:.1f} MB), "
          f"compact records {args.map_container_path}")
    print()
    print("--- xml/OpenDataDetector.xml ---")
    sys.stdout.writelines(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile="odd-v4", tofile="odd-map", n=2))
    print("--- factory/CMakeLists.txt ---")
    sys.stdout.writelines(difflib.unified_diff(
        cbefore.splitlines(True), cafter.splitlines(True),
        fromfile="odd-v4", tofile="odd-map", n=2))


if __name__ == "__main__":
    main()
