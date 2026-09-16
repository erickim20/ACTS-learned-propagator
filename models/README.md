# models

What was measured, as far as it can be published while staying self-consistent.

## What is here

| file | bytes | md5 | what it is |
| --- | ---: | --- | --- |
| `gtheta_S.npz` | 50,150 | `49a2ddd421afbe5908b8a7e2dbea6805` | the fitted model, as numpy arrays |
| `sigma_head_S.npz` | 648,004 | `ee727760caec0c1b5be6721b51caf838` | the per-jump sigma head for the same fit |
| `fixtures/muon_jumps.bin` | 6,488,780 | `3709ef6cde491bca96a7f95cf4189f53` | 4,000 real ODD jumps onto their own module planes, the input `bench_kernel` reads |

`gtheta_S.npz` and `sigma_head_S.npz` are the source of `cpp/gtheta_weights.hpp`.
Re-exporting from them reproduces the tracked header:

    python -m prop.export_kernel --model models/gtheta_S.npz \
        --head models/sigma_head_S.npz --no-field

The `source:` comment the exporter writes names the paths it was given, so the
file md5 will differ from the tracked one. The model does not. Compare the two
with the comments removed:

    grep -v '^//' cpp/gtheta_weights.hpp | md5sum

which is `833ac83de2ed3dce591883139bda3b9b` for the tracked header and for a
re-export from these two files.

`fixtures/muon_jumps.bin` is geometry and truth only. It carries no model, so it
is valid against any build, and `bench_kernel` can now run from a clone.

## What is not here, and why

**The process noise tables are not here.** They are measured, not computed, and
they cannot be published with this model, because they were not measured on it.

A Q table records in its header the md5 of the `cpp/gtheta_weights.hpp` whose
residuals it was built from, and `NoiseTable::load` refuses a table whose md5
does not match the binary about to arm it. Every table on the machine where
these were measured names a different model from the one this repository ships.

| table | bytes | md5 | field gate | material | branches | weights md5 in header | equals this repository's model |
| --- | ---: | --- | ---: | :---: | ---: | --- | :---: |
| `q_table_B.bin` | 15,695 | `43b123f78a01d0cb39d1d325391d63ae` | none | off | 1 | none in header | unverifiable |
| `q_B_brah2.bin` | 31,374 | `36df6eecdf5f3783145db0207ce0f275` | none | off | 2 | none in header | unverifiable |
| `q_B_g000.bin` | 31,414 | `7c499e255666579cee89b56d530a913f` | 0 | off | 2 | `b8b80f83167772e39de2b312f51e0f74` | no |
| `q_B_g005.bin` | 31,414 | `486e5516e3b2903dd7f5559a2a8c167f` | 0.05 | off | 2 | `b8b80f83167772e39de2b312f51e0f74` | no |
| `q_B_g010.bin` | 31,414 | `e346a924fbfd1b50bf99d59641d47e6b` | 0.10 | off | 2 | `b8b80f83167772e39de2b312f51e0f74` | no |
| `q_B_g020.bin` | 31,414 | `daaba6c9156e86259e7c75c1cc2738ad` | 0.20 | off | 2 | `b8b80f83167772e39de2b312f51e0f74` | no |
| `q_B_g040.bin` | 31,414 | `4b0590764f34b1273c8f0cbf357d039d` | 0.40 | off | 2 | `b8b80f83167772e39de2b312f51e0f74` | no |
| `q_B_g999.bin` | 31,414 | `55890999bd743e5ec479e607f2374156` | 999 | off | 2 | `b8b80f83167772e39de2b312f51e0f74` | no |

Header fields read from each file with the layout `prop.export_qtable` writes.
This repository's `cpp/gtheta_weights.hpp` is `22c48f83868da542ad6311d3911937ed`
and `b8b80f83167772e39de2b312f51e0f74` is a different fit, not a different
spelling of the same one: with comments removed the two headers differ in 974 of
their 1,023 weight lines.

The two tables with no md5 in the header predate the provenance format.
`NoiseTable::load` refuses those as well, and asks for a re-export with
`--weights` and `--field-gate`.

Publishing the tables anyway would put files in this repository that this
repository's own loader rejects. They stay where they were measured until
either the model they were measured on is published here, or they are
re-exported against this one.

**`reference.bin` is not here.** `test_kernel` replays it, and it holds the
outputs of `closed_loop.predict`, so it belongs to whichever model produced it.
The copy that exists was written in the same minute as the other model's
weights and it carries no provenance of its own, so it cannot be published
against this model without checking it first. `prop.export_kernel` regenerates
it from `gtheta_S.npz` and a teacher parquet.

**`oddb.npz` and `cpp/field.bin` are not here and do not need to be.**
`prop.build_fieldmap` rebuilds `oddb.npz` from the image's `odd-bfield.csv` and
`prop.export_fieldbin` writes `field.bin` from it. See the "Generated inputs"
table in the top-level README.

## What a reader can and cannot do with this

| | |
| --- | --- |
| rebuild `cpp/gtheta_weights.hpp` | yes, from the two npz here |
| run `bench_kernel` | yes, from `fixtures/muon_jumps.bin` |
| run `test_kernel` | no, `reference.bin` is not published |
| arm the stepper with a measured `Q` | no, no table here matches this model |
