# models

The fitted model this repository ships, the process noise tables measured on
it, and what is still missing.

## What is here

| file | bytes | md5 | what it is |
| --- | ---: | --- | --- |
| `tables/q_B_g000.bin` | 31,414 | `9c2215b70a55d7d4e16de56688907eea` | process noise, field gate 0 |
| `tables/q_B_g005.bin` | 31,414 | `749bb929cad89f6a97ba9ac7eca99c8e` | process noise, field gate 0.05 |
| `tables/q_B_g010.bin` | 31,414 | `bed7872aac86018e1abfe390be43ff0d` | process noise, field gate 0.10 |
| `tables/q_B_g020.bin` | 31,414 | `b06dc26827d26081e403a6f55b3782cf` | process noise, field gate 0.20 |
| `tables/q_B_g040.bin` | 31,414 | `f03ca2d7e8b7bf6a10520ccc3e5cfa98` | process noise, field gate 0.40 |
| `tables/q_B_g999.bin` | 31,414 | `1ed4d5b068061e9c856ae23bfaa3127e` | process noise, field gate 999 |
| `gtheta_gonly.npz` | 50,150 | `6882d2159847ef3e2289ba8607654864` | the fit `cpp/gtheta_weights.hpp` holds |
| `sigma_head_gonly.npz` | 647,870 | `1d7ece65500936b70dba4743d32a8c5b` | the per-jump sigma head for that fit |
| `gtheta_S.npz` | 50,150 | `49a2ddd421afbe5908b8a7e2dbea6805` | a different fit, see below |
| `sigma_head_S.npz` | 648,004 | `ee727760caec0c1b5be6721b51caf838` | the per-jump sigma head for that other fit |
| `fixtures/muon_jumps.bin` | 6,488,780 | `3709ef6cde491bca96a7f95cf4189f53` | 4,000 real ODD jumps onto their own module planes, the input `bench_kernel` reads |

Every md5 is of the file as committed.

## The model and the tables agree

`cpp/gtheta_weights.hpp` holds the fit every table here was measured on. The
header identifies the model by the digest of its non-comment lines:

    grep -v '^[[:space:]]*//' cpp/gtheta_weights.hpp | md5sum

which is `6221937713ebc4d742f45477595ec4de`, and that is the 32 characters each
table carries at byte 24 of its own header.

| | |
| --- | --- |
| `cpp/gtheta_weights.hpp` | 156,507 bytes, md5 `3945e23e28c20afc33a29a169839c259` |
| its non-comment digest | `6221937713ebc4d742f45477595ec4de` |
| the stamp in all six tables | `6221937713ebc4d742f45477595ec4de` |

`gtheta_gonly.npz` and `sigma_head_gonly.npz` are that fit. Re-exporting them
rebuilds the header:

    python -m prop.export_kernel --model models/gtheta_gonly.npz \
        --head models/sigma_head_gonly.npz --pairs <teacher> --no-field

and the result has the same non-comment digest,
`6221937713ebc4d742f45477595ec4de`. It is not the same FILE as the header here:
the `source:` comment on line 2 records the paths the exporter was given, so a
re-export names the reader's own paths and the file md5 differs. The digest
excludes comments for exactly that reason.

`cpp/incontainer_build_ckf.sh` computes that digest and compiles it in, and
`NoiseTable::load` compares it against what the table records. The digest is
over the non-comment lines rather than the whole file because the `source:`
comment on line 2 names the paths the exporter was given, which differ between
any two copies of one model. Over the whole file the stamp named the file and
not the model, and a table stamped against one copy was refused by a build made
from the other.

## What each table is

| table | magic | field gate | material | branches | stamp |
| --- | --- | ---: | :---: | ---: | --- |
| `tables/q_B_g000.bin` | `QTB3` | 0 | off | 2 | `6221937713ebc4d742f45477595ec4de` |
| `tables/q_B_g005.bin` | `QTB3` | 0.05 | off | 2 | `6221937713ebc4d742f45477595ec4de` |
| `tables/q_B_g010.bin` | `QTB3` | 0.10 | off | 2 | `6221937713ebc4d742f45477595ec4de` |
| `tables/q_B_g020.bin` | `QTB3` | 0.20 | off | 2 | `6221937713ebc4d742f45477595ec4de` |
| `tables/q_B_g040.bin` | `QTB3` | 0.40 | off | 2 | `6221937713ebc4d742f45477595ec4de` |
| `tables/q_B_g999.bin` | `QTB3` | 999 | off | 2 | `6221937713ebc4d742f45477595ec4de` |

Header fields read from each file with the layout `prop.export_qtable` writes:
six little-endian ints, then the 32 ascii characters of the stamp, then the
field gate as a double, for a 64-byte header.

Material is off on every table. A table measured with material on would double
count the scattering ACTS adds at every material surface, and `NoiseTable::load`
refuses one.

The field gate is part of what a table measures and not a runtime preference.
Above the threshold the network transports the jump and at or below it the helix
runs alone, so the threshold decides which transport made each jump the
residuals came from. `NoiseTable::load` refuses a table whose gate is not the
gate the run arms.

Two branches means the file carries a separate measurement for the jumps the
network fired on and for the jumps it did not, and the stepper picks the one
each jump was made by.

## Which one to arm

`tables/q_B_g040.bin` at `FIELD_GATE=0.40`, with the sigma head off, the fired
branch scaled by 0.70 and the helix branch by 0.85.

| the recommended configuration | efficiency [%] | fake ratio [%] |
| --- | ---: | ---: |
| single muons | 96.129 | 0.2949 |
| ttbar at pileup 200 | 87.757 | 8.1557 |

After ambiguity resolution, against 49,938 truth particles over five bins of
10,000 single-muon events at seeds 101 to 105, and 16,753 truth particles over
twenty pileup-200 ttbar events at five seeds. The two branch scales are the
source change this branch also carries.

## What is not here, and why

**Two fits are here and only one of them has tables.** `gtheta_gonly.npz` and
`sigma_head_gonly.npz` are the fit `cpp/gtheta_weights.hpp` holds and every
table was measured on, trained on the gated population, digest
`6221937713ebc4d742f45477595ec4de`. `gtheta_S.npz` and `sigma_head_S.npz` are
the whole-population fit, digest `833ac83de2ed3dce591883139bda3b9b`. With
comments removed the two headers differ in 974 of their 1,023 weight lines.
They are two fits and not two spellings of one.

| model | non-comment digest | npz | bytes | md5 | here |
| --- | --- | --- | ---: | --- | :---: |
| gated, which the header and the tables are | `6221937713ebc4d742f45477595ec4de` | `gtheta_gonly.npz` | 50,150 | `6882d2159847ef3e2289ba8607654864` | yes |
| | | `sigma_head_gonly.npz` | 647,870 | `1d7ece65500936b70dba4743d32a8c5b` | yes |
| whole population | `833ac83de2ed3dce591883139bda3b9b` | `gtheta_S.npz` | 50,150 | `49a2ddd421afbe5908b8a7e2dbea6805` | yes |
| | | `sigma_head_S.npz` | 648,004 | `ee727760caec0c1b5be6721b51caf838` | yes |

The whole-population pair stays. It is the model this repository published
first and the one the reports up to that point were taken on, so removing it
would delete the only published copy of a model those numbers belong to. Arming
a table with a build made from it is not a risk the reader has to avoid by
reading this file: `NoiseTable::load` compares the digest and refuses, and the
message names both values. No table here is measured against
`833ac83de2ed3dce591883139bda3b9b` and none is planned, because that would be a
new measurement and not a re-export.

**Two older tables are not here.** They carry no stamp at all, so which model
they measure cannot be read out of the file, and `NoiseTable::load` refuses a
non-provenance table outright and asks for a re-export with `--weights` and
`--field-gate`.

| table | bytes | md5 | magic | branches |
| --- | ---: | --- | --- | ---: |
| `q_table_B.bin` | 15,695 | `43b123f78a01d0cb39d1d325391d63ae` | `QTAB` | 1 |
| `q_B_brah2.bin` | 31,374 | `36df6eecdf5f3783145db0207ce0f275` | `QTB2` | 2 |

**`reference.bin` is not here.** `test_kernel` replays it and it holds the
outputs of `closed_loop.predict`, so it belongs to whichever model produced it,
and it carries no provenance of its own: there is no magic and no digest in the
file, only three little-endian counts and then doubles. The copy that exists is
592,012 bytes, md5 `8d59beed6240bb5484d035212005cbda`, and which model wrote it
cannot be read out of it. It would go up unlabelled or not at all.
`prop.export_kernel` regenerates it from a model and a teacher parquet.

**`oddb.npz` and `cpp/field.bin` are not here and do not need to be.**
`prop.build_fieldmap` rebuilds `oddb.npz` from the image's `odd-bfield.csv` and
`prop.export_fieldbin` writes `field.bin` from it. See the "Generated inputs"
table in the top-level README.

## What a reader can and cannot do with this

| | |
| --- | --- |
| arm the stepper with a measured `Q` | yes, any of the six tables, at its own gate |
| check that a table and the build agree before arming | yes, the digest command above against byte 24 of the table |
| run `bench_kernel` | yes, from `fixtures/muon_jumps.bin` |
| rebuild `cpp/gtheta_weights.hpp` | yes, `prop.export_kernel` from `gtheta_gonly.npz` and `sigma_head_gonly.npz`; the digest matches, the file md5 does not, because line 2 names the exporter's own paths |
| run `test_kernel` | no, `reference.bin` is not published |
| reproduce the measurement that made a table | no, that needs the residual dump and the simulated sample, neither of which is here |
