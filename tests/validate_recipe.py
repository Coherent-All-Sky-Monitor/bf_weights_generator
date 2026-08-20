#!/usr/bin/env python
"""Byte-for-byte validation of make_cal_and_weights.py.

Reproduces three existing deployed/validated products through the DRIVER into
a scratch dir and compares every HDF5 dataset bit-exactly (np.array_equal on
the raw arrays, plus md5 of the dataset bytes).

  T1  cal_sun_2026-08-19_thr1_phaseonly_16ant.h5              (no static)
  T2  cal_sun_2026-08-19_thr1_phaseonly_16ant_nightstaticB.h5 (static 02:45-03:15 UT)
  T3  weights_aug19_16ant_solartrack_512_int8_CAL0819.h5      (grid from T1's cal)

Not a pytest test: it needs the raw visibilities on /mnt and the three
reference products on /mnt/nvme5, so it is run by hand after any change to
the driver. Exit code 0 = all three bit-exact.

    /home/casm/software/dev/casm_venvs/casm_offline_env/bin/python \\
      /home/casm/software/dev/bf_weights_generator/tests/validate_recipe.py
"""
import hashlib
import os
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0, "/home/casm/software/dev/bf_weights_generator")
from bf_weights_generator.make_cal_and_weights import RecipeParams, run

SCRATCH = "/mnt/nvme5/solar0819/_validate_recipe_scratch"
S16 = [9, 10, 15, 19, 22, 23, 24, 26, 30, 32, 36, 38, 40, 42, 44, 45]
LAY = "/home/casm/software/dev/antenna_layouts/casm_antenna_layout_2026-08-07.csv"
WIN19 = ("2026-08-19 20:41:30", "2026-08-19 21:41:30")

T1_REF = ("/mnt/nvme5/solar0819/newcal_build/"
          "cal_sun_2026-08-19_thr1_phaseonly_16ant.h5")
T2_REF = ("/mnt/nvme5/solar0819/cal_static_night/"
          "cal_sun_2026-08-19_thr1_phaseonly_16ant_nightstaticB.h5")
T2_STATIC = ("/mnt/nvme5/solar0819/cal_static_night/"
             "static_aug20_0245_0315UT.npz")
T3_REF = ("/mnt/nvme5/solar0819/newcal_build/"
          "weights_aug19_16ant_solartrack_512_int8_CAL0819.h5")


def _md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _datasets(path):
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj[...]
    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return out


def compare(label, new, ref, only=None):
    a, b = _datasets(new), _datasets(ref)
    keys = sorted(set(a) | set(b)) if only is None else list(only)
    rows, ok_all = [], True
    for k in keys:
        if k not in a or k not in b:
            rows.append((k, "MISSING", "-", "-"))
            ok_all = False
            continue
        ok = (a[k].shape == b[k].shape and a[k].dtype == b[k].dtype
              and np.array_equal(a[k], b[k]))
        ok_all &= ok
        rows.append((k, "YES" if ok else "NO", _md5(a[k]), _md5(b[k])))
    print(f"\n=== {label} ===")
    print(f"  new: {new}\n  ref: {ref}")
    print(f"  {'dataset':<22} {'bit-exact':<10} {'md5(new)':<34} {'md5(ref)'}")
    for k, s, m1, m2 in rows:
        print(f"  {k:<22} {s:<10} {m1:<34} {m2}")
    print(f"  ALL DATASETS BIT-EXACT: {ok_all}")
    return ok_all, rows


def main():
    if os.path.exists(SCRATCH):
        shutil.rmtree(SCRATCH)
    os.makedirs(SCRATCH)

    common = dict(out_dir=SCRATCH, layout_csv=LAY, antennas=S16, ref_ant=9,
                  cal_source="sun", source_window=WIN19,
                  make_weights=False, diagnostics=False, notebook=False)

    # ---- T1: no-static cal -------------------------------------------
    r1 = run(RecipeParams(tag="T1_nostatic", **common))
    ok1, rows1 = compare("T1 cal (no static)", r1["cal_file"], T1_REF)

    # ---- T2: static 02:45-03:15 UT cal --------------------------------
    r2 = run(RecipeParams(tag="T2_staticB", static_window=("2026-08-20 02:45",
                                                           "2026-08-20 03:15"),
                          static_path=T2_STATIC, **common))
    ok2, rows2 = compare("T2 cal (night static B)", r2["cal_file"], T2_REF)

    # ---- T3: solar-track weights from the EXISTING T1 cal -------------
    r3 = run(RecipeParams(
        out_dir=SCRATCH, tag="T3_solartrack", layout_csv=LAY, antennas=S16,
        ref_ant=9, cal_path=T1_REF, make_weights=True,
        grid_mode="bounds", alt_min_deg=0.0, alt_max_deg=33.0,
        az_min_deg=258.0, az_max_deg=292.0, n_beams=512,
        mult_min=0.05, mult_max=0.60, mult_n=56, trim="lowest",
        verify_beams=(30, 83, 353), diagnostics=False, notebook=False))
    ok3, rows3 = compare("T3 solar-track weights", r3["weights_file"], T3_REF,
                         only=["weights_int8", "pointings/alt_deg",
                               "pointings/az_deg"])
    ok3b, _ = compare("T3 solar-track weights (ALL datasets)",
                      r3["weights_file"], T3_REF)

    print("\n================ SUMMARY ================")
    for lab, ok in [("T1 cal no-static", ok1), ("T2 cal night-static B", ok2),
                    ("T3 weights solar-track (required datasets)", ok3),
                    ("T3 weights solar-track (all datasets)", ok3b)]:
        print(f"  {lab:<45} {'MATCH' if ok else 'MISMATCH'}")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())
