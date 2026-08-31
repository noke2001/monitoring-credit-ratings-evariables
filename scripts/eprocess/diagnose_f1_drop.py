"""
Why did F1 fall from ~0.63 to ~0.19?

    python tests_and_plots/diagnose_f1_drop.py

Scores ONE fixed set of alarms, from the corrected e-process, under both the
old "Regime" metric and the new windowed metric. If the method had degraded,
the old metric would also report a low number. If only the definition changed,
the old metric reproduces ~0.6 on the same alarms.

The decisive column is the CHANCE baseline: what the same metric awards to
alarms placed uniformly at random, matched in count. A precision that does not
beat its own chance baseline is not evidence of anything.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eprocess"))

from benchmark_eprocesses import load_vae
from src.sequential import DeadzoneEProcess

RULE = "=" * 92


def old_regime_metric(df, alarm_col="is_alarm"):
    """
    The metric behind the 0.6265 in the withdrawn report.

      precision = share of alarms whose NEXT rating transition is a downgrade
                  (no time limit)
      recall    = share of downgrades preceded by ANY alarm, ever
                  (no time limit)
    """
    dates = pd.to_datetime(df["dates"]).to_numpy()
    ids = df["isin"].to_numpy()
    is_down = df["is_downgrade"].to_numpy().astype(int)
    is_chg = df["is_rating_change"].to_numpy().astype(int)

    ch_by_id, al_by_id = {}, {}
    for p in np.flatnonzero(is_chg == 1):
        ch_by_id.setdefault(ids[p], []).append(p)
    alarm_pos = np.flatnonzero(df[alarm_col].to_numpy().astype(bool))
    for p in alarm_pos:
        al_by_id.setdefault(ids[p], []).append(p)
    down_pos = np.flatnonzero(is_down == 1)
    if len(alarm_pos) == 0 or len(down_pos) == 0:
        return dict(P=0.0, R=0.0, F1=0.0, n_alarms=len(alarm_pos))

    tp = 0
    for p in alarm_pos:
        nxt = [q for q in ch_by_id.get(ids[p], ()) if dates[q] > dates[p]]
        if nxt and is_down[nxt[0]] == 1:
            tp += 1
    prec = tp / len(alarm_pos)

    caught = 0
    for q in down_pos:
        if any(dates[p] < dates[q] for p in al_by_id.get(ids[q], ())):
            caught += 1
    rec = caught / len(down_pos)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(P=100 * prec, R=100 * rec, F1=f1, n_alarms=len(alarm_pos))


def windowed_metric(df, H, event="is_rating_change", alarm_col="is_alarm"):
    """Precision and recall over the same H-month window on both sides."""
    dates = pd.to_datetime(df["dates"]).to_numpy()
    ids = df["isin"].to_numpy()
    ev = np.flatnonzero(df[event].to_numpy().astype(float) == 1)
    al = np.flatnonzero(df[alarm_col].to_numpy().astype(bool))
    if len(al) == 0 or len(ev) == 0:
        return dict(P=0.0, R=0.0, F1=0.0, n_alarms=len(al))

    def months(a, b):
        A, B = pd.Timestamp(a), pd.Timestamp(b)
        return (A.year - B.year) * 12 + (A.month - B.month)

    ev_by, al_by = {}, {}
    for q in ev:
        ev_by.setdefault(ids[q], []).append(q)
    for p in al:
        al_by.setdefault(ids[p], []).append(p)

    tp_a = sum(1 for p in al
               if any(0 < months(dates[q], dates[p]) <= H
                      for q in ev_by.get(ids[p], ())))
    tp_e = sum(1 for q in ev
               if any(0 < months(dates[q], dates[p]) <= H
                      for p in al_by.get(ids[q], ())))
    prec, rec = tp_a / len(al), tp_e / len(ev)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(P=100 * prec, R=100 * rec, F1=f1, n_alarms=len(al))


def random_alarms(df, n, seed=0):
    """Count-matched alarms placed uniformly at random over bond-months."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    flags = np.zeros(len(df), dtype=bool)
    flags[rng.choice(len(df), size=n, replace=False)] = True
    out["rand_alarm"] = flags
    return out


def main():
    print(RULE)
    print("  WHY F1 FELL FROM ~0.63 TO ~0.19")
    print(RULE)
    df = load_vae()
    prev = df.groupby("isin")["enc_y"].shift(1)
    df["is_downgrade"] = ((df["enc_y"] > prev) & (df["is_rating_change"] == 1)).astype(int)

    eng = DeadzoneEProcess(0.20, delta=0.75, lam=0.50)
    res = eng.run_sequential_test(df, z_col="pit", cooldown_months=12)
    n_al = int(res["is_alarm"].sum())
    print(f"\nOne fixed alarm set from the corrected e-process: {n_al:,} alarms "
          f"on {len(df):,} bond-months.")
    print("Both metrics below score EXACTLY these alarms. Nothing about the "
          "method changes between rows.\n")

    rnd = random_alarms(res, n_al, seed=1)

    print(f"{'metric':<46}{'P (%)':>9}{'R (%)':>9}{'F1':>9}   {'chance F1':>10}")
    print("-" * 92)

    m = old_regime_metric(res)
    r = old_regime_metric(rnd, alarm_col="rand_alarm")
    print(f"{'OLD: next-transition regime, no time limit':<46}"
          f"{m['P']:>9.2f}{m['R']:>9.2f}{m['F1']:>9.4f}   {r['F1']:>10.4f}")
    print(f"{'     ^ chance baseline (random alarms)':<46}"
          f"{r['P']:>9.2f}{r['R']:>9.2f}{r['F1']:>9.4f}")

    for H in (12, 24):
        m = windowed_metric(res, H)
        r = windowed_metric(rnd, H, alarm_col="rand_alarm")
        print(f"{f'NEW: any transition within {H}m (both sides)':<46}"
              f"{m['P']:>9.2f}{m['R']:>9.2f}{m['F1']:>9.4f}   {r['F1']:>10.4f}")
        print(f"{'     ^ chance baseline (random alarms)':<46}"
              f"{r['P']:>9.2f}{r['R']:>9.2f}{r['F1']:>9.4f}")

    # --- what the unbounded metric is actually measuring -------------------
    dates = pd.to_datetime(df["dates"]).to_numpy()
    ids = df["isin"].to_numpy()
    is_chg = df["is_rating_change"].to_numpy().astype(int)
    is_down = df["is_downgrade"].to_numpy().astype(int)
    ch_by = {}
    for p in np.flatnonzero(is_chg == 1):
        ch_by.setdefault(ids[p], []).append(p)
    nxt_is_down, has_next = 0, 0
    for i in range(len(df)):
        nxt = [q for q in ch_by.get(ids[i], ()) if dates[q] > dates[i]]
        if nxt:
            has_next += 1
            nxt_is_down += is_down[nxt[0]]
    print(f"\n{RULE}\n  WHAT THE UNBOUNDED METRIC WAS MEASURING\n{RULE}")
    print(f"  Bond-months that ever see another transition: "
          f"{100*has_next/len(df):.2f}%")
    print(f"  Of those, share whose next transition is a downgrade: "
          f"{100*nxt_is_down/max(1,has_next):.2f}%")
    print(f"  => an alarm placed at random on such a month scores that same "
          f"precision by construction.")
    print(f"  Median bond history: {df.groupby('isin').size().median():.0f} months, "
          f"so 'no time limit' can mean a decade of lead time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
