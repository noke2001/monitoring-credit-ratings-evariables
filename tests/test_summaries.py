"""Certification of the candidate summary functions of Section 4.3.

Checks each construction against a closed form where one exists, and the three
scoring rules against the defining property of a proper rule: the expected
score is optimised by reporting the truth.

    python tests/test_summaries.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.summaries import (SUMMARIES, brier, directional_deviation,  # noqa: E402
                           log_score, ranked_probability,
                           transition_risk_ratio)
FAIL=[]
def ck(n,c,d=""):
    print(f"  [{'ok ' if c else 'FAIL'}] {n}{'  '+d if d else ''}");  FAIL.append(n) if not c else None

# perfect forecast: every loss minimal, DD zero, TRR minimal
P = np.eye(6); k = np.arange(6)
ck("perfect forecast -> DD = 0", np.allclose(directional_deviation(P,k),0))
ck("perfect forecast -> log score = 0", np.allclose(log_score(P,k),0))
ck("perfect forecast -> RPS = 0", np.allclose(ranked_probability(P,k),0))
ck("perfect forecast -> Brier = -1 (minimum)", np.allclose(brier(P,k),-1))
# Brier reward form in [-1,1]; loss = -reward
p = np.array([[.1,.2,.3,.2,.1,.1]]); 
for kk in range(6):
    r = 2*p[0,kk] - (p**2).sum()
    ck(f"Brier loss == -reward (class {kk})", np.isclose(brier(p,[kk])[0], -r))
# RPS is order-aware: being wrong by 5 notches costs more than by 1
p2 = np.zeros((1,6)); p2[0,0]=1.0
ck("RPS penalises distance", ranked_probability(p2,[5])[0] > ranked_probability(p2,[1])[0],
   f"{ranked_probability(p2,[5])[0]:.1f} > {ranked_probability(p2,[1])[0]:.1f}")
ck("Brier does NOT penalise distance",
   np.isclose(brier(p2,[5])[0], brier(p2,[1])[0]), "order-blind, as documented")
ck("log score does NOT penalise distance",
   np.isclose(log_score(p2,[5])[0], log_score(p2,[1])[0]))
# TRR direction
pd_ = np.array([[0,0,0,.1,.4,.5]])   # mass below (worse than) class 2
ck("TRR > 0 when downgrade mass dominates", transition_risk_ratio(pd_,[2])[0] > 0)
pu = np.array([[.5,.4,.1,0,0,0]])
ck("TRR < 0 when upgrade mass dominates", transition_risk_ratio(pu,[3])[0] < 0)
# propriety: expected score minimised at the truth (MC)
rng=np.random.default_rng(0); q=np.array([.05,.15,.4,.25,.1,.05])
y=rng.choice(6,200000,p=q)
Pq=np.tile(q,(len(y),1))
for nm,fn in [("Brier",brier),("RPS",ranked_probability),("log",log_score)]:
    truth=np.nanmean(fn(Pq,y))
    worse=[]
    for _ in range(12):
        r_=rng.dirichlet(np.ones(6)*3)
        worse.append(np.nanmean(fn(np.tile(r_,(len(y),1)),y)))
    ck(f"{nm} is proper (truth beats 12 random reports)", truth < min(worse),
       f"{truth:.4f} < {min(worse):.4f}")
# --- the modified TRR (dead-zone) -------------------------------------------
conf = np.array([[0.0, 0.0, 0.995, 0.003, 0.001, 0.001]])   # P(stay) = 0.995
unsure = np.array([[0.0, 0.0, 0.50, 0.30, 0.10, 0.10]])     # P(stay) = 0.50
ck("deadzone neutralises a confident row",
   transition_risk_ratio(conf, [2], deadzone=0.99)[0] == 0.0)
ck("deadzone leaves an unsure row alone",
   np.isclose(transition_risk_ratio(unsure, [2], deadzone=0.99)[0],
              transition_risk_ratio(unsure, [2])[0]))
ck("deadzone neutral is 1 in the raw ratio form",
   transition_risk_ratio(conf, [2], log=False, deadzone=0.99)[0] == 1.0)
ck("threshold above P(stay) is a no-op",
   np.isclose(transition_risk_ratio(conf, [2], deadzone=0.999)[0],
              transition_risk_ratio(conf, [2])[0]))
# the neutral value must sit BETWEEN downgrade-leaning and upgrade-leaning rows,
# which is what parks dead-zoned bonds mid-rank rather than at an edge
dn_lean = np.array([[0.0, 0.0, 0.90, 0.06, 0.02, 0.02]])
up_lean = np.array([[0.02, 0.06, 0.90, 0.0, 0.01, 0.01]])
ck("neutral sits between the two directions",
   transition_risk_ratio(up_lean, [2])[0] < 0.0 < transition_risk_ratio(dn_lean, [2])[0],
   f"{transition_risk_ratio(up_lean,[2])[0]:.2f} < 0 < {transition_risk_ratio(dn_lean,[2])[0]:.2f}")
ck("both dead-zone variants are registered",
   {"log_TRR_dz_0.9", "log_TRR_dz_0.9999"} <= set(SUMMARIES))

# NaN handling
ck("non-finite reference class -> NaN", np.isnan(directional_deviation(p,[np.nan])[0]))
print()
if FAIL:
    print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL))
    sys.exit(1)
print("all summary-function checks passed")
