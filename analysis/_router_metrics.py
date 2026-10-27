# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
"""Calculation functions extracted unchanged from the retained analysis source."""
import numpy as np

def scores(gold, pred):
    confusion = [[int(np.sum((gold == y) & (pred == p))) for p in [0, 1, 2, -1]] for y in [0, 1, 2]]
    per_class = {}
    f1s = []
    for y in [0, 1, 2]:
        tp = int(np.sum((gold == y) & (pred == y)))
        n_gold = int(np.sum(gold == y))
        n_pred = int(np.sum(pred == y))
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gold if n_gold else 0.0
        f1 = 2 * tp / (n_gold + n_pred) if n_gold + n_pred else 0.0
        per_class[str(y)] = dict(precision=precision, recall=recall, f1=f1, support=n_gold)
        f1s.append(f1)
    return dict(n=len(gold), correct=int(np.sum(gold == pred)), accuracy=float(np.mean(gold == pred)),
                macro_f1=float(np.mean(f1s)), invalid=int(np.sum(pred == -1)),
                confusion_rows_gold012_columns_pred012invalid=confusion, per_class=per_class)


def boot_metric(y, p):
    # y: B x N; p: B x N x methods
    acc = np.mean(y[:, :, None] == p, axis=1)
    f1 = np.zeros_like(acc)
    for cls in [0, 1, 2]:
        tp = np.sum((y[:, :, None] == cls) & (p == cls), axis=1)
        denom = np.sum(y == cls, axis=1)[:, None] + np.sum(p == cls, axis=1)
        f1 += np.divide(2 * tp, denom, out=np.zeros_like(acc), where=denom != 0) / 3
    return acc, f1


def interval(x):
    return [float(v) for v in np.quantile(x, [0.025, 0.975])]

