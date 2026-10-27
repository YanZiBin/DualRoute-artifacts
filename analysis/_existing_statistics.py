# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
"""Calculation functions extracted unchanged from the retained analysis source."""
import csv
import math
import statistics as st
from scipy.stats import t
DEVICES = {"High": "high", "Mid": "midrange", "Low": "low"}

def csvout(path, data):
    if not data:
        raise ValueError(f"Empty output: {path}")
    keys = list(dict.fromkeys(k for row in data for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def mean_ci(values):
    n = len(values)
    mean, sd = st.mean(values), st.stdev(values)
    half = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "df": n - 1,
            "ci95_low": mean - half, "ci95_high": mean + half}


def welch(a, b):
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff = st.mean(a) - st.mean(b)
    half = float(t.ppf(0.975, df)) * se
    return {"difference": diff, "se": se, "df": df,
            "ci95_low": diff - half, "ci95_high": diff + half}


def mixtures(observations, out, scope):
    result, thresholds, strata = [], [], []
    for device in DEVICES:
        for metric, unit in [("net_mah_request", "mAh/request"), ("active_seconds_request", "s/request")]:
            means = {}
            for group in ["Launch", "Workflow"]:
                for mode in ["Routed", "Cloud-only"]:
                    a = [x[metric] for x in observations if x["device"] == device and x["workload"] == group and x["mode"] == mode]
                    means[group, mode] = st.mean(a)
                strata.append({"device": device, "metric": metric, "unit": unit, "workload": group,
                               "routed_mean": means[group, "Routed"], "cloud_mean": means[group, "Cloud-only"],
                               "difference": means[group, "Routed"] - means[group, "Cloud-only"]})
            dl = means["Launch", "Routed"] - means["Launch", "Cloud-only"]
            dw = means["Workflow", "Routed"] - means["Workflow", "Cloud-only"]
            for w in scope["mixture_launch_weights"]:
                result.append({"device": device, "metric": metric, "unit": unit,
                               "launch_weight": w, "workflow_weight": 1 - w,
                               "routed_interpolated": w * means["Launch", "Routed"] + (1 - w) * means["Workflow", "Routed"],
                               "cloud_interpolated": w * means["Launch", "Cloud-only"] + (1 - w) * means["Workflow", "Cloud-only"],
                               "difference_routed_minus_cloud": w * dl + (1 - w) * dw,
                               "status": "conditional interpolation; no current mixed measurement"})
            thresholds.append({"device": device, "metric": metric, "unit": unit,
                               "launch_weight_zero_crossing": dw / (dw - dl),
                               "status": "point estimate only; not measured or validated deployment threshold"})
    csvout(out / "C_category_cost_differences.csv", strata)
    csvout(out / "C_mixture_25_50_75.csv", result)
    csvout(out / "C_conditional_zero_crossings.csv", thresholds)
    return result, thresholds

