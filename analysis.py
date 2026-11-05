from pathlib import Path
import json
import math
import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import norm, chisquare
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CUTOFF = pd.Timestamp("2026-08-22 23:59:59")


def load_metrics():
    assignments = pd.read_csv(ROOT / "data/assignments.csv", parse_dates=["assigned_at"])
    events = pd.read_csv(ROOT / "data/events.csv", parse_dates=["event_at"])
    if assignments.isna().any().any() or events.isna().any().any():
        raise ValueError("Missing values")
    if assignments["user_id"].duplicated().any() or set(assignments["variant"]) != {"A", "B"}:
        raise ValueError("Invalid assignments")
    if (assignments["assigned_at"] + pd.Timedelta(days=7) > CUTOFF).any():
        raise ValueError("Observation window is incomplete")
    if not events["user_id"].isin(assignments["user_id"]).all():
        raise ValueError("Unknown users in events")
    timeline = events.merge(assignments[["user_id", "assigned_at"]], on="user_id", validate="many_to_one")
    if (timeline["event_at"] < timeline["assigned_at"]).any() or (timeline["event_at"] > CUTOFF).any():
        raise ValueError("Invalid event timestamps")
    duplicate_events = int(events.duplicated().sum())
    events = events.drop_duplicates()
    with sqlite3.connect(":memory:") as connection:
        assignments.to_sql("assignments", connection, index=False)
        events.to_sql("events", connection, index=False)
        metrics = pd.read_sql_query((ROOT / "sql/user_metrics.sql").read_text(), connection)
    return metrics, {"users": len(assignments), "raw_events": len(timeline), "duplicate_events": duplicate_events}


def compare_binary(metrics, column):
    grouped = metrics.groupby("variant")[column].agg(["sum", "count"])
    successes = grouped["sum"].to_numpy(dtype=float)
    samples = grouped["count"].to_numpy(dtype=float)
    rates = successes / samples
    difference = float(rates[1] - rates[0])
    pooled = successes.sum() / samples.sum()
    pooled_se = math.sqrt(pooled * (1 - pooled) * (1 / samples[0] + 1 / samples[1]))
    se = math.sqrt(float(np.sum(rates * (1 - rates) / samples)))
    if pooled_se == 0:
        raise ValueError("Metric has no variance")
    return {
        "a_users": int(samples[0]), "b_users": int(samples[1]),
        "a_successes": int(successes[0]), "b_successes": int(successes[1]),
        "a_rate": float(rates[0]), "b_rate": float(rates[1]),
        "difference": difference, "relative_uplift": float(difference / rates[0]) if rates[0] else None,
        "p_value": float(2 * norm.sf(abs(difference / pooled_se))),
        "ci_low": difference - norm.ppf(0.975) * se,
        "ci_high": difference + norm.ppf(0.975) * se,
        "one_sided_upper": difference + norm.ppf(0.95) * se
    }


def planned_sample(baseline=0.23, mde=0.02):
    second = baseline + mde
    mean = (baseline + second) / 2
    value = (norm.ppf(0.975) * math.sqrt(2 * mean * (1 - mean))
             + norm.ppf(0.8) * math.sqrt(baseline * (1 - baseline) + second * (1 - second))) ** 2 / mde ** 2
    return math.ceil(value)


def run():
    folder = ROOT / "reports"
    folder.mkdir(exist_ok=True)
    metrics, quality = load_metrics()
    metrics.to_csv(folder / "user_metrics.csv", index=False)
    primary = compare_binary(metrics, "activated_7d")
    guardrail = compare_binary(metrics, "had_error_7d")
    counts = metrics["variant"].value_counts().reindex(["A", "B"])
    srm = float(chisquare(counts, f_exp=[len(metrics) / 2] * 2).pvalue)
    funnel = metrics.groupby("variant").agg(users=("user_id", "count"),
        identity_started=("identity_started", "sum"), identity_passed=("identity_passed", "sum"),
        activated_7d=("activated_7d", "sum"))
    funnel.to_csv(folder / "funnel.csv")
    balance = pd.crosstab(metrics["platform"], metrics["variant"], normalize="columns")
    balance.to_csv(folder / "platform_balance.csv")
    passed = bool(srm >= 0.01 and primary["ci_low"] > 0 and guardrail["one_sided_upper"] < 0.01)
    report = {"quality": quality, "planned_users_per_group": planned_sample(), "srm_p_value": srm,
              "primary": primary, "guardrail": guardrail, "decision_passed": passed}
    (folder / "statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in [(axes[0], primary, "Activation within 7 days"),
                           (axes[1], guardrail, "Users with application errors")]:
        ax.bar(["A", "B"], [100 * key["a_rate"], 100 * key["b_rate"]], color=["#71869a", "#3a7d69"], width=0.55)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("Users, %")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(folder / "metrics.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
