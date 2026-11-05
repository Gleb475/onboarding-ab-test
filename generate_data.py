from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def generate():
    rng = np.random.default_rng(20261105)
    users = []
    events = []
    for user_id in range(1, 24001):
        group = "B" if rng.random() < 0.5 else "A"
        platform = "android" if rng.random() < 0.58 else "ios"
        assigned = pd.Timestamp("2026-08-01") + pd.Timedelta(seconds=int(rng.integers(0, 14 * 86400)))
        users.append((user_id, group, assigned, platform))
        events.append((user_id, "onboarding_opened", assigned))
        if rng.random() < 0.82:
            started = assigned + pd.Timedelta(hours=float(rng.uniform(0.1, 12)))
            events.append((user_id, "identity_started", started))
            if rng.random() < 0.78:
                verified = started + pd.Timedelta(hours=float(rng.uniform(0.1, 24)))
                events.append((user_id, "identity_passed", verified))
                probability = 0.36 + (0.035 if group == "B" else 0)
                if rng.random() < probability:
                    activated = verified + pd.Timedelta(hours=float(rng.uniform(0.1, 144)))
                    events.append((user_id, "activated", activated))
        if rng.random() < (0.032 if group == "B" else 0.03):
            error_at = assigned + pd.Timedelta(hours=float(rng.uniform(0, 168)))
            events.append((user_id, "application_error", error_at))
    assignment = pd.DataFrame(users, columns=["user_id", "variant", "assigned_at", "platform"])
    event_log = pd.DataFrame(events, columns=["user_id", "event_name", "event_at"])
    duplicates = event_log.sample(frac=0.01, random_state=54)
    event_log = pd.concat([event_log, duplicates], ignore_index=True).sort_values("event_at")
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    assignment.to_csv(folder / "assignments.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    event_log.to_csv(folder / "events.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    generate()
