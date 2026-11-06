"""Generate a synthetic poultry-farm dataset (sensor + health-log data).

Each row is one flock on one day. A latent "disease pressure" builds up in some
flocks over several days; early symptoms (feed/water change, cough score,
droppings, slight mortality rise) appear gradually, with noise, so the label
is learnable but not trivially separable.

Run:  python data_gen.py
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_FLOCKS = 60
N_DAYS = 70


def simulate_flock(flock_id: int) -> pd.DataFrame:
    size = int(RNG.integers(3000, 12000))
    start_age = int(RNG.integers(5, 30))
    house = RNG.choice(["Shed-A", "Shed-B", "Shed-C", "Shed-D"])
    vacc_gap = int(RNG.integers(3, 25))  # days since last vaccination at day 0
    sick_flock = RNG.random() < 0.45
    onset = int(RNG.integers(15, N_DAYS - 12)) if sick_flock else None
    ramp = int(RNG.integers(5, 10))  # days for symptoms to build up

    rows = []
    for d in range(N_DAYS):
        age = start_age + d
        pressure = 0.0
        if onset is not None:
            if d < onset - ramp:
                pressure = 0.0
            elif d <= onset + 6:  # build-up, then plateau at peak
                pressure = min(1.0, (d - (onset - ramp)) / ramp)
            else:  # treatment / recovery: symptoms fade over ~6 days
                pressure = max(0.0, 1.0 - (d - (onset + 6)) / 6)
        p = pressure
        # Flag the build-up and active phase (symptoms clearly above baseline);
        # healthy-looking recovery days are not flagged.
        label = 1 if (onset is not None and d <= onset + 6 and p >= 0.3) else 0

        temp = RNG.normal(27 - 0.12 * (age / 7), 0.9) + 0.9 * p
        hum = RNG.normal(62, 5) + 5 * p
        nh3 = max(0, RNG.normal(11, 3) + 6 * p + 0.04 * age)
        feed = RNG.normal(45 + 1.6 * age ** 0.9 / 3, 3) * (1 - 0.16 * p)
        water = RNG.normal(1.9 * feed + 8, 6) * (1 + 0.13 * p)
        mort = max(0.0, RNG.normal(0.04, 0.03) + 0.14 * p ** 1.5)
        cough = float(np.clip(RNG.normal(0.5, 0.5) + 2.6 * p, 0, 5))
        dropp = float(np.clip(RNG.normal(4, 2.5) + 20 * p, 0, 100))
        lethargy = float(np.clip(RNG.normal(0.6, 0.6) + 2.2 * p, 0, 5))
        egg_or_gain = RNG.normal(100, 3.5) * (1 - 0.06 * p)  # % of expected performance
        vacc_days = vacc_gap + d if d < 30 else (d - 30) % 25  # revaccination cycle

        rows.append(
            dict(
                flock_id=f"F{flock_id:03d}",
                shed=house,
                day=d,
                flock_size=size,
                bird_age_days=age,
                house_temp_c=temp,
                humidity_pct=hum,
                ammonia_ppm=nh3,
                feed_g_per_bird=feed,
                water_ml_per_bird=water,
                mortality_pct=mort,
                cough_sneeze_score=cough,
                abnormal_droppings_pct=dropp,
                lethargy_score=lethargy,
                performance_index=egg_or_gain,
                days_since_vaccination=vacc_days,
                early_disease=label,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.concat([simulate_flock(i) for i in range(1, N_FLOCKS + 1)], ignore_index=True)

    # Realistic mess: missing sensor readings, a few faulty/out-of-range values, label noise.
    for col, frac in [
        ("house_temp_c", 0.03), ("humidity_pct", 0.03), ("ammonia_ppm", 0.04),
        ("feed_g_per_bird", 0.02), ("water_ml_per_bird", 0.02),
        ("cough_sneeze_score", 0.05), ("abnormal_droppings_pct", 0.05),
    ]:
        df.loc[RNG.random(len(df)) < frac, col] = np.nan
    bad = RNG.random(len(df)) < 0.004
    df.loc[bad, "house_temp_c"] = RNG.choice([-40.0, 85.0, 999.0], bad.sum())
    flip = RNG.random(len(df)) < 0.025
    df.loc[flip, "early_disease"] = 1 - df.loc[flip, "early_disease"]

    df.round(3).to_csv("data/poultry_flock_data.csv", index=False)
    print(df.shape, "rows x cols")
    print("Positive rate:", round(df["early_disease"].mean(), 3))


if __name__ == "__main__":
    main()
