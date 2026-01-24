from anomlib.datasets.kaggle_energy.load import load_train
from anomlib.detectors.energy_timeseries import EnergyTimeSeriesDetector

df = load_train("data/kaggle/train.csv")

det = EnergyTimeSeriesDetector(direction="low", threshold=3.5, min_duration="6H").fit(df)
events, scores = det.detect(df)

print("num events:", len(events))
print("sample event:", events[0] if events else None)

# quick sanity: how many labeled anomaly points fall within predicted events?
if "anomaly" in df.columns and len(events) > 0:
    df = df.copy()
    df["pred"] = 0
    for e in events:
        m = (df["building_id"] == e.entity_id) & (df["timestamp"] >= e.start) & (df["timestamp"] <= e.end)
        df.loc[m, "pred"] = 1

    tp = ((df["pred"] == 1) & (df["anomaly"] == 1)).sum()
    fp = ((df["pred"] == 1) & (df["anomaly"] == 0)).sum()
    fn = ((df["pred"] == 0) & (df["anomaly"] == 1)).sum()
    print("tp fp fn:", tp, fp, fn)
