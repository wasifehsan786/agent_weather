import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from agentweather import features, train  # noqa: E402
from tests.test_features import make_raw  # noqa: E402


@pytest.fixture(scope="module")
def feats():
    return features.build_features(make_raw(hours=24 * 90))


def test_feature_columns_have_no_targets_or_flags(feats):
    cols = train.feature_columns(feats)
    assert "temp_c" in cols and "hour_sin" in cols and "temp_c_lag24h" in cols
    assert not any(c.startswith("target_") for c in cols)
    assert not {"ts", "city", "has_target", "is_complete", "n_imputed", "wind_deg"} & set(cols)


def test_time_split_has_gap_and_no_overlap(feats):
    data = train.usable_rows(feats, train.feature_columns(feats))
    tr, va = train.split_time(data, val_days=10, gap_h=24)
    assert tr["ts"].max() < va["ts"].min()
    assert va["ts"].min() - tr["ts"].max() >= pd.Timedelta(hours=24)
    assert len(tr) > len(va) > 0


def test_train_all_scores_every_metric_and_model(feats):
    results, models, val, cols = train.train_all(feats, val_days=10)
    assert set(results["metric"]) == set(train.METRICS)
    assert set(results["model"]) == {"naive", "ridge", "random_forest"}
    assert len(results) == 3 * len(train.METRICS)
    assert (results["rmse"] >= 0).all()
    # temperature in the test data is a clean daily cycle, so models must beat the naive value
    temp = results[results["metric"] == "temp_c"].set_index("model")
    assert temp.loc["ridge", "rmse"] <= temp.loc["naive", "rmse"]


def test_rain_probability_is_a_probability(feats):
    results, models, val, cols = train.train_all(feats, val_days=10)
    for family in ("ridge", "random_forest"):
        p = train.predict(models[("rain", family)], val[cols], "rain")
        assert p.min() >= 0 and p.max() <= 1


def test_predictions_respect_limits():
    p = train.clip(np.array([-5.0, 50.0, 140.0]), "humidity_pct")
    assert p.min() == 0 and p.max() == 100


def test_best_per_metric_ignores_naive():
    res = pd.DataFrame([
        {"metric": "a", "model": "naive", "rmse": 1.0},
        {"metric": "a", "model": "ridge", "rmse": 3.0},
        {"metric": "a", "model": "random_forest", "rmse": 2.0},
    ])
    assert train.best_per_metric(res) == {"a": "random_forest"}


def test_replace_only_when_clearly_better():
    assert train.should_replace(1.0, None) is True
    assert train.should_replace(0.9, 1.0) is True
    assert train.should_replace(1.0, 1.0) is False
    assert train.should_replace(0.999, 1.0) is False   # 0.1% better is noise
    assert train.should_replace(1.1, 1.0) is False


class _FakeEntry:
    def __init__(self, store, name, metrics):
        self.store, self.name, self.metrics = store, name, metrics

    def save(self, folder):
        import shutil
        import tempfile
        keep = tempfile.mkdtemp()
        shutil.copytree(folder, keep, dirs_exist_ok=True)
        self.path = keep
        self.store.setdefault(self.name, []).append(self)

    def download(self):
        return self.path


class _FakeRegistry:
    """Stands in for the Hopsworks model registry in tests."""

    def __init__(self):
        self.store = {}
        registry = self

        class _Sk:
            def create_model(self, name, metrics, description, input_example):
                return _FakeEntry(registry.store, name, metrics)

        self.sklearn = _Sk()

    def get_best_model(self, name, metric, direction):
        entries = self.store.get(name)
        if not entries:
            return None
        return min(entries, key=lambda e: e.metrics[metric])


def test_registry_first_run_registers_second_run_keeps(feats):
    results, models, val, cols = train.train_all(feats, val_days=10)
    mr = _FakeRegistry()
    first = train.register_all(mr, results, models, val, cols)
    assert all(d["registered"] for d in first)
    assert len(mr.store) == len(train.METRICS)
    # the same models again are not strictly better, so nothing new is saved
    second = train.register_all(mr, results, models, val, cols)
    assert not any(d["registered"] for d in second)
    assert all(len(v) == 1 for v in mr.store.values())
