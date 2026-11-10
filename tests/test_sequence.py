from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

pytest.importorskip("sklearn")

from apex.ml.dataset import build_dataset
from apex.ml.sequence import SEQ_FEATURES, WINDOW, GRUClassifier, build_sequences
from test_ml import GROUPS, universe


def test_sequences_align_with_rows_and_ignore_the_future():
    full = universe()
    cut = date(2025, 3, 31)
    truncated = {k: v.filter(pl.col("ts").dt.date() <= cut) for k, v in full.items()}
    data_full, data_cut = build_dataset(GROUPS, full), build_dataset(GROUPS, truncated)
    seq_full = build_sequences(GROUPS, full, data_full)
    seq_cut = build_sequences(GROUPS, truncated, data_cut)
    assert seq_full.shape == (data_full.height, WINDOW, len(SEQ_FEATURES))
    keep = (data_full["date"] <= cut).to_numpy()
    assert keep.sum() == data_cut.height > 1000
    np.testing.assert_array_equal(seq_full[keep], seq_cut)
    # The last step of each window is the row's own bar: its return matches ret_1.
    full_window = ~np.isnan(seq_full[:, -1, 0])
    assert full_window.sum() > 2000
    np.testing.assert_allclose(seq_full[full_window, -1, 0],
                               data_full["ret_1"].to_numpy()[full_window], atol=1e-6)


def test_gru_learns_an_obvious_pattern():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (600, 20, 3)).astype(np.float32)
    y = (x[:, -5:, 0].mean(axis=1) > 0).astype(np.float64)
    model = GRUClassifier(hidden=16, epochs=30, lr=1e-2, batch=64).fit(x[:500], y[:500])
    p = model.predict(x[500:])
    assert ((p > 0.5) == (y[500:] > 0.5)).mean() > 0.85
    assert p.min() >= 0 and p.max() <= 1
