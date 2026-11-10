"""Small helpers that keep pandas frames behaving the same everywhere in APEX.

Convention: missing values are NaN inside frames and None in row dicts, so callers can
keep writing `value is None` checks. The `ts` column is a UTC timestamp.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def records(frame: pd.DataFrame) -> list[dict]:
    """Rows as plain dicts, with NaN/NaT turned into None and timestamps as datetimes."""
    if frame.empty:
        return []
    cols = list(frame.columns)
    arrays = []
    for c in cols:
        s = frame[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            arrays.append([None if pd.isna(v) else v.to_pydatetime() for v in s])
        elif pd.api.types.is_float_dtype(s):
            arrays.append([None if math.isnan(v) else float(v) for v in s.to_numpy()])
        else:
            arrays.append([None if _isna(v) else v for v in s.tolist()])
    return [dict(zip(cols, vals)) for vals in zip(*arrays)]


def _isna(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT
    except TypeError:
        return False


def dates(ts: pd.Series) -> pd.Series:
    """Calendar dates (python `date` objects) of a UTC timestamp column."""
    return pd.Series([v.date() for v in ts], index=ts.index, dtype=object)


def matrix(frame: pd.DataFrame, columns) -> np.ndarray:
    return frame[list(columns)].to_numpy(dtype=np.float64, na_value=np.nan)
