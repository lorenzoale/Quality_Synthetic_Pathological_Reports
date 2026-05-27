from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from sample_values import sample_receptor_value, sample_value


DEFAULT_BIN_COLUMNS = (
    "taille_tumor_0",
    "ganglions_preleves",
    "ganglions_atteints",
    "ki67_tumor_0",
)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    value_str = str(value).strip().lower()
    return value_str in {"", "unknown", "nan", "none"}


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _sample_bin_value(col_name: str, raw_value: object, other: Optional[int] = None) -> object:
    """
    Sample a concrete value from a binned/raw value.

    Rules:
    - unknown / missing values are preserved as-is
    - already numeric single values are preserved
    - known bins are sampled through sample_values.sample_value
    """
    if _is_missing(raw_value):
        return raw_value

    raw_str = str(raw_value).strip()

    # If the value is already a plain integer, keep it unchanged.
    if raw_str.isdigit():
        return int(raw_str)

    # Delegate recognized bins/ranges to the provided sampler.
    try:
        return sample_value(col_name, raw_str, other=other)
    except Exception:
        # If the bin is not supported, keep the original value.
        return raw_value


def _sample_receptor_percent(status_value: object, receptor: str) -> object:
    """
    Sample RE/RP percentage only when the receptor status is positive.
    Otherwise preserve missing values or return None for non-positive cases.
    """
    if _is_missing(status_value):
        return 0

    status = _normalize_text(status_value)
    if status in {"positif", "positifs", "positive", "positives", "1", "oui", "yes"}:
        return int(sample_receptor_value(receptor))
    else:
        return 0
    return None


def sample_row(
    row: pd.Series,
    bin_columns: Iterable[str] = DEFAULT_BIN_COLUMNS,
    add_receptor_percentages: bool = True,
) -> pd.Series:
    """
    Sample one dataframe row and return a new sampled row.

    Sampling logic:
    - taille_tumor_0, ganglions_preleves, ganglions_atteints, ki67_tumor_0:
      sample concrete values from bins when applicable.
    - ganglions_atteints is constrained to be <= ganglions_preleves.
    - optional columns re_perc / rp_perc are added when RE/RP are positive.
    """
    sampled = row.copy()

    # Sample ganglions_preleves first because ganglions_atteints depends on it.
    sampled_nodes_examined: Optional[int] = None
    if "ganglions_preleves" in sampled.index:
        sampled_value = _sample_bin_value("ganglions_preleves", sampled["ganglions_preleves"])
        sampled["ganglions_preleves"] = sampled_value
        if isinstance(sampled_value, (int, np.integer)):
            sampled_nodes_examined = int(sampled_value)

    # Sample the remaining bin columns.
    for col in bin_columns:
        if col not in sampled.index or col == "ganglions_preleves":
            continue

        if col == "ganglions_atteints":
            sampled[col] = _sample_bin_value(col, sampled[col], other=sampled_nodes_examined)
        else:
            sampled[col] = _sample_bin_value(col, sampled[col])

    # Safety check in case a pre-existing concrete value exceeded the sampled count.
    if (
        "ganglions_preleves" in sampled.index
        and "ganglions_atteints" in sampled.index
        and isinstance(sampled.get("ganglions_preleves"), (int, np.integer))
        and isinstance(sampled.get("ganglions_atteints"), (int, np.integer))
    ):
        sampled["ganglions_atteints"] = min(
            int(sampled["ganglions_atteints"]),
            int(sampled["ganglions_preleves"]),
        )

    if add_receptor_percentages:
        if "re_tumor_0" in sampled.index:
            sampled["re_perc"] = _sample_receptor_percent(sampled["re_tumor_0"], "RE")
        if "rp_tumor_0" in sampled.index:
            sampled["rp_perc"] = _sample_receptor_percent(sampled["rp_tumor_0"], "RP")

    return sampled


def sample_dataframe(
    df: pd.DataFrame,
    seed: Optional[int] = None,
    bin_columns: Iterable[str] = DEFAULT_BIN_COLUMNS,
    add_receptor_percentages: bool = True,
) -> pd.DataFrame:
    """
    Return a new DataFrame where binned pathology values are sampled row by row.

    Parameters
    ----------
    df:
        Input DataFrame.
    seed:
        Optional NumPy seed for reproducible sampling.
    bin_columns:
        Columns to resolve from bins/ranges into concrete values.
    add_receptor_percentages:
        If True, create/update re_perc and rp_perc when receptor status is positive.
    """
    if seed is not None:
        np.random.seed(seed)

    sampled_df = df.apply(
        sample_row,
        axis=1,
        bin_columns=tuple(bin_columns),
        add_receptor_percentages=add_receptor_percentages,
    )

    return sampled_df.reset_index(drop=True)


