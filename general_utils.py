"""General data, statistics, and window utilities for the dataset notebook."""

import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist


# Copied from forecast_utils.py because it is cleaner and more complete than the old notebook loader.
def prepare_pandas_data(
    df,
    users_dim=1,
    date_col=None,
    dates=None,
    names=None,
    drop=None,
    aggr=None,
    aggr_period=None,
):
    """Return a clean time-series dataframe with series in columns."""
    df = df.copy()

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

    if users_dim == 0:
        df = df.T

    if dates is not None:
        df.index = pd.to_datetime(dates)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.RangeIndex(len(df))

    if drop is not None:
        if isinstance(drop, str):
            drop = drop.split(";")

        drop_labels = []
        for item in drop:
            if item in df.columns:
                drop_labels.append(item)
            elif str(item) in df.columns:
                drop_labels.append(str(item))
            elif isinstance(item, (int, np.integer)) or str(item).isdigit():
                drop_labels.append(df.columns[int(item)])
            else:
                drop_labels.append(item)

        df = df.drop(columns=drop_labels)

    if names is None:
        df.columns = [f"series_{i}" for i in range(df.shape[1])]
    else:
        df.columns = names

    if aggr is not None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Aggregation requires a DatetimeIndex. Provide `date_col` or `dates`.")
        if aggr_period is None:
            raise ValueError("If `aggr` is set, `aggr_period` must be provided.")

        if aggr == "sum":
            df = df.resample(aggr_period).sum()
        elif aggr == "mean":
            df = df.resample(aggr_period).mean()
        elif aggr == "last":
            df = df.resample(aggr_period).last()
        elif aggr == "first":
            df = df.resample(aggr_period).first()
        elif aggr == "asfreq":
            df = df.asfreq(aggr_period)
        else:
            raise ValueError("`aggr` must be one of: None, 'sum', 'mean', 'last', 'first', 'asfreq'.")

    return df


def normalize_from_lookback(x, y=None, eps=1e-8):
    """Normalize arrays using statistics from the lookback only."""
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sigma = np.nanstd(x)
    sigma = sigma if sigma > eps else 1.0

    x_norm = (x - mu) / sigma
    y_norm = None if y is None else (np.asarray(y, dtype=float) - mu) / sigma
    return x_norm, y_norm, mu, sigma


def denormalize(x, mu, sigma):
    """Map normalized values back to the original scale."""
    return np.asarray(x, dtype=float) * sigma + mu


def check_window_bounds(data, t, lookback, horizon):
    """Validate that a target index supports the requested forecast window."""
    n = len(data)
    if lookback < 1:
        raise ValueError("lookback must be >= 1.")
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    if t - lookback + 1 < 0:
        raise ValueError(f"t={t} is too small for lookback={lookback}. Minimum t is {lookback - 1}.")
    if t + horizon >= n:
        raise ValueError(f"t={t} is too large for horizon={horizon}. Maximum t is {n - horizon - 1}.")


def fetch_window(data, series_i, t, lookback, horizon, normalize=False):
    """Extract target lookback and future horizon slices from a dataframe."""
    check_window_bounds(data, t, lookback, horizon)

    x0 = t - lookback + 1
    x1 = t + 1
    y0 = t + 1
    y1 = t + horizon + 1

    x = data.iloc[x0:x1, series_i].to_numpy(dtype=float)
    y = data.iloc[y0:y1, series_i].to_numpy(dtype=float)

    mu, sigma = None, None
    if normalize:
        x, y, mu, sigma = normalize_from_lookback(x, y)

    return {
        "series_i": series_i,
        "series_name": data.columns[series_i],
        "t": t,
        "lookback": x,
        "horizon": y,
        "dates_lookback": data.index[x0:x1],
        "dates_horizon": data.index[y0:y1],
        "mu": mu,
        "sigma": sigma,
        "normalized": normalize,
    }


def make_timestamps(data, t, lookback, horizon):
    """Return context/future timestamps, falling back to hourly dummy dates."""
    if isinstance(data.index, pd.DatetimeIndex):
        context_timestamps = data.index[t - lookback + 1 : t + 1]
        future_timestamps = data.index[t + 1 : t + horizon + 1]
    else:
        context_timestamps = pd.date_range("2000-01-01", periods=lookback, freq="1h")
        future_timestamps = pd.date_range(
            context_timestamps[-1] + pd.Timedelta(hours=1),
            periods=horizon,
            freq="1h",
        )
    return context_timestamps, future_timestamps


def mse(y, pred):
    """Return mean squared error while ignoring NaNs."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.nanmean((y - pred) ** 2))


def set_seed(seed):
    """Sets RNG seeds when seed is not None."""
    if seed == "None":
        seed = None
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

def symlog(x, linthresh=1):
    """Signed log transform with linear threshold."""
    return np.sign(x) * np.log1p(np.abs(x / linthresh)) * linthresh

def normalize(x, mean=None, std=None, mode="standard", axis=-1, eps=1e-8):
    """Normalizes numpy arrays / torch tensors with raw/standard/instance modes."""
    if mode == "raw":
        return x

    is_torch = torch.is_tensor(x)

    if mode == "standard":
        if mean is None or std is None:
            raise ValueError("standard normalization requires mean and std.")
        if is_torch:
            return (x - mean) / (std + torch.tensor(eps, device=x.device, dtype=x.dtype))
        return (x - mean) / (std + eps)

    if mode == "instance":
        if is_torch:
            m = x.mean(dim=axis, keepdim=True) if mean is None else mean
            s = x.std(dim=axis, keepdim=True) if std is None else std
            return (x - m) / (s + torch.tensor(eps, device=x.device, dtype=x.dtype))
        m = x.mean(axis=axis, keepdims=True) if mean is None else mean
        s = x.std(axis=axis, keepdims=True) if std is None else std
        return (x - m) / (s + eps)

    raise ValueError("Unknown normalization mode.")

def filter_df(df, mask):
    """Masks df entries where mask is True."""
    out = df.copy()
    out[mask] = pd.NA
    return out

def filter_dict(dico, keys):
    """Filters a dict by a list of keys."""
    return {k: dico[k] for k in keys}

def cte_mask(df, lookback):
    """Returns mask of constant rolling windows of length lookback."""
    return df.rolling(window=lookback).std() == 0

def get_normal_stats(x):
    """Returns per-sample mean/std over last dimension."""
    mean = x.mean(dim=-1, keepdim=True).detach()
    std = x.std(dim=-1, keepdim=True).detach()
    return mean, std

def unroll_windows(dataloader, cap=None, normal=False, mean=None, std=None, seed=None):
    """Unrolls windows from a torch dataloader into tensors."""
    set_seed(seed)

    X, Y, C = [], [], []
    carry_on, total = True, 0
    while carry_on:
        for x, c, y, indiv, date in dataloader:
            total += x.shape[0]
            if normal:
                if mean is None and std is None:
                    mean, std = get_normal_stats(x)
                x, y = normalize(x, mean=mean, std=std, mode="standard"), normalize(y, mean=mean, std=std, mode="standard")
            X.append(x)
            Y.append(y)
            C.append(c)
            if cap is not None and total + x.shape[0] > cap:
                carry_on = True
                break
        if cap is None or total >= cap:
            carry_on = False

    return torch.concat(X), torch.concat(Y), torch.concat(C)

def get_trend(df, window=1000):
    """Rolling mean trend."""
    return df.rolling(window=window).mean().iloc[window:]

def get_aggr(df, window=100):
    """Block-wise mean aggregation with block size window."""
    n = len(df)
    if n == 0:
        return df.copy()
    block_ids = np.arange(n) // window
    block_means = df.groupby(block_ids).mean()
    out = df.copy()
    for pos, idx in enumerate(df.index):
        out.loc[idx] = block_means.loc[block_ids[pos]]
    return out

def split_six_way(df, time_splits=(0.6, 0.4), indiv_split=1.0, seed=0):
    """Six-way split for dataframe."""
    set_seed(seed)

    if len(time_splits) not in (2, 3):
        raise ValueError("time_splits must have length 2 or 3")

    n = len(df)
    cols = list(df.columns)

    if len(time_splits) == 2:
        a, b = time_splits
        t1, t2 = int(a * n), n
    else:
        a, b, c = time_splits
        t1, t2 = int(a * n), int((a + b) * n)

    k_primary = int(indiv_split * len(cols))
    perm = np.random.permutation(cols)
    primary_cols = list(perm[:k_primary])
    secondary_cols = list(perm[k_primary:])

    df_primary = df[primary_cols] if primary_cols else df.iloc[:, :0]
    df_secondary = df[secondary_cols] if secondary_cols else df.iloc[:, :0]

    if len(time_splits) == 2:
        train = df_primary.iloc[:t1]
        valid1 = df_primary.iloc[:0]
        test1 = df_primary.iloc[t1:]
        valid2 = df_secondary.iloc[:t1]
        valid3 = df_secondary.iloc[:0]
        test2 = df_secondary.iloc[t1:]
    else:
        train = df_primary.iloc[:t1]
        valid1 = df_primary.iloc[t1:t2]
        test1 = df_primary.iloc[t2:]
        valid2 = df_secondary.iloc[:t1]
        valid3 = df_secondary.iloc[t1:t2]
        test2 = df_secondary.iloc[t2:]

    return {"train": train, "valid1": valid1, "test1": test1, "valid2": valid2, "valid3": valid3, "test2": test2}

def get_train_stats(df_dict, cache):
    """Returns global mean/std from df_dict['train'] with caching."""
    if cache.get("train_stats") is not None:
        return cache["train_stats"]
    vals = df_dict["train"].values.astype(float)
    cache["train_stats"] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
    return cache["train_stats"]

def normalize_xy(X, Y, norm_mode, train_stats=None):
    """Normalizes window arrays X/Y with raw/standard/instance modes."""
    if norm_mode == "raw":
        return X, Y
    if norm_mode == "standard":
        if train_stats is None:
            raise ValueError("train_stats required for standard normalization.")
        mu, sig = train_stats
        Xn = normalize(X, mean=mu, std=sig, mode="standard", axis=1)
        Yn = normalize(Y, mean=mu, std=sig, mode="standard", axis=1) if (Y is not None and Y.size) else Y
        return Xn, Yn
    if norm_mode == "instance":
        Xn = normalize(X, mode="instance", axis=1)
        Yn = normalize(Y, mean=X.mean(axis=1, keepdims=True), std=X.std(axis=1, keepdims=True), mode="standard", axis=1) if (Y is not None and Y.size) else Y
        return Xn, Yn
    raise ValueError("Unknown normalization mode.")

def compute_views_df(df, lags=168, horizon=24):
    """Builds raw/fourier/gamma data views."""
    return {"raw": df, "fourier": get_fourier_df(df), "gamma": get_gamma_df(df, lags=int(lags), horizon=int(horizon))}

def ensure_gamma_view(cache, df, lags, horizon):
    """Ensures cache['dfs']['gamma'] matches (lags,horizon)."""
    gp = (int(lags), int(horizon))
    if cache.get("gamma_params") != gp:
        cache["dfs"]["gamma"] = get_gamma_df(df, lags=gp[0], horizon=gp[1])
        cache["gamma_params"] = gp

def build_concat_samples(df_dict, keys, first_key, L, H, N_ref, ignore_cte, mode, norm_mode, seed=None, train_stats=None):
    """Builds concatenated sample matrix and dataset labels."""
    set_seed(seed)

    ref_len = len(df_dict[first_key])
    if ref_len == 0:
        return np.empty((0, 0)), np.array([]), {}

    feats_all, labels_all, sizes = [], [], {}

    for name in keys:
        df = df_dict[name]
        n_dates = len(df)
        if n_dates == 0:
            sizes[name] = 0
            continue

        N = max(1, int(n_dates / float(ref_len) * N_ref)) if ref_len > 0 else max(1, N_ref)
        X, Y = sample_windows_df(df, L, H, N, columns=None, ignore_cte=ignore_cte, seed=seed)
        if X.size == 0:
            sizes[name] = 0
            continue

        X, Y = normalize_xy(X, Y, norm_mode, train_stats=train_stats)
        A = X if mode == "inputs" else (np.concatenate([X, Y], axis=1) if (Y is not None and Y.size) else np.empty((0, L + H)))
        if A.size == 0:
            sizes[name] = 0
            continue

        feats_all.append(A)
        labels_all.append(np.array([name] * A.shape[0], dtype=object))
        sizes[name] = int(A.shape[0])

    if not feats_all:
        return np.empty((0, 0)), np.array([]), {}

    return np.concatenate(feats_all, axis=0), np.concatenate(labels_all, axis=0), sizes

def time_series_features(
    df: pd.DataFrame,
    alpha: float = 0.05,
    stationarity: bool = False,
    max_pacf_lag: int = 40,
    max_seasonality_peaks: int = 10,
    seasonality_min_period: int = 2,
    seasonality_max_period: int | None = None,
    hist_bins: int = 30,
    top_k_lags: int = 3,
    top_k_seasonalities: int = 3,
) -> pd.DataFrame:
    """
    Compute panel-level descriptive features for a time-series DataFrame shaped (dates x variates).
    """
    try:
        from scipy.signal import find_peaks, periodogram
        from scipy.spatial.distance import pdist
        from scipy.stats import kurtosis, norm, skew
        from statsmodels.tsa.stattools import adfuller, pacf
    except Exception as e:
        raise ImportError("This function requires scipy and statsmodels.") from e

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame shaped (dates x variates).")
    if df.empty or df.shape[1] == 0:
        raise ValueError("df must have at least 1 row and 1 column.")

    df = df.astype(float).copy()
    n_dates, n_variates = df.shape
    n_instances = n_dates * n_variates

    def now() -> float:
        return time.perf_counter()

    def minutes(t0: float) -> float:
        return round((time.perf_counter() - t0) / 60.0, 2)

    def add(rows: list, feature: str, value, elapsed_min: float) -> None:
        rows.append((feature, value, f"{elapsed_min:.2f}"))

    def as_datetime_index(index: pd.Index) -> pd.DatetimeIndex | None:
        if isinstance(index, pd.DatetimeIndex):
            return index
        try:
            return pd.to_datetime(index, errors="raise")
        except Exception:
            return None

    def format_timedelta(td) -> str | float:
        if pd.isna(td):
            return np.nan
        total_seconds = int(pd.Timedelta(td).total_seconds())
        days, rem = divmod(total_seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes_, seconds = divmod(rem, 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        parts.append(f"{hours}h")
        parts.append(f"{minutes_:02d}m")
        parts.append(f"{seconds:02d}s")
        return " ".join(parts)

    def sampling_rate(index: pd.Index):
        dt_index = as_datetime_index(index)
        if dt_index is None or len(dt_index) < 2:
            return np.nan, np.nan, np.nan
        diffs = dt_index.to_series().diff().dropna()
        if diffs.empty:
            return np.nan, np.nan, np.nan
        median_delta = diffs.median()
        seconds = median_delta.total_seconds()
        freq_hz = 1.0 / seconds if seconds and seconds > 0 else np.nan
        hours_per_sample = seconds / 3600.0 if seconds and seconds > 0 else np.nan
        return median_delta, freq_hz, hours_per_sample

    def clean(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x[np.isfinite(x)]

    def difference(x: np.ndarray, order: int) -> np.ndarray:
        y = clean(x)
        for _ in range(order):
            if y.size < 2:
                return np.array([], dtype=float)
            y = np.diff(y)
        return y

    def adf_stationary(x: np.ndarray) -> bool:
        x = clean(x)
        if x.size < 10:
            return False
        try:
            return adfuller(x, autolag="AIC")[1] < alpha
        except Exception:
            return False

    def top_significant_pacf_lags(x: np.ndarray) -> list[int]:
        x = clean(x)
        if x.size < 20:
            return []
        nlags = min(max_pacf_lag, x.size // 2)
        if nlags < 1:
            return []
        try:
            vals = pacf(x, nlags=nlags, method="yw")
            bound = norm.ppf(1 - alpha / 2) / np.sqrt(x.size)
            sig = [(lag, abs(vals[lag])) for lag in range(1, len(vals)) if abs(vals[lag]) > bound]
            sig.sort(key=lambda t: t[1], reverse=True)
            return [lag for lag, _ in sig[:top_k_lags]]
        except Exception:
            return []

    def top_seasonality_periods(x: np.ndarray) -> list[int]:
        x = clean(x)
        n = len(x)
        if n < 20:
            return []
        x = x - x.mean()
        f, pxx = periodogram(x, scaling="density")
        if f.size < 3:
            return []
        max_period = seasonality_max_period or (n // 2)
        min_f = 1.0 / max_period
        max_f = 1.0 / seasonality_min_period
        mask = (f >= min_f) & (f <= max_f)
        f2, p2 = f[mask], pxx[mask]
        if f2.size < 3 or np.nanmax(p2) <= 0:
            return []
        peaks, _ = find_peaks(p2, prominence=0.10 * np.nanmax(p2))
        if peaks.size == 0:
            return []
        order = np.argsort(p2[peaks])[::-1]
        periods = []
        for idx in peaks[order]:
            period = int(round(1.0 / f2[idx])) if f2[idx] > 0 else None
            if period and seasonality_min_period <= period <= max_period and period not in periods:
                periods.append(period)
            if len(periods) == top_k_seasonalities:
                break
        return periods

    def symmetric_kl_mean(panel: pd.DataFrame) -> float:
        vals = clean(panel.to_numpy().ravel())
        if vals.size < 10:
            return np.nan
        edges = np.histogram_bin_edges(vals, bins=hist_bins)
        eps = 1e-12
        dists = []
        for col in panel.columns:
            x = clean(panel[col].to_numpy())
            if x.size == 0:
                dists.append(None)
                continue
            hist, _ = np.histogram(x, bins=edges, density=False)
            p = hist.astype(float) + eps
            dists.append(p / p.sum())
        out = []
        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                if dists[i] is None or dists[j] is None:
                    continue
                pi, pj = dists[i], dists[j]
                out.append(0.5 * (np.sum(pi * np.log(pi / pj)) + np.sum(pj * np.log(pj / pi))))
        return float(np.mean(out)) if out else np.nan

    def mean_offdiag(mat: pd.DataFrame | np.ndarray) -> float:
        a = np.asarray(mat, dtype=float)
        if a.ndim != 2 or a.shape[0] < 2:
            return np.nan
        mask = ~np.eye(a.shape[0], dtype=bool)
        vals = a[mask]
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else np.nan

    def mean_euclidean_distance(panel: pd.DataFrame) -> float:
        if panel.shape[1] < 2:
            return np.nan
        x = panel.fillna(panel.mean()).fillna(0.0)
        d = pdist(x.T.to_numpy(), metric="euclidean")
        return float(d.mean()) if d.size else np.nan

    def spectral_centroid_hours(x: np.ndarray, hours_per_sample: float) -> float:
        x = clean(x)
        if x.size < 4 or not np.isfinite(hours_per_sample) or hours_per_sample <= 0:
            return np.nan
        x = x - x.mean()
        f, pxx = periodogram(x, scaling="spectrum")
        mask = (f > 0) & np.isfinite(pxx) & (pxx > 0)
        f, pxx = f[mask], pxx[mask]
        if f.size == 0 or pxx.sum() == 0:
            return np.nan
        centroid_cycles_per_sample = float(np.sum(f * pxx) / np.sum(pxx))
        if centroid_cycles_per_sample <= 0:
            return np.nan
        return hours_per_sample / centroid_cycles_per_sample

    rows: list[tuple[str, object, str]] = []

    t0 = now()
    elapsed = minutes(t0)
    add(rows, "No. of Variates", n_variates, elapsed)
    add(rows, "No. of Dates", n_dates, elapsed)
    add(rows, "No. of Instances", n_instances, elapsed)

    t0 = now()
    median_delta, freq_hz, hours_per_sample = sampling_rate(df.index)
    elapsed = minutes(t0)
    add(rows, "Sampling Rate", format_timedelta(median_delta), elapsed)
    add(rows, "Sampling Rate (Hz)", freq_hz, elapsed)

    t0 = now()
    missing_per_variate = df.isna().sum(axis=0)
    total_missing = int(missing_per_variate.sum())
    max_missing_one_variate = int(missing_per_variate.max()) if len(missing_per_variate) else 0
    elapsed = minutes(t0)
    add(rows, "No. of Detected Missing Values", total_missing, elapsed)
    add(rows, "Max No. of Missing Values", max_missing_one_variate, elapsed)

    if stationarity:
        t0 = now()
        level_count = diff1_count = diff2_count = 0
        for col in df.columns:
            x = df[col].to_numpy()
            level_count += adf_stationary(x)
            diff1_count += adf_stationary(difference(x, 1))
            diff2_count += adf_stationary(difference(x, 2))
        elapsed = minutes(t0)
        add(rows, "No. of Stationary Features", int(level_count), elapsed)
        add(rows, "No. of Stationary Features after 1st Order Diff", int(diff1_count), elapsed)
        add(rows, "No. of Stationary Features after 2nd Order Diff", int(diff2_count), elapsed)

    t0 = now()
    lag_counter = Counter()
    for col in df.columns:
        lag_counter.update(top_significant_pacf_lags(df[col].to_numpy()))
    add(rows, "Significant Lags (lag, counts)", lag_counter.most_common(top_k_lags), minutes(t0))

    t0 = now()
    season_counter = Counter()
    for col in df.columns:
        season_counter.update(top_seasonality_periods(df[col].to_numpy()))
    add(rows, "Seasonalities (seasonality, counts)", season_counter.most_common(top_k_seasonalities), minutes(t0))

    t0 = now()
    skews = [skew(clean(df[col].to_numpy()), bias=False) for col in df.columns if clean(df[col].to_numpy()).size >= 3]
    kurts = [kurtosis(clean(df[col].to_numpy()), bias=False) + 3 for col in df.columns if clean(df[col].to_numpy()).size >= 4]
    elapsed = minutes(t0)
    add(rows, "Mean Skewness", round(float(np.mean(skews)), 2) if skews else np.nan, elapsed)
    add(rows, "Mean Kurtosis", round(float(np.mean(kurts)), 2) if kurts else np.nan, elapsed)

    t0 = now()
    mean_kl = symmetric_kl_mean(df)
    corr = df.corr()
    mean_corr = mean_offdiag(corr) if corr.shape[1] >= 2 else np.nan
    mean_euclid = mean_euclidean_distance(df)
    elapsed = minutes(t0)
    add(rows, "Mean Symmetric KL Div.", round(mean_kl,2), elapsed)
    add(rows, "Mean Correlation", round(mean_corr,2), elapsed)
    add(rows, "Mean Euclidean Distance", round(mean_euclid,2), elapsed)

    t0 = now()
    centroids_h = [spectral_centroid_hours(df[col].to_numpy(), hours_per_sample) for col in df.columns]
    centroids_h = [x for x in centroids_h if np.isfinite(x)]
    add(
        rows,
        "Mean Spectral Centroid (hours)",
        round(float(np.mean(centroids_h)), 2) if centroids_h else np.nan,
        minutes(t0),
    )

    return pd.DataFrame(rows, columns=["Feature", "Value", "Compute (min)"])

def get_fourier_df(df, eps=1e-8):
    """Per-column FFT magnitude of standardized series."""
    return df.apply(lambda x: np.abs(np.fft.fft((x - x.mean()) / (x.std() + eps))))

def get_gammas(data, lookback, horizon, eps=1e-8):
    """Returns alpha/beta dataframes from rolling lookback/horizon stats."""
    lookback_means = data.rolling(window=lookback).mean().iloc[lookback:]
    lookback_stds = data.rolling(window=lookback).std().iloc[lookback:]
    horizon_means = data.rolling(window=horizon).mean().shift(-horizon).iloc[:-horizon]
    horizon_stds = data.rolling(window=horizon).std().shift(-horizon).iloc[:-horizon]
    alphas = horizon_stds.iloc[lookback:] / (lookback_stds.iloc[:-horizon] + eps)
    betas = (horizon_means.iloc[lookback:] - lookback_means.iloc[:-horizon]) / (lookback_stds.iloc[:-horizon] + eps)
    return alphas, betas

def get_gamma_df(df, lags, horizon, eps=1e-8):
    """Concatenates alpha and beta into a single dataframe."""
    alphas_df, betas_df = get_gammas(df, lags, horizon, eps=eps)
    return pd.concat((alphas_df, betas_df))

def get_dataset_stats(df_dict, lags, horizon, remove_train_cte=True, remove_eval_cte=True, save_path=None):
    """Computes dataset-wide mean/std and average alpha/beta for each split."""
    gammas_dict = {k: get_gammas(df_dict[k], lags, horizon) for k in df_dict}
    stats_dict = {}
    for key in df_dict:
        if (key == "train" and remove_train_cte) or (key != "train" and remove_eval_cte):
            mask = cte_mask(df_dict[key], lags)
            clean_df = filter_df(df_dict[key], mask)
            clean_alphas = filter_df(gammas_dict[key][0], mask)
            clean_betas = filter_df(gammas_dict[key][1], mask)
        else:
            clean_df, clean_alphas, clean_betas = df_dict[key], gammas_dict[key][0], gammas_dict[key][1]
        stats_dict[key] = {
            "mean": float(np.nanmean(clean_df.values)),
            "stds": float(np.nanmean(np.nanstd(clean_df.values, axis=0))),
            "std": float(np.nanstd(clean_df.values)),
            "alpha": float(np.nanmean(clean_alphas)),
            "beta": float(np.nanmean(clean_betas)),
        }

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(stats_dict, f, indent=4)

    return stats_dict

def sample_windows_df(df, lookback, horizon, n_windows, columns=None, ignore_cte=False, seed=None):
    """Samples windows from df and returns lookbacks and horizons arrays."""
    set_seed(seed)

    cols = list(df.columns) if columns is None else [c for c in columns if c in df.columns]
    L, H, N = int(lookback), int(horizon), int(n_windows)

    if L <= 0 or H <= 0 or N <= 0 or len(cols) == 0:
        return np.empty((0, L)), np.empty((0, H))

    X, Y = [], []
    for col in cols:
        x = df[col].values
        n = len(x)
        if n < L + H:
            continue

        possible_t = np.arange(L, n - H + 1)
        if len(possible_t) == 0:
            continue

        needed = N - len(X)
        if needed <= 0:
            break

        t_indices = possible_t if needed >= len(possible_t) else np.random.choice(possible_t, size=needed, replace=False)
        for t in t_indices:
            look = x[t - L:t]
            if ignore_cte and np.std(look) == 0:
                continue
            X.append(look)
            Y.append(x[t:t + H])

        if len(X) >= N:
            break

    if len(X) == 0:
        return np.empty((0, L)), np.empty((0, H))

    return np.asarray(X)[:N], np.asarray(Y)[:N]

def window_mean_std(windows):
    """Computes per-window mean/std from lookbacks."""
    x = np.asarray(windows)
    return (np.array([]), np.array([])) if x.size == 0 else (x.mean(axis=1), x.std(axis=1))

def window_alpha_beta(lookbacks, horizons, eps=1e-6):
    """Computes per-window alpha/beta from lookbacks and horizons."""
    X = np.asarray(lookbacks)
    Y = np.asarray(horizons)
    if X.size == 0 or Y.size == 0:
        return np.array([]), np.array([])
    mL, sL = X.mean(axis=1), X.std(axis=1)
    mH, sH = Y.mean(axis=1), Y.std(axis=1)
    denom = sL + eps
    return sH / denom, (mH - mL) / denom
