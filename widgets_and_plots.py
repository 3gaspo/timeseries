"""Plotting and ipywidget utilities for the dataset notebook."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
from tqdm.notebook import tqdm
from sklearn.manifold import TSNE, MDS
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import squareform

from general_utils import (
    build_concat_samples,
    compute_views_df,
    cte_mask,
    ensure_gamma_view,
    get_aggr,
    get_gammas,
    get_train_stats,
    get_trend,
    normalize_xy,
    sample_windows_df,
    set_seed,
    split_six_way,
    symlog,
    window_alpha_beta,
    window_mean_std,
)
from clustering_utils import (
    calculate_distances,
    energy_distance_multivariate,
    get_centroids,
    get_cluster_dicts,
    get_cluster_distances,
    get_cluster_heterogeneity,
    get_clusters,
    init_clusters,
)

def identify_cte(df, lookback, show=True, save_path=""):
    """Finds and optionally plots constant rolling window counts."""
    stds_mask = cte_mask(df, lookback)
    row_idxs, col_idxs = np.where(stds_mask)

    counts = {}
    for j in col_idxs:
        counts[j] = counts.get(j, 0) + 1

    total = int(len(col_idxs))
    if not counts:
        print("No constant windows found!")
        return

    vals = np.array(list(counts.values()))
    max_col_idx = max(counts, key=counts.get)
    print(f"Found {len(counts)} users with constant windows!")
    print(f"Total windows: {total}")
    print(f"Max per user:  {counts[max_col_idx]} (user {df.columns[max_col_idx]})")
    print(f"Mean per user: {vals.mean():.2f}")

    plt.figure(figsize=(6, 4))
    plt.hist(list(counts.values()), bins=100)
    plt.yscale("log")
    plt.title("Constant windows per individual")
    plt.xlabel("Individuals")
    plt.ylabel("log(counts)")
    if show:
        plt.show()
    else:
        plt.savefig((save_path or "") + "constants_hist.pdf")
    plt.close()

def plot_series_widget(df, trend_window=1000, aggr_window=100, gamma_lookback=168, gamma_horizon=24, gamma_eps=1e-8):
    """Interactive series widget with apply-gated numeric params."""
    cache = {"dfs": {}, "params": {}}

    dataframe_names = ["original", "alpha", "beta", "means", "aggregate"]
    column_names = list(df.columns)

    dataframe_dropdown = widgets.Dropdown(options=dataframe_names, value="original", description="Data:")
    column_dropdown = widgets.Dropdown(options=column_names, value=column_names[0], description="User:")
    next_button = widgets.Button(description="Next")
    axis_button = widgets.Button(description="Toggle Axis")
    apply_button = widgets.Button(description="Apply")

    L_widget = widgets.IntText(value=gamma_lookback, description="L:")
    H_widget = widgets.IntText(value=gamma_horizon, description="H:")
    trend_widget = widgets.IntText(value=trend_window, description="Trend:")
    aggr_widget = widgets.IntText(value=aggr_window, description="Aggr:")

    output = widgets.Output()
    axis_state = {"show": True}
    applied = {"L": int(L_widget.value), "H": int(H_widget.value), "trend": int(trend_widget.value), "aggr": int(aggr_widget.value)}

    def relevant_widgets(name):
        if name in ("alpha", "beta"):
            return [L_widget, H_widget]
        if name == "means":
            return [trend_widget]
        if name == "aggregate":
            return [aggr_widget]
        return []

    def update_param_ui():
        name = dataframe_dropdown.value
        for w in [L_widget, H_widget, trend_widget, aggr_widget]:
            w.layout.display = "none"
        for w in relevant_widgets(name):
            w.layout.display = "block"

    def precompute_all_default():
        p = {"gamma_lookback": applied["L"], "gamma_horizon": applied["H"], "trend_window": applied["trend"], "aggr_window": applied["aggr"], "gamma_eps": float(gamma_eps)}
        cache["params"] = p
        cache["dfs"]["original"] = df
        alphas_df, betas_df = get_gammas(df, p["gamma_lookback"], p["gamma_horizon"], eps=p["gamma_eps"])
        cache["dfs"]["alpha"] = alphas_df
        cache["dfs"]["beta"] = betas_df
        cache["dfs"]["means"] = get_trend(df, window=p["trend_window"])
        cache["dfs"]["aggregate"] = get_aggr(df, window=p["aggr_window"])

    def recompute_selected():
        p = {"gamma_lookback": applied["L"], "gamma_horizon": applied["H"], "trend_window": applied["trend"], "aggr_window": applied["aggr"], "gamma_eps": float(gamma_eps)}
        cache["params"] = p
        name = dataframe_dropdown.value
        if name in ("alpha", "beta"):
            alphas_df, betas_df = get_gammas(df, p["gamma_lookback"], p["gamma_horizon"], eps=p["gamma_eps"])
            cache["dfs"]["alpha"], cache["dfs"]["beta"] = alphas_df, betas_df
        elif name == "means":
            cache["dfs"]["means"] = get_trend(df, window=p["trend_window"])
        elif name == "aggregate":
            cache["dfs"]["aggregate"] = get_aggr(df, window=p["aggr_window"])
        else:
            cache["dfs"]["original"] = df

    def update_plot():
        with output:
            output.clear_output(wait=True)
            name = dataframe_dropdown.value
            col = column_dropdown.value
            df_current = cache["dfs"].get(name, None)
            if df_current is None:
                precompute_all_default()
                df_current = cache["dfs"][name]
            if col not in df_current.columns:
                print(f"Column '{col}' not in DataFrame '{name}'.")
                return
            plt.figure(figsize=(15, 4))
            plt.plot(df_current[col])
            if axis_state["show"]:
                plt.title(f"{name} - {col}")
                plt.grid(True)
                plt.xlabel("Index")
                plt.ylabel("Value")
            else:
                plt.axis("off")
            plt.show()

    def on_apply(_):
        applied["L"] = int(L_widget.value)
        applied["H"] = int(H_widget.value)
        applied["trend"] = int(trend_widget.value)
        applied["aggr"] = int(aggr_widget.value)
        recompute_selected()
        update_plot()

    def on_choice_change(_):
        update_param_ui()
        update_plot()

    def on_next(_):
        idx = column_names.index(column_dropdown.value)
        column_dropdown.value = column_names[(idx + 1) % len(column_names)]

    def on_axis(_):
        axis_state["show"] = not axis_state["show"]
        update_plot()

    precompute_all_default()
    update_param_ui()

    dataframe_dropdown.observe(on_choice_change, names="value")
    column_dropdown.observe(on_choice_change, names="value")
    next_button.on_click(on_next)
    axis_button.on_click(on_axis)
    apply_button.on_click(on_apply)

    display(
        widgets.HBox([dataframe_dropdown, column_dropdown, next_button, axis_button]),
        widgets.HBox([L_widget, H_widget, trend_widget, aggr_widget, apply_button]),
        output,
    )
    update_plot()


def plot_window_widget(df, default_lookback=168, default_horizon=24, eps=1e-6):
    """Single-window visualization widget with apply-gated numeric params."""
    full_data = df.copy()
    cols = list(full_data.columns)
    n_rows = full_data.shape[0]

    t_widget = widgets.IntText(value=n_rows // 2, description="t:")
    L_widget = widgets.IntText(value=default_lookback, description="L:")
    H_widget = widgets.IntText(value=default_horizon, description="H:")
    user_dropdown = widgets.Dropdown(options=cols, value=cols[0], description="User:")
    norm_button = widgets.ToggleButton(value=False, description="Normalize")
    axis_button = widgets.ToggleButton(value=True, description="Axis/Stats")
    apply_button = widgets.Button(description="Apply")

    output_plot = widgets.Output()
    output_stats = widgets.Output()

    applied = {
        "t": int(t_widget.value),
        "L": int(L_widget.value),
        "H": int(H_widget.value),
    }

    def clamp_t(t, L, H):
        t = max(L, int(t))
        return min(t, n_rows - H)

    def render():
        with output_plot:
            output_plot.clear_output(wait=True)

            t = int(applied["t"])
            L = int(applied["L"])
            H = int(applied["H"])
            col = user_dropdown.value
            do_norm = bool(norm_button.value)
            show_axis = bool(axis_button.value)

            if L <= 0 or H <= 0:
                print("L and H must be > 0.")
                return

            t2 = clamp_t(t, L, H)
            if t2 != t:
                applied["t"] = t2
                t = t2

            series = full_data[col].values
            look = series[t - L:t]
            hor = series[t:t + H]

            m = float(np.mean(look))
            s = float(np.std(look))

            alpha, beta = window_alpha_beta(look[None, :], hor[None, :], eps=eps)
            alpha = float(alpha[0]) if alpha.size else None
            beta = float(beta[0]) if beta.size else None

            lookback_vals = full_data[col].iloc[t - L:t].copy()
            horizon_vals = full_data[col].iloc[t:t + H].copy()

            if do_norm and s > 0:
                lookback_vals = (lookback_vals - m) / s
                horizon_vals = (horizon_vals - m) / s
                ylabel = "Z-score"
                title_suffix = " (normalized)"
            else:
                ylabel = "Value"
                title_suffix = ""

            x_look = list(range(t - L, t)) + ([t] if len(horizon_vals) else [])
            y_look = list(lookback_vals.values) + (
                [float(horizon_vals.iloc[0])] if len(horizon_vals) else []
            )
            x_hor = list(range(t, t + H))
            y_hor = list(horizon_vals.values)

            plt.figure(figsize=(12, 4))
            plt.plot(x_look, y_look, label=f"L={L}")
            plt.plot(x_hor, y_hor, label=f"H={H}")
            plt.axvline(x=t, linestyle="--", label="split")

            if show_axis:
                plt.title(f"{col} - t={t}{title_suffix}")
                plt.xlabel("Index")
                plt.ylabel(ylabel)
                plt.grid(True)
                plt.legend()
            else:
                plt.axis("off")

            plt.show()

        with output_stats:
            output_stats.clear_output(wait=True)

            if not show_axis:
                return

            if alpha is None:
                print("Invalid window.")
                return

            print("--- Stats ---")
            print(f"mean:  {m:.4f}")
            print(f"std:   {s:.4f}")
            print(f"alpha: {alpha:.4f}")
            print(f"beta:  {beta:.4f}")

    def on_apply(_):
        applied["t"] = int(t_widget.value)
        applied["L"] = int(L_widget.value)
        applied["H"] = int(H_widget.value)
        render()

    def on_choice_change(_):
        render()

    for w in [user_dropdown, norm_button, axis_button]:
        w.observe(on_choice_change, names="value")

    apply_button.on_click(on_apply)

    display(
        widgets.HBox([user_dropdown, norm_button, axis_button]),
        widgets.HBox([t_widget, L_widget, H_widget, apply_button]),
        widgets.HBox([output_plot, output_stats]),
    )

    on_apply(None)


def _periodic_angle(values, period):
    v = np.asarray(values, dtype=float)
    return (np.pi / 2 - (v * 2 * np.pi / float(period))) % (2 * np.pi)

def _weighted_median(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w)
    v, w = v[m], w[m]
    if v.size == 0:
        return np.nan
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.median(v))
    return float(v[np.searchsorted(cw, cw[-1] / 2.0)])

def _periodic_stats(values, weights, period):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w)
    v, w = v[m], w[m]
    if v.size == 0:
        return {"mode": np.nan, "median": np.nan, "arith_mean": np.nan, "circ_mean": np.nan}

    mode = float(v[int(np.argmax(w))])
    median = _weighted_median(v, w)

    wsum = float(np.sum(w))
    arith = float(np.sum(v * w) / wsum) if wsum > 0 else float(np.mean(v))

    angles = v * (2 * np.pi / float(period))
    S = float(np.sum(np.sin(angles) * w) / wsum) if wsum > 0 else float(np.mean(np.sin(angles)))
    C = float(np.sum(np.cos(angles) * w) / wsum) if wsum > 0 else float(np.mean(np.cos(angles)))
    circ_angle = float(np.arctan2(S, C) % (2 * np.pi))
    circ = float(circ_angle * (float(period) / (2 * np.pi)))

    return {"mode": mode, "median": median, "arith_mean": arith, "circ_mean": circ}

def _base_periodic_polar_plot(values, weights, period, ticklabels=None, title=""):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    fig.set_size_inches(3.2, 3.2)

    theta = _periodic_angle(values, period)
    w = weights.copy()
    w[~np.isfinite(w)] = 0.0

    if w.size == 0:
        ax.set_title(title, pad=20)
        return fig, ax, np.array([]), np.array([])

    wmax = float(np.max(w)) if np.max(w) > 0 else 1.0
    w_norm = w / wmax

    width = (2 * np.pi / float(period)) * 0.85
    ax.bar(theta, w_norm, width=width, alpha=0.25, edgecolor="none")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.15)

    if ticklabels is not None:
        ax.set_xticks(_periodic_angle(values, period))
        ax.set_xticklabels(ticklabels, fontsize=7)

    ax.set_title(title, pad=22, fontsize=10)
    return fig, ax, theta, w_norm

def plot_seasonality_widget(df):
    """Circular seasonality widget (daily/weekly/yearly) with precomputed aggregations."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame shaped (dates x users).")
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError("df.index must be datetime-like for seasonality plots.") from e

    cols = list(df.columns)
    if not cols:
        raise ValueError("df must have at least one column.")

    user_options = [("All Sensors (Mean)", "__mean__"), ("All Sensors (Sum)", "__sum__")] + [(c, c) for c in cols]
    seasonality_options = [("Daily", "daily"), ("Weekly", "weekly"), ("Yearly", "yearly")]

    seasonality_dropdown = widgets.Dropdown(options=seasonality_options, value="daily", description="Season:")
    user_dropdown = widgets.Dropdown(options=user_options, value="__mean__", description="User:")
    axis_button = widgets.ToggleButton(value=True, description="Axis")
    output = widgets.Output()

    cache = {"aggr": {}, "stats": {}}

    def build_series(key):
        if key == "__mean__":
            return df.mean(axis=1)
        if key == "__sum__":
            return df.sum(axis=1)
        return df[key]

    def group_key(seasonality):
        if seasonality == "daily":
            return df.index.hour, np.arange(24), 24, [str(i) for i in range(24)]
        if seasonality == "weekly":
            return df.index.dayofweek, np.arange(7), 7, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return df.index.month, np.arange(1, 13), 12, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def precompute():
        for _, skey in seasonality_options:
            idx_group, full_vals, period, labels = group_key(skey)
            cache["aggr"][skey] = {}
            cache["stats"][skey] = {}
            for _, ukey in user_options:
                s = build_series(ukey)
                g = s.groupby(idx_group).mean()
                g = g.reindex(full_vals).astype(float)
                values = g.index.to_numpy(dtype=float)
                weights = g.to_numpy(dtype=float)
                cache["aggr"][skey][ukey] = (values, weights, period, labels)
                cache["stats"][skey][ukey] = _periodic_stats(values, weights, period)

    def plot_current():
        with output:
            output.clear_output(wait=True)

            skey = seasonality_dropdown.value
            ukey = user_dropdown.value
            values, weights, period, labels = cache["aggr"][skey][ukey]
            stats = cache["stats"][skey][ukey]

            title = f"{skey.capitalize()} seasonality\n{user_dropdown.label}"
            fig, ax, _, _ = _base_periodic_polar_plot(values, weights, period, ticklabels=labels, title=title)

            bar_width = (2 * np.pi / float(period)) * 0.10
            r = 1.08
            ax.bar([_periodic_angle([stats["arith_mean"]], period)[0]], [r], width=bar_width, alpha=0.7, color="red",
                   label=f'Arith Mean ({stats["arith_mean"]:.1f})')
            ax.bar([_periodic_angle([stats["circ_mean"]], period)[0]], [r], width=bar_width, alpha=0.8, color="green",
                   label=f'Periodic Mean ({stats["circ_mean"]:.1f})')
            ax.bar([_periodic_angle([stats["median"]], period)[0]], [r], width=bar_width, alpha=0.7, color="purple",
                   label=f'Median ({stats["median"]:.1f})')
            ax.bar([_periodic_angle([stats["mode"]], period)[0]], [r], width=bar_width, alpha=0.7, color="orange",
                   label=f'Mode ({stats["mode"]:.1f})')

            if axis_button.value:
                ax.legend(bbox_to_anchor=(1.18, 1.05), loc="upper left", borderaxespad=0, fontsize=8)
            else:
                if ax.legend_ is not None:
                    ax.legend_.remove()
                ax.set_axis_off()

            plt.show()

    def on_change(_):
        plot_current()

    for w in [seasonality_dropdown, user_dropdown, axis_button]:
        w.observe(on_change, names="value")

    precompute()
    display(widgets.HBox([seasonality_dropdown, user_dropdown, axis_button]), output)
    plot_current()

def plot_stats_widget(df, seed=None):
    """Stats widget sampling windows per user with apply-gated numeric params."""
    full_data = df.copy()
    columns = list(full_data.columns)
    cache = {"params": None, "plot_df": {"Mean/Std": None, "Alpha/Beta": None}}
    applied = {"L": 168, "H": 24, "N": 100}

    L_widget = widgets.IntText(value=applied["L"], description="L:")
    H_widget = widgets.IntText(value=applied["H"], description="H:")
    N_widget = widgets.IntText(value=applied["N"], description="N:")
    apply_button = widgets.Button(description="Apply")

    user_dropdown = widgets.Dropdown(options=columns, value=columns[0], description="User:")
    type_dropdown = widgets.Dropdown(options=["Mean/Std", "Alpha/Beta"], value="Mean/Std", description="Type:")
    log_button = widgets.ToggleButton(value=True, description="Log")
    filter_button = widgets.ToggleButton(value=False, description="Filter cte")
    resample_button = widgets.Button(description="Resample")
    next_button = widgets.Button(description="Next User")
    output = widgets.Output()

    def compute_cache():
        set_seed(seed)
        records_ms, records_ab = [], []
        linthresh = 1
        L, H, N = int(applied["L"]), int(applied["H"]), int(applied["N"])
        ignore_cte = bool(filter_button.value)

        for col in columns:
            X, Y = sample_windows_df(full_data, L, H, N, columns=[col], ignore_cte=ignore_cte, seed=seed)
            means, stds = window_mean_std(X)
            records_ms += [{"user": col, "mean": m, "std": s} for m, s in zip(means, stds)]
            alphas, betas = window_alpha_beta(X, Y)
            records_ab += [{"user": col, "alpha": a, "beta": b} for a, b in zip(alphas, betas)]

        df_ms = pd.DataFrame(records_ms) if records_ms else None
        df_ab = pd.DataFrame(records_ab) if records_ab else None

        if df_ms is not None and not df_ms.empty:
            df_ms["mean_symlog"] = symlog(df_ms["mean"], linthresh=linthresh)
            df_ms["std_symlog"] = symlog(df_ms["std"], linthresh=linthresh)
        if df_ab is not None and not df_ab.empty:
            df_ab["alpha_symlog"] = symlog(df_ab["alpha"], linthresh=linthresh)
            df_ab["beta_symlog"] = symlog(df_ab["beta"], linthresh=linthresh)

        cache["plot_df"]["Mean/Std"] = df_ms
        cache["plot_df"]["Alpha/Beta"] = df_ab
        cache["params"] = (L, H, N, ignore_cte)

    def ensure_cached(force=False):
        params = (int(applied["L"]), int(applied["H"]), int(applied["N"]), bool(filter_button.value))
        if force or cache["params"] != params:
            compute_cache()

    def plot_current():
        with output:
            output.clear_output(wait=True)
            ensure_cached(force=False)

            plot_type = type_dropdown.value
            plot_df = cache["plot_df"].get(plot_type, None)
            if plot_df is None or plot_df.empty:
                print("No data.")
                return

            highlight = user_dropdown.value
            use_log = bool(log_button.value)
            plot_df = plot_df.copy()
            plot_df["type"] = plot_df["user"].apply(lambda u: "user" if u == highlight else "all")
            plot_df = plot_df.sort_values(by="type", ascending=True)

            if plot_type == "Mean/Std":
                x_base, y_base = "mean", "std"
                x_label_raw, y_label_raw = "Mean", "Std"
            else:
                x_base, y_base = "beta", "alpha"
                x_label_raw, y_label_raw = "Beta", "Alpha"

            if use_log:
                x_col = f"{x_base}_symlog"
                y_col = f"{y_base}_symlog"
                x_label, y_label = f"{x_label_raw} (log)", f"{y_label_raw} (log)"
            else:
                x_col, y_col = x_base, y_base
                x_label, y_label = x_label_raw, y_label_raw

            g = sns.jointplot(
                data=plot_df,
                x=x_col,
                y=y_col,
                hue="type",
                palette={"user": "red", "all": "blue"},
                kind="scatter",
                height=7,
                s=20,
                marginal_kws=dict(common_norm=False, fill=True, alpha=0.5),
            )
            g.ax_joint.set_xlabel(x_label)
            g.ax_joint.set_ylabel(y_label)
            g.fig.suptitle(f"L={int(applied['L'])}, H={int(applied['H'])}, N={int(applied['N'])} - {plot_type}", y=1.02)
            plt.show()

    def on_apply(_):
        applied["L"] = int(L_widget.value)
        applied["H"] = int(H_widget.value)
        applied["N"] = int(N_widget.value)
        ensure_cached(force=True)
        plot_current()

    def on_resample(_):
        cache["params"] = None
        ensure_cached(force=True)
        plot_current()

    def on_next(_):
        idx = columns.index(user_dropdown.value)
        user_dropdown.value = columns[(idx + 1) % len(columns)]

    for w in [user_dropdown, type_dropdown, log_button, filter_button]:
        w.observe(lambda _: plot_current(), names="value")

    resample_button.on_click(on_resample)
    next_button.on_click(on_next)
    apply_button.on_click(on_apply)

    compute_cache()
    display(
        widgets.HBox([type_dropdown, log_button, filter_button, resample_button]),
        widgets.HBox([user_dropdown, next_button]),
        widgets.HBox([L_widget, H_widget, N_widget, apply_button]),
        output,
    )
    plot_current()

def plot_stats_dict_widget(df_dict, seed=None):
    """Stats widget over datasets with apply-gated numeric params."""
    keys = list(df_dict.keys())
    cache = {"params": None, "plot_df": {"Mean/Std": None, "Alpha/Beta": None}}
    applied = {"L": 168, "H": 24, "N_ref": 100}

    L_widget = widgets.IntText(value=applied["L"], description="L:")
    H_widget = widgets.IntText(value=applied["H"], description="H:")
    N_widget = widgets.IntText(value=applied["N_ref"], description="N (ref):")
    apply_button = widgets.Button(description="Apply")

    type_dropdown = widgets.Dropdown(options=["Mean/Std", "Alpha/Beta"], value="Mean/Std", description="Type:")
    log_button = widgets.ToggleButton(value=True, description="Log")
    filter_button = widgets.ToggleButton(value=False, description="Filter cte")
    resample_button = widgets.Button(description="Resample")
    output = widgets.Output()

    def compute_cache():
        set_seed(seed)
        records_ms, records_ab = [], []
        linthresh = 1
        L, H, N_ref = int(applied["L"]), int(applied["H"]), int(applied["N_ref"])
        ignore_cte = bool(filter_button.value)

        ref_len = len(df_dict[keys[0]])
        if ref_len == 0:
            cache["plot_df"]["Mean/Std"] = None
            cache["plot_df"]["Alpha/Beta"] = None
            cache["params"] = (L, H, N_ref, ignore_cte)
            return

        for name in keys:
            df = df_dict[name]
            n_dates = len(df)
            if n_dates == 0:
                continue
            N = max(1, int(n_dates / float(ref_len) * N_ref))
            for col in list(df.columns):
                X, Y = sample_windows_df(df, L, H, N, columns=[col], ignore_cte=ignore_cte, seed=seed)
                means, stds = window_mean_std(X)
                records_ms += [{"dataset": name, "mean": m, "std": s} for m, s in zip(means, stds)]
                alphas, betas = window_alpha_beta(X, Y)
                records_ab += [{"dataset": name, "alpha": a, "beta": b} for a, b in zip(alphas, betas)]

        df_ms = pd.DataFrame(records_ms) if records_ms else None
        df_ab = pd.DataFrame(records_ab) if records_ab else None

        if df_ms is not None and not df_ms.empty:
            df_ms["mean_symlog"] = symlog(df_ms["mean"], linthresh=linthresh)
            df_ms["std_symlog"] = symlog(df_ms["std"], linthresh=linthresh)
        if df_ab is not None and not df_ab.empty:
            df_ab["alpha_symlog"] = symlog(df_ab["alpha"], linthresh=linthresh)
            df_ab["beta_symlog"] = symlog(df_ab["beta"], linthresh=linthresh)

        cache["plot_df"]["Mean/Std"] = df_ms
        cache["plot_df"]["Alpha/Beta"] = df_ab
        cache["params"] = (L, H, N_ref, ignore_cte)

    def ensure_cached(force=False):
        params = (int(applied["L"]), int(applied["H"]), int(applied["N_ref"]), bool(filter_button.value))
        if force or cache["params"] != params:
            compute_cache()

    def plot_current():
        with output:
            output.clear_output(wait=True)
            ensure_cached(force=False)

            plot_type = type_dropdown.value
            plot_df = cache["plot_df"].get(plot_type, None)
            if plot_df is None or plot_df.empty:
                print("No data.")
                return

            use_log = bool(log_button.value)
            if plot_type == "Mean/Std":
                x_base, y_base = "mean", "std"
                x_label_raw, y_label_raw = "Mean", "Std"
            else:
                x_base, y_base = "beta", "alpha"
                x_label_raw, y_label_raw = "Beta", "Alpha"

            if use_log:
                x_col = f"{x_base}_symlog"
                y_col = f"{y_base}_symlog"
                x_label, y_label = f"{x_label_raw} (log)", f"{y_label_raw} (log)"
            else:
                x_col, y_col = x_base, y_base
                x_label, y_label = x_label_raw, y_label_raw

            g = sns.jointplot(
                data=plot_df,
                x=x_col,
                y=y_col,
                hue="dataset",
                kind="scatter",
                height=7,
                s=20,
                marginal_kws=dict(common_norm=False, fill=True, alpha=0.5),
            )
            g.ax_joint.set_xlabel(x_label)
            g.ax_joint.set_ylabel(y_label)
            g.fig.suptitle(f"L={int(applied['L'])}, H={int(applied['H'])}, N_ref={int(applied['N_ref'])} - {plot_type}", y=1.02)
            plt.show()

    def on_apply(_):
        applied["L"] = int(L_widget.value)
        applied["H"] = int(H_widget.value)
        applied["N_ref"] = int(N_widget.value)
        ensure_cached(force=True)
        plot_current()

    def on_resample(_):
        cache["params"] = None
        ensure_cached(force=True)
        plot_current()

    for w in [type_dropdown, log_button, filter_button]:
        w.observe(lambda _: plot_current(), names="value")

    resample_button.on_click(on_resample)
    apply_button.on_click(on_apply)

    compute_cache()
    display(
        widgets.HBox([type_dropdown, log_button, filter_button, resample_button]),
        widgets.HBox([L_widget, H_widget, N_widget, apply_button]),
        output,
    )
    plot_current()

def plot_distances_dict_widget(df_dict, seed=None):
    """Distance-matrix widget over datasets with apply-gated numeric params."""
    assert "train" in df_dict, "df_dict must include a 'train' dataframe for standard normalization."

    keys = list(df_dict.keys())
    cache = {"params": None, "matrices": None, "train_stats": None}
    applied = {"L": 168, "H": 24, "N_ref": 100}

    filter_button = widgets.ToggleButton(value=False, description="Filter cte")
    norm_dropdown = widgets.Dropdown(options=["raw", "standard", "instance"], value="raw", description="Norm:")
    dist_dropdown = widgets.Dropdown(options=["raw input", "raw joint", "mean/std", "alpha/beta"], value="raw input", description="Dist:")

    L_widget = widgets.IntText(value=applied["L"], description="L:")
    H_widget = widgets.IntText(value=applied["H"], description="H:")
    N_widget = widgets.IntText(value=applied["N_ref"], description="N (ref):")
    apply_button = widgets.Button(description="Apply")

    output = widgets.Output()

    def compute_matrices_for_norm(L, H, N_ref, ignore_cte, norm_mode):
        set_seed(seed)
        ref_len = len(df_dict[keys[0]])
        if ref_len == 0:
            return None

        train_stats = get_train_stats(df_dict, cache) if norm_mode == "standard" else None
        samples, feats = {}, {}

        for name in keys:
            df = df_dict[name]
            n_dates = len(df)
            N = max(1, int(n_dates / float(ref_len) * N_ref)) if ref_len > 0 else max(1, N_ref)

            X, Y = sample_windows_df(df, L, H, N, columns=None, ignore_cte=ignore_cte, seed=seed)
            X, Y = normalize_xy(X, Y, norm_mode, train_stats=train_stats)
            J = np.concatenate([X, Y], axis=1) if (X.size and Y.size) else np.empty((0, L + H))

            samples[name] = {"X": X, "J": J}
            m, s = window_mean_std(X)
            a, b = window_alpha_beta(X, Y)
            feats[name] = {
                "mean/std": np.stack([m, s], axis=1) if len(m) else np.empty((0, 2)),
                "alpha/beta": np.stack([a, b], axis=1) if len(a) else np.empty((0, 2)),
            }

        def pairwise_matrix(mode):
            M = np.full((len(keys), len(keys)), np.nan, dtype=float)
            for i, ki in enumerate(keys):
                for j, kj in enumerate(keys):
                    if i == j:
                        M[i, j] = 0.0
                        continue
                    if mode == "raw input":
                        A, B = samples[ki]["X"], samples[kj]["X"]
                    elif mode == "raw joint":
                        A, B = samples[ki]["J"], samples[kj]["J"]
                    elif mode == "mean/std":
                        A, B = feats[ki]["mean/std"], feats[kj]["mean/std"]
                    else:
                        A, B = feats[ki]["alpha/beta"], feats[kj]["alpha/beta"]
                    M[i, j] = np.nan if (A.size == 0 or B.size == 0) else energy_distance_multivariate(A, B)
            return M

        return {m: pairwise_matrix(m) for m in ["raw input", "raw joint", "mean/std", "alpha/beta"]}

    def compute_all_norms(L, H, N_ref, ignore_cte):
        return {nm: compute_matrices_for_norm(L, H, N_ref, ignore_cte, nm) for nm in ["raw", "standard", "instance"]}

    def ensure_cached(force=False):
        L, H, N_ref = int(applied["L"]), int(applied["H"]), int(applied["N_ref"])
        ignore_cte = bool(filter_button.value)
        params = (L, H, N_ref, ignore_cte)
        if force or cache["params"] != params:
            cache["matrices"] = compute_all_norms(L, H, N_ref, ignore_cte)
            cache["params"] = params

    def plot_current():
        with output:
            output.clear_output(wait=True)
            ensure_cached(force=False)

            mats_all = cache["matrices"]
            if mats_all is None:
                print("No data.")
                return

            mats = mats_all.get(norm_dropdown.value, None)
            if mats is None:
                print("No data.")
                return

            M = mats[dist_dropdown.value]
            plt.figure(figsize=(4 + 0.35 * len(keys), 3 + 0.25 * len(keys)))
            im = plt.imshow(M, aspect="auto")
            plt.colorbar(im)
            plt.xticks(np.arange(len(keys)), keys, rotation=45, ha="right")
            plt.yticks(np.arange(len(keys)), keys)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    plt.text(j, i, "nan" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center")
            plt.title(f"Distances ({dist_dropdown.value}) | Norm={norm_dropdown.value} | L={int(applied['L'])}, H={int(applied['H'])}, N_ref={int(applied['N_ref'])}")
            plt.tight_layout()
            plt.show()

    def on_apply(_):
        applied["L"] = int(L_widget.value)
        applied["H"] = int(H_widget.value)
        applied["N_ref"] = int(N_widget.value)
        ensure_cached(force=True)
        plot_current()

    def on_choice_change(_):
        plot_current()

    for w in [filter_button, norm_dropdown, dist_dropdown]:
        w.observe(on_choice_change, names="value")
    apply_button.on_click(on_apply)

    ensure_cached(force=True)
    display(
        widgets.HBox([filter_button, norm_dropdown, dist_dropdown]),
        widgets.HBox([L_widget, H_widget, N_widget, apply_button]),
        output,
    )
    plot_current()

def plot_distances(distances_matrix, show=True, path="", name="distances.pdf"):
    """Plots histogram of distances."""
    plt.figure(figsize=(10, 4))
    plt.hist(distances_matrix[np.triu_indices(distances_matrix.shape[0], k=1)], bins=100)
    plt.title("Distances histogram")
    plt.xlabel("Distances")
    plt.ylabel("Counts")
    if show:
        plt.show()
    else:
        plt.savefig(path + name)
    plt.close()

def plot_dendogram(Z, show=True, path="", name="dendogram.pdf"):
    """Plots dendrogram."""
    plt.figure(figsize=(15, 4))
    shc.dendrogram(Z)
    plt.title("Dendogram")
    plt.xticks([])
    plt.xlabel("")
    if show:
        plt.show()
    else:
        plt.savefig(path + name)
    plt.close()

def plot_centroids(centroids, show=True, path="", name="centroids.pdf"):
    """Plots cluster centroids."""
    plt.figure(figsize=(15, 4))
    for i, centroid in enumerate(centroids):
        plt.plot(centroid, label=f"Cluster {i + 1}")
    plt.title("Centroids of clusters")
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig(path + name)
    plt.close()

def plot_heterogeneity(df, show=True, path="", name="heterogeneity.pdf", N_clusters=None, seed=None):
    """Plots heterogeneity vs number of clusters."""
    set_seed(seed)
    heterogeneities = []
    if N_clusters is None:
        N_clusters = [1, 2, 3, 4, 5, 10, 20, df.shape[1] // 10, df.shape[1] // 5, df.shape[1] // 2, df.shape[1]]
    N_clusters = np.sort(N_clusters)
    Z, _ = init_clusters(df)
    for n_clusters in tqdm(N_clusters):
        _, cluster_indices = get_clusters(Z, n_clusters)
        heterogeneities.append(get_cluster_heterogeneity(df, cluster_indices))
    plt.figure(figsize=(6, 4))
    plt.plot(N_clusters, heterogeneities)
    plt.xlabel("Number of clusters")
    plt.ylabel("Heterogeneity")
    if show:
        plt.show()
    else:
        plt.savefig(path + name)
    plt.close()

def plot_centroids_widget(df):
    """Centroid plotting widget with apply-gated numeric params."""
    cache = {"dfs": {}, "gamma_params": None}
    applied = {"n_clusters": 3}

    dataset_dropdown = widgets.Dropdown(options=["raw", "fourier", "gamma"], value="fourier", description="Data:")
    lags_dropdown = widgets.Dropdown(options=[24, 168, 336, 504, 672], value=168, description="Lags:")
    horizon_dropdown = widgets.Dropdown(options=[24, 168, 336, 504, 672], value=24, description="Horizon:")
    n_clusters_slider = widgets.IntSlider(min=2, max=min(30, df.shape[1]), step=1, value=applied["n_clusters"], description="Clusters:", continuous_update=False)
    apply_button = widgets.Button(description="Apply")
    output = widgets.Output()

    cache["dfs"] = compute_views_df(df, lags=int(lags_dropdown.value), horizon=int(horizon_dropdown.value))
    cache["gamma_params"] = (int(lags_dropdown.value), int(horizon_dropdown.value))

    def toggle_gamma_ui():
        show = dataset_dropdown.value == "gamma"
        lags_dropdown.layout.display = "block" if show else "none"
        horizon_dropdown.layout.display = "block" if show else "none"

    def update_plot():
        with output:
            output.clear_output(wait=True)
            if dataset_dropdown.value == "gamma":
                ensure_gamma_view(cache, df, lags_dropdown.value, horizon_dropdown.value)
            current_df = cache["dfs"][dataset_dropdown.value]
            n_clusters = min(int(applied["n_clusters"]), current_df.shape[1])
            Z, _ = init_clusters(current_df)
            _, cluster_indices = get_clusters(Z, n_clusters)
            print("Feature-based centroids:")
            plot_centroids(get_centroids(current_df, cluster_indices))
            if dataset_dropdown.value != "raw":
                print("Raw centroids:")
                plot_centroids(get_centroids(df, cluster_indices))

    def on_apply(_):
        applied["n_clusters"] = int(n_clusters_slider.value)
        update_plot()

    def on_choice_change(_):
        toggle_gamma_ui()
        update_plot()

    for w in [dataset_dropdown, lags_dropdown, horizon_dropdown]:
        w.observe(on_choice_change, names="value")
    apply_button.on_click(on_apply)

    toggle_gamma_ui()
    display(
        widgets.HBox([dataset_dropdown, lags_dropdown, horizon_dropdown]),
        widgets.HBox([n_clusters_slider, apply_button]),
        output,
    )
    update_plot()

def plot_clustering_widget(df):
    """Clustering diagnostics widget with apply button."""
    cache = {"dfs": {}, "gamma_params": None, "Z": {"raw": None, "fourier": None, "gamma": None}, "dist": {"raw": None, "fourier": None, "gamma": None}}
    dataset_dropdown = widgets.Dropdown(options=["raw", "fourier", "gamma"], value="fourier", description="Data:")
    lags_dropdown = widgets.Dropdown(options=[24, 168, 336, 504, 672], value=168, description="Lags:")
    horizon_dropdown = widgets.Dropdown(options=[24, 168, 336, 504, 672], value=24, description="Horizon:")
    apply_button = widgets.Button(description="Apply")
    output = widgets.Output()

    cache["dfs"] = compute_views_df(df, lags=int(lags_dropdown.value), horizon=int(horizon_dropdown.value))
    cache["gamma_params"] = (int(lags_dropdown.value), int(horizon_dropdown.value))

    def toggle_gamma_ui():
        show = dataset_dropdown.value == "gamma"
        lags_dropdown.layout.display = "block" if show else "none"
        horizon_dropdown.layout.display = "block" if show else "none"

    def compute_Z(dataset_type):
        if dataset_type == "gamma":
            ensure_gamma_view(cache, df, lags_dropdown.value, horizon_dropdown.value)
        Z, D = init_clusters(cache["dfs"][dataset_type])
        cache["Z"][dataset_type], cache["dist"][dataset_type] = Z, D

    def update_plot(force=False):
        with output:
            output.clear_output(wait=True)
            dt = dataset_dropdown.value
            if force or cache["Z"][dt] is None or dt == "gamma":
                compute_Z(dt)
            print(f"Dendrogram ({dt}):")
            plot_dendogram(cache["Z"][dt])
            print(f"Distances ({dt}):")
            plot_distances(cache["dist"][dt])

    def on_apply(_):
        update_plot(force=True)

    def on_choice_change(_):
        toggle_gamma_ui()
        update_plot(force=False)

    for w in [dataset_dropdown, lags_dropdown, horizon_dropdown]:
        w.observe(on_choice_change, names="value")
    apply_button.on_click(on_apply)

    toggle_gamma_ui()
    display(widgets.HBox([dataset_dropdown, lags_dropdown, horizon_dropdown, apply_button]), output)
    update_plot(force=True)

def plots_stats_clusters_widget(df, seed=None):
    """Joint stats widget colored by clusters with apply-gated numeric params."""
    cache = {"dfs": {}, "gamma_params": None, "stats_params": None, "stats_df": None, "cluster_params": None, "labels": None, "plot_df": None}
    columns = list(df.columns)

    dataset_dropdown = widgets.Dropdown(options=["raw", "fourier", "gamma"], value="fourier", description="Data:")
    lags_dropdown = widgets.Dropdown(options=[24, 168, 336, 504, 672], value=168, description="Lags:")
    horizon_dropdown = widgets.Dropdown(options=[24, 168, 336, 504, 672], value=24, description="Horizon:")

    type_dropdown = widgets.Dropdown(options=["Mean/Std", "Alpha/Beta"], value="Mean/Std", description="Type:")
    log_button = widgets.ToggleButton(value=True, description="Log")
    filter_button = widgets.ToggleButton(value=False, description="Filter cte")

    n_clusters_slider = widgets.IntSlider(min=2, max=min(30, df.shape[1]), step=1, value=3, description="Clusters:", continuous_update=False)
    L_widget = widgets.IntText(value=168, description="L:")
    H_widget = widgets.IntText(value=24, description="H:")
    N_widget = widgets.IntText(value=100, description="N:")
    apply_button = widgets.Button(description="Apply")
    output = widgets.Output()

    applied = {"n_clusters": int(n_clusters_slider.value), "L": int(L_widget.value), "H": int(H_widget.value), "N": int(N_widget.value)}

    cache["dfs"] = compute_views_df(df, lags=int(lags_dropdown.value), horizon=int(horizon_dropdown.value))
    cache["gamma_params"] = (int(lags_dropdown.value), int(horizon_dropdown.value))

    def ensure_gamma():
        if dataset_dropdown.value == "gamma":
            ensure_gamma_view(cache, df, lags_dropdown.value, horizon_dropdown.value)

    def toggle_gamma_ui():
        show = dataset_dropdown.value == "gamma"
        lags_dropdown.layout.display = "block" if show else "none"
        horizon_dropdown.layout.display = "block" if show else "none"

    def precompute_stats():
        set_seed(seed)
        L, H, N = int(applied["L"]), int(applied["H"]), int(applied["N"])
        plot_type = type_dropdown.value
        ignore_cte = bool(filter_button.value)
        records = []
        linthresh = 1

        for col in columns:
            X, Y = sample_windows_df(df, L, H, N, columns=[col], ignore_cte=ignore_cte, seed=seed)
            if plot_type == "Mean/Std":
                x_vals, y_vals = window_mean_std(X)
                x_name, y_name = "mean", "std"
            else:
                x_vals, y_vals = window_alpha_beta(X, Y)
                x_name, y_name = "alpha", "beta"
            records += [{"user": col, x_name: xv, y_name: yv} for xv, yv in zip(x_vals, y_vals)]

        if not records:
            cache["stats_df"] = None
            return

        df_temp = pd.DataFrame(records)
        if plot_type == "Mean/Std":
            df_temp["mean_symlog"] = symlog(df_temp["mean"], linthresh=linthresh)
            df_temp["std_symlog"] = symlog(df_temp["std"], linthresh=linthresh)
        else:
            df_temp["alpha_symlog"] = symlog(df_temp["alpha"], linthresh=linthresh)
            df_temp["beta_symlog"] = symlog(df_temp["beta"], linthresh=linthresh)
        cache["stats_df"] = df_temp

    def compute_labels():
        ensure_gamma()
        dt = dataset_dropdown.value
        n_clusters = min(int(applied["n_clusters"]), cache["dfs"][dt].shape[1])
        Z, _ = init_clusters(cache["dfs"][dt])
        labels, _ = get_clusters(Z, n_clusters)
        cache["labels"] = labels

    def build_plot_df():
        if cache["stats_df"] is None:
            cache["plot_df"] = None
            return
        compute_labels()
        labels = cache["labels"]
        label_map = {columns[i]: labels[i] for i in range(len(columns))}
        plot_df = cache["stats_df"].copy()
        plot_df["cluster"] = plot_df["user"].map(lambda u: f"c{label_map.get(u, 0)}")
        cache["plot_df"] = plot_df

    def update_plot():
        with output:
            output.clear_output(wait=True)

            stats_params = (int(applied["L"]), int(applied["H"]), int(applied["N"]), type_dropdown.value, bool(filter_button.value))
            if cache["stats_params"] != stats_params:
                cache["stats_params"] = stats_params
                precompute_stats()

            cluster_params = (dataset_dropdown.value, int(applied["n_clusters"]), int(lags_dropdown.value), int(horizon_dropdown.value), type_dropdown.value)
            if cache["cluster_params"] != cluster_params or cache["plot_df"] is None:
                cache["cluster_params"] = cluster_params
                build_plot_df()

            plot_df = cache["plot_df"]
            if plot_df is None or plot_df.empty:
                print("No data.")
                return

            plot_type = type_dropdown.value
            use_log = bool(log_button.value)

            if plot_type == "Mean/Std":
                x_base, y_base = "mean", "std"
                x_label_raw, y_label_raw = "Mean", "Std"
            else:
                x_base, y_base = "beta", "alpha"
                x_label_raw, y_label_raw = "Beta", "Alpha"

            if use_log:
                x_col, y_col = f"{x_base}_symlog", f"{y_base}_symlog"
                x_label, y_label = f"{x_label_raw} (log)", f"{y_label_raw} (log)"
            else:
                x_col, y_col = x_base, y_base
                x_label, y_label = x_label_raw, y_label_raw

            g = sns.jointplot(
                data=plot_df,
                x=x_col,
                y=y_col,
                hue="cluster",
                kind="scatter",
                height=7,
                s=20,
                marginal_kws=dict(common_norm=False, fill=True, alpha=0.5),
            )
            g.ax_joint.set_xlabel(x_label)
            g.ax_joint.set_ylabel(y_label)
            g.fig.suptitle(f"{dataset_dropdown.value} | clusters={int(applied['n_clusters'])} | L={int(applied['L'])}, H={int(applied['H'])}, N={int(applied['N'])} - {plot_type}", y=1.02)
            plt.show()

    def on_apply(_):
        applied["n_clusters"] = int(n_clusters_slider.value)
        applied["L"] = int(L_widget.value)
        applied["H"] = int(H_widget.value)
        applied["N"] = int(N_widget.value)
        update_plot()

    def on_choice_change(_):
        toggle_gamma_ui()
        update_plot()

    for w in [dataset_dropdown, lags_dropdown, horizon_dropdown, type_dropdown, log_button, filter_button]:
        w.observe(lambda _: on_choice_change(None), names="value")
    apply_button.on_click(on_apply)

    toggle_gamma_ui()
    display(
        widgets.HBox([dataset_dropdown, lags_dropdown, horizon_dropdown]),
        widgets.HBox([type_dropdown, log_button, filter_button]),
        widgets.HBox([n_clusters_slider]),
        widgets.HBox([L_widget, H_widget, N_widget, apply_button]),
        output,
    )
    on_apply(None)

def plot_tsne_dict_widget(df_dict, seed=None):
    """t-SNE widget over datasets with apply-gated numeric params and shared helpers."""
    assert "train" in df_dict, "df_dict must include a 'train' dataframe for standard normalization."

    keys = list(df_dict.keys())
    first_key = keys[0]

    cache = {"params": None, "train_stats": None, "embeddings": None}
    applied = {"L": 168, "H": 24, "N_ref": 200, "perplexity": 30, "lr": 200}

    mode_dropdown = widgets.Dropdown(options=[("inputs only", "inputs"), ("inputs + outputs (joint)", "joint")], value="inputs", description="Data:")
    norm_dropdown = widgets.Dropdown(options=["raw", "standard", "instance"], value="raw", description="Norm:")
    filter_button = widgets.ToggleButton(value=False, description="Filter cte")

    L_widget = widgets.IntText(value=applied["L"], description="L:")
    H_widget = widgets.IntText(value=applied["H"], description="H:")
    N_widget = widgets.IntText(value=applied["N_ref"], description="N (ref):")
    perplexity_widget = widgets.IntText(value=applied["perplexity"], description="Perp:")
    lr_widget = widgets.IntText(value=applied["lr"], description="LR:")
    apply_button = widgets.Button(description="Apply")

    output = widgets.Output()

    def compute_tsne(A, perplexity, lr):
        if A.size == 0 or A.shape[0] < 2:
            return np.empty((0, 2))
        perp = max(2, min(int(perplexity), (A.shape[0] - 1) // 3 if A.shape[0] > 6 else 2))
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate=float(lr),
            init="pca",
            random_state=None if seed is None else int(seed),
            max_iter=1000,
            verbose=0,
        )
        return tsne.fit_transform(A)

    def ensure_cached(force=False):
        params = (int(applied["L"]), int(applied["H"]), int(applied["N_ref"]), bool(filter_button.value), mode_dropdown.value, int(applied["perplexity"]), int(applied["lr"]))
        if not force and cache["params"] == params:
            return

        train_stats = get_train_stats(df_dict, cache)
        embs = {}
        for nm in ["raw", "standard", "instance"]:
            A, labels, sizes = build_concat_samples(
                df_dict,
                keys,
                first_key,
                L=int(applied["L"]),
                H=int(applied["H"]),
                N_ref=int(applied["N_ref"]),
                ignore_cte=bool(filter_button.value),
                mode=mode_dropdown.value,
                norm_mode=nm,
                seed=seed,
                train_stats=train_stats if nm == "standard" else None,
            )
            embs[nm] = {"Z": compute_tsne(A, applied["perplexity"], applied["lr"]), "labels": labels, "sizes": sizes}
        cache["embeddings"] = embs
        cache["params"] = params

    def plot_current():
        with output:
            output.clear_output(wait=True)
            ensure_cached(force=False)

            pack = None if cache["embeddings"] is None else cache["embeddings"].get(norm_dropdown.value, None)
            if pack is None or pack["Z"].size == 0:
                print("No samples (check L/H/N_ref, or data too short).")
                return

            Z, labels, sizes = pack["Z"], pack["labels"], pack["sizes"]
            key_to_color = {k: f"C{i}" for i, k in enumerate(keys)}

            plt.figure(figsize=(7, 6))
            for k in keys:
                mask = labels == k
                if not np.any(mask):
                    continue
                plt.scatter(Z[mask, 0], Z[mask, 1], s=14, alpha=(0.85 if k == first_key else 0.25), c=key_to_color[k], label=f"{k} (n={sizes.get(k, 0)})", edgecolors="none")

            plt.title(
                f"t-SNE | mode={mode_dropdown.value} | norm={norm_dropdown.value} | "
                f"L={int(applied['L'])}, H={int(applied['H'])}, N_ref={int(applied['N_ref'])}, "
                f"filter_cte={bool(filter_button.value)} | perp={int(applied['perplexity'])}, lr={int(applied['lr'])}"
            )
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.grid(True, alpha=0.2)
            plt.legend(loc="best", frameon=True)
            plt.tight_layout()
            plt.show()

    def on_apply(_):
        applied["L"] = int(L_widget.value)
        applied["H"] = int(H_widget.value)
        applied["N_ref"] = int(N_widget.value)
        applied["perplexity"] = int(perplexity_widget.value)
        applied["lr"] = int(lr_widget.value)
        ensure_cached(force=True)
        plot_current()

    def on_choice_change(_):
        plot_current()

    for w in [norm_dropdown, mode_dropdown, filter_button]:
        w.observe(on_choice_change, names="value")
    apply_button.on_click(on_apply)

    ensure_cached(force=True)
    display(
        widgets.HBox([mode_dropdown, norm_dropdown, filter_button]),
        widgets.HBox([L_widget, H_widget, N_widget, perplexity_widget, lr_widget, apply_button]),
        output,
    )
    plot_current()

def plot_users_widget(df, seed=None):
    """2D user scatter via MDS on distance matrices from raw/fourier/gamma views.

    Adds an optional highlight list (comma-separated iloc positions / ranges) plotted in red.
    Highlighting never triggers embedding recomputation.
    """
    columns = list(df.columns)
    cache = {"dfs": {}, "gamma_params": None, "params": None, "embeddings": None}
    applied = {"n_init": 4, "max_iter": 300}
    state = {"highlight_idx": set()}

    datatype_dropdown = widgets.Dropdown(options=["raw", "fourier", "gamma"], value="raw", description="Data:")
    metric_dropdown = widgets.Dropdown(options=["cosine", "euclidean"], value="cosine", description="Metric:")
    lags_dropdown = widgets.Dropdown(options=[24, 168, 336, 504, 672], value=168, description="Lags:")
    horizon_dropdown = widgets.Dropdown(options=[24, 168, 336, 504, 672], value=24, description="Horizon:")
    n_init_widget = widgets.IntText(value=applied["n_init"], description="n_init:")
    max_iter_widget = widgets.IntText(value=applied["max_iter"], description="max_iter:")
    highlight_text = widgets.Text(value="", description="Highlight:", placeholder="e.g. 1,2,3,100-103,109")
    apply_button = widgets.Button(description="Apply")
    output = widgets.Output()

    cache["dfs"] = compute_views_df(df, lags=int(lags_dropdown.value), horizon=int(horizon_dropdown.value))
    cache["gamma_params"] = (int(lags_dropdown.value), int(horizon_dropdown.value))

    col_to_idx = {str(c): i for i, c in enumerate(columns)}

    def parse_highlight(s):
        """Parse comma-separated iloc positions and ranges (1-based).

        Examples:
        - "1,2,10"
        - "100-103, 109, 200-205"

        Returns a set of 0-based integer indices suitable for df.iloc[:, idx].
        """
        if s is None:
            return set()
        s = str(s).strip()
        if not s:
            return set()

        n = len(columns)
        toks = [t.strip() for t in s.replace(";", ",").split(",")]
        idx = set()

        for t in toks:
            if not t:
                continue

            # range token
            if "-" in t:
                parts = [p.strip() for p in t.split("-", 1)]
                if len(parts) == 2 and parts[0] and parts[1]:
                    try:
                        a = int(parts[0])
                        b = int(parts[1])
                        if a > b:
                            a, b = b, a
                        for pos in range(a, b + 1):
                            if 1 <= pos <= n:
                                idx.add(pos - 1)
                    except Exception:
                        pass
                continue

            # single position
            try:
                pos = int(t)
                if 1 <= pos <= n:
                    idx.add(pos - 1)
            except Exception:
                pass

        return idx

    def toggle_gamma_ui():
        show = datatype_dropdown.value == "gamma"
        lags_dropdown.layout.display = "block" if show else "none"
        horizon_dropdown.layout.display = "block" if show else "none"

    def compute_mds(df_view, metric, n_init, max_iter):
        D = squareform(calculate_distances(df_view, metric=metric))
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            n_init=int(n_init),
            max_iter=int(max_iter),
            random_state=None if seed is None else int(seed),
            normalized_stress="auto",
        )
        return mds.fit_transform(D)

    def ensure_cached(force=False):
        ensure_gamma_view(cache, df, lags_dropdown.value, horizon_dropdown.value)
        gp = tuple(cache["gamma_params"]) if cache.get("gamma_params") is not None else None
        params = (int(applied["n_init"]), int(applied["max_iter"]), gp, None if seed is None else int(seed))
        if not force and cache["params"] == params:
            return
        embs = {dt: {} for dt in ["raw", "fourier", "gamma"]}
        for dt in ["raw", "fourier", "gamma"]:
            df_view = cache["dfs"][dt]
            for met in ["cosine", "euclidean"]:
                embs[dt][met] = compute_mds(df_view, met, n_init=applied["n_init"], max_iter=applied["max_iter"])
        cache["embeddings"] = embs
        cache["params"] = params

    def plot_current():
        with output:
            output.clear_output(wait=True)
            ensure_cached(force=False)
            Z = None if cache["embeddings"] is None else cache["embeddings"].get(datatype_dropdown.value, {}).get(metric_dropdown.value, None)
            if Z is None or getattr(Z, "size", 0) == 0 or Z.shape[0] != len(columns):
                print("No embedding.")
                return

            colors = np.array(["C0"] * len(columns), dtype=object)
            if state["highlight_idx"]:
                for i in state["highlight_idx"]:
                    if 0 <= int(i) < len(columns):
                        colors[int(i)] = "red"

            plt.figure(figsize=(7, 6))
            plt.scatter(Z[:, 0], Z[:, 1], s=22, alpha=0.5, c=colors)
            title = f"MDS users | data={datatype_dropdown.value} | metric={metric_dropdown.value} | n_init={int(applied['n_init'])} | max_iter={int(applied['max_iter'])}"
            if state["highlight_idx"]:
                title += f" | highlighted={len(state['highlight_idx'])}"
            plt.title(title)
            plt.xlabel("MDS 1")
            plt.ylabel("MDS 2")
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.show()

    def on_apply(_):
        # update highlight (never forces recompute)
        state["highlight_idx"] = parse_highlight(highlight_text.value)

        new_n_init = int(n_init_widget.value)
        new_max_iter = int(max_iter_widget.value)
        need_recompute = (new_n_init != int(applied["n_init"])) or (new_max_iter != int(applied["max_iter"]))
        applied["n_init"] = new_n_init
        applied["max_iter"] = new_max_iter

        ensure_cached(force=need_recompute)
        plot_current()

    def on_choice_change(_):
        toggle_gamma_ui()
        plot_current()

    for w in [datatype_dropdown, metric_dropdown, lags_dropdown, horizon_dropdown]:
        w.observe(on_choice_change, names="value")
    apply_button.on_click(on_apply)

    toggle_gamma_ui()
    ensure_cached(force=True)
    display(
        widgets.HBox([datatype_dropdown, metric_dropdown, lags_dropdown, horizon_dropdown]),
        widgets.HBox([n_init_widget, max_iter_widget, highlight_text, apply_button]),
        output,
    )
    plot_current()

def plot_umap_dict_widget(df_dict, seed=None):
    """UMAP widget over datasets with apply-gated numeric params and cached norms."""
    try:
        import umap.umap_ as umap
    except Exception as e:
        raise ImportError("UMAP requires `umap-learn` (pip install umap-learn).") from e

    assert "train" in df_dict, "df_dict must include a 'train' dataframe for standard normalization."

    keys = list(df_dict.keys())
    first_key = keys[0]

    cache = {"params": None, "train_stats": None, "embeddings": None}
    applied = {"L": 168, "H": 24, "N_ref": 200, "n_neighbors": 15, "min_dist": 0.1}

    mode_dropdown = widgets.Dropdown(options=[("inputs only", "inputs"), ("inputs + outputs (joint)", "joint")], value="inputs", description="Data:")
    norm_dropdown = widgets.Dropdown(options=["raw", "standard", "instance"], value="raw", description="Norm:")
    filter_button = widgets.ToggleButton(value=False, description="Filter cte")

    L_widget = widgets.IntText(value=applied["L"], description="L:")
    H_widget = widgets.IntText(value=applied["H"], description="H:")
    N_widget = widgets.IntText(value=applied["N_ref"], description="N (ref):")
    n_neighbors_widget = widgets.IntText(value=applied["n_neighbors"], description="k:")
    min_dist_widget = widgets.FloatText(value=applied["min_dist"], description="min_d:")
    apply_button = widgets.Button(description="Apply")
    output = widgets.Output()

    def compute_umap(A, n_neighbors, min_dist):
        if A.size == 0 or A.shape[0] < 2:
            return np.empty((0, 2))
        nn = max(2, min(int(n_neighbors), max(2, A.shape[0] - 1)))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=nn,
            min_dist=float(min_dist),
            random_state=None if seed is None else int(seed),
        )
        return reducer.fit_transform(A)

    def ensure_cached(force=False):
        params = (
            int(applied["L"]),
            int(applied["H"]),
            int(applied["N_ref"]),
            bool(filter_button.value),
            mode_dropdown.value,
            int(applied["n_neighbors"]),
            float(applied["min_dist"]),
        )
        if not force and cache["params"] == params:
            return

        train_stats = get_train_stats(df_dict, cache)
        embs = {}
        for nm in ["raw", "standard", "instance"]:
            A, labels, sizes = build_concat_samples(
                df_dict,
                keys,
                first_key,
                L=int(applied["L"]),
                H=int(applied["H"]),
                N_ref=int(applied["N_ref"]),
                ignore_cte=bool(filter_button.value),
                mode=mode_dropdown.value,
                norm_mode=nm,
                seed=seed,
                train_stats=train_stats if nm == "standard" else None,
            )
            embs[nm] = {"Z": compute_umap(A, applied["n_neighbors"], applied["min_dist"]), "labels": labels, "sizes": sizes}
        cache["embeddings"] = embs
        cache["params"] = params

    def plot_current():
        with output:
            output.clear_output(wait=True)
            ensure_cached(force=False)

            pack = None if cache["embeddings"] is None else cache["embeddings"].get(norm_dropdown.value, None)
            if pack is None or pack["Z"].size == 0:
                print("No samples (check L/H/N_ref, or data too short).")
                return

            Z, labels, sizes = pack["Z"], pack["labels"], pack["sizes"]
            key_to_color = {k: f"C{i}" for i, k in enumerate(keys)}

            plt.figure(figsize=(7, 6))
            for k in keys:
                mask = labels == k
                if not np.any(mask):
                    continue
                plt.scatter(Z[mask, 0], Z[mask, 1], s=14, alpha=(0.85 if k == first_key else 0.25), c=key_to_color[k], label=f"{k} (n={sizes.get(k, 0)})", edgecolors="none")

            plt.title(
                f"UMAP | mode={mode_dropdown.value} | norm={norm_dropdown.value} | "
                f"L={int(applied['L'])}, H={int(applied['H'])}, N_ref={int(applied['N_ref'])}, "
                f"filter_cte={bool(filter_button.value)} | k={int(applied['n_neighbors'])}, min_d={float(applied['min_dist'])}"
            )
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.grid(True, alpha=0.2)
            plt.legend(loc="best", frameon=True)
            plt.tight_layout()
            plt.show()

    def on_apply(_):
        applied["L"] = int(L_widget.value)
        applied["H"] = int(H_widget.value)
        applied["N_ref"] = int(N_widget.value)
        applied["n_neighbors"] = int(n_neighbors_widget.value)
        applied["min_dist"] = float(min_dist_widget.value)
        ensure_cached(force=True)
        plot_current()

    def on_choice_change(_):
        plot_current()

    for w in [norm_dropdown, mode_dropdown, filter_button]:
        w.observe(on_choice_change, names="value")
    apply_button.on_click(on_apply)

    ensure_cached(force=True)
    display(
        widgets.HBox([mode_dropdown, norm_dropdown, filter_button]),
        widgets.HBox([L_widget, H_widget, N_widget, n_neighbors_widget, min_dist_widget, apply_button]),
        output,
    )
    plot_current()
