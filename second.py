# -*- coding: utf-8 -*-
"""
ONE combined model-comparison table (all years/months pooled together - not
split per month), for dropping straight into a slide or paper.

This is intentionally a separate, self-contained script from the plotting
scripts: it doesn't produce any per-month output, just a single scorecard
covering the whole record, plus the CSV behind it.

Metrics computed per model (all against the 2km ground truth, on the GT
grid): Bias, PBIAS, RMSE, Corr (r), KGE, Mean FSS, and Mean CSI (Threat
Score). The raw 8km ERA5 input is included as a baseline row (interpolated
onto the 2km grid) for context, but is excluded from the "best" highlighting
since it isn't a downscaling method being ranked.
"""
import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import to_rgba
from scipy.ndimage import uniform_filter

# =============================================================================
# 0. CONFIG
# =============================================================================
GT_FILES = [
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Singapore_Data/GT_from_Prasanna/1995/pr_V3-WMC-2_ERA5_historical_reanalysis_SINGV-RCM_vn5_day_19950601-19950630.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Singapore_Data/GT_from_Prasanna/1995/pr_V3-WMC-2_ERA5_historical_reanalysis_SINGV-RCM_vn5_day_19950501-19950531.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Singapore_Data/GT_from_Prasanna/1995/pr_V3-WMC-2_ERA5_historical_reanalysis_SINGV-RCM_vn5_day_19950801-19950831.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Singapore_Data/GT_from_Prasanna/2005/pr_V3-WMC-2_ERA5_historical_reanalysis_SINGV-RCM_vn5_day_20050501-20050531.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Singapore_Data/GT_from_Prasanna/2005/pr_V3-WMC-2_ERA5_historical_reanalysis_SINGV-RCM_vn5_day_20050701-20050731.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Singapore_Data/GT_from_Prasanna/2014/pr_V3-WMC-2_ERA5_historical_reanalysis_SINGV-RCM_vn5_day_20141001-20141031.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Singapore_Data/GT_from_Prasanna/2014/pr_V3-WMC-2_ERA5_historical_reanalysis_SINGV-RCM_vn5_day_20140501-20140531.nc",
]

LR_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/precip_rcm_8km_daily_data_1995-2014_mmday_corrected.nc"

# Same palette/assignment as the other two scripts, so a model is always the
# same color everywhere you look at it.
MODELS = [
    {
        "name": "fm", "label": "2km CorrDiff (Flow Matching)", "color": "#1f77b4",
        "template": "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/VERSION5/corrdiff_fm/outputs/downscaled_2km_fm/precip_2km_{year}.nc",
        "var": "precip_mean",
    },
    {
        "name": "srdrn", "label": "2km SRDRN", "color": "#2ca02c",
        "template": "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/VERSION5/srdrn/outputs/downscaled_2km_srdrn/precip_2km_{year}.nc",
        "var": "precip_mean",
    },
    {
        "name": "srgan", "label": "2km SRGAN", "color": "#ff7f0e",
        "template": "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/VERSION5/srgan/outputs/downscaled_2km_srgan/precip_2km_{year}.nc",
        "var": "precip_mean",
    },
    {
        "name": "edm", "label": "2km CorrDiff (EDM)", "color": "#d62728",
        "template": "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/VERSION5/corrdiff_edm/outputs/downscaled_2km_edm/precip_2km_{year}.nc",
        "var": None,
    },
    {
        "name": "wgan_gp", "label": "2km WGAN-GP", "color": "#9467bd",
        "template": "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/VERSION5/wgan_gp/outputs/downscaled_2km_wgan_gp/precip_2km_{year}.nc",
        "var": "precip_mean",
    },
]

INCLUDE_LR_BASELINE = True  # show the raw 8km input as a non-ranked context row

OUT_DIR = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/VERSION5/images/"

MODEL_DX_KM = 2.0
THRESHOLDS = [1.0, 5.0, 10.0, 20.0]                      # mm/day, for FSS + CSI
WINDOW_SIZES = np.array([1, 3, 5, 11, 21, 31, 41, 51])   # pixels, for FSS

# Speed knob. FSS loops over every day x window x threshold, which can get
# slow over a long combined record. 1 = use every day (slowest, most
# accurate); raise to 2-3 if a run is too slow for your queue.
FSS_DAY_STRIDE = 1

FILENAME_RE = re.compile(r"_day_(\d{8})-(\d{8})\.nc$")
VAR_CANDIDATES = ("precip_mean", "precip", "pr", "sr", "output", "downscaled")

# Which direction is "better" for each metric column, used to pick which
# cell to highlight in the rendered table.
METRIC_DIRECTIONS = {
    "Bias (mm/day)": "abs_min",
    "PBIAS (%)": "abs_min",
    "RMSE (mm/day)": "min",
    "Corr (r)": "max",
    "KGE": "max",
    "Mean FSS": "max",
    "Mean CSI": "max",
}


# =============================================================================
# 1. FILE GROUPING / VARIABLE RESOLUTION (same conventions as the other scripts)
# =============================================================================
def group_gt_files_by_year(gt_files):
    grouped = {}
    for path in gt_files:
        m = FILENAME_RE.search(path)
        if not m:
            print(f"  [skip] Could not parse date range from filename: {path}")
            continue
        start_date = datetime.strptime(m.group(1), "%Y%m%d")
        end_date = datetime.strptime(m.group(2), "%Y%m%d")
        grouped.setdefault(start_date.year, []).append((start_date, end_date, path))
    for year in grouped:
        grouped[year].sort(key=lambda x: x[0])
    return grouped


def get_model_var(ds, model_cfg):
    if model_cfg.get("var"):
        return model_cfg["var"]
    for cand in VAR_CANDIDATES:
        if cand in ds.data_vars:
            return cand
    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        print(f"  [warn] '{model_cfg['name']}': falling back to only variable '{data_vars[0]}'.")
        return data_vars[0]
    raise KeyError(f"Can't tell which variable to read for '{model_cfg['name']}'. Available: {data_vars}")


def year_time_slice(ds_full, year, start_date, end_date):
    jan1 = datetime(year, 1, 1)
    return ds_full.isel(time=slice((start_date - jan1).days, (end_date - jan1).days + 1))


def lr_time_slice(ds_lr_full, start_date, end_date):
    try:
        return ds_lr_full.sel(time=slice(pd.Timestamp(start_date), pd.Timestamp(end_date)))
    except Exception:
        lr_start = datetime(1995, 1, 1)
        return ds_lr_full.isel(time=slice((start_date - lr_start).days, (end_date - lr_start).days + 1))


# =============================================================================
# 2. METRIC FUNCTIONS
# =============================================================================
def compute_error_metrics(gt_stack, model_stack):
    """Bias, PBIAS, RMSE (pixel-day pooled), and Corr/KGE (on the domain-mean
    daily series, which is the conventional way KGE is applied - it wants a
    single time series, not a spatial field)."""
    valid = ~np.isnan(gt_stack) & ~np.isnan(model_stack)
    obs, pred = gt_stack[valid], model_stack[valid]
    diff = pred - obs
    bias = float(np.mean(diff)) if diff.size else np.nan
    rmse = float(np.sqrt(np.mean(diff ** 2))) if diff.size else np.nan
    pbias = float(100.0 * np.sum(diff) / np.sum(obs)) if diff.size and np.sum(obs) != 0 else np.nan

    gt_daily = np.nanmean(gt_stack, axis=(1, 2))
    model_daily = np.nanmean(model_stack, axis=(1, 2))
    vd = ~np.isnan(gt_daily) & ~np.isnan(model_daily)
    if vd.sum() > 1 and np.std(gt_daily[vd]) != 0 and np.mean(gt_daily[vd]) != 0:
        r = float(np.corrcoef(gt_daily[vd], model_daily[vd])[0, 1])
        alpha = float(np.std(model_daily[vd]) / np.std(gt_daily[vd]))
        beta = float(np.mean(model_daily[vd]) / np.mean(gt_daily[vd]))
        kge = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    else:
        r = kge = np.nan
    return bias, pbias, rmse, r, kge


def compute_mean_fss(gt_stack, model_stack, day_stride=1):
    idx = range(0, gt_stack.shape[0], day_stride)
    vals = []
    for thresh in THRESHOLDS:
        for window in WINDOW_SIZES:
            daily = []
            for t in idx:
                obs_bin = (gt_stack[t] >= thresh).astype(float)
                pred_bin = (model_stack[t] >= thresh).astype(float)
                obs_frac = uniform_filter(obs_bin, size=window, mode='constant', cval=0.0)
                pred_frac = uniform_filter(pred_bin, size=window, mode='constant', cval=0.0)
                mse = np.nanmean((obs_frac - pred_frac) ** 2)
                ref = np.nanmean(obs_frac ** 2 + pred_frac ** 2)
                if ref != 0:
                    daily.append(1.0 - mse / ref)
            if daily:
                vals.append(np.mean(daily))
    return float(np.mean(vals)) if vals else np.nan


def compute_mean_csi(gt_stack, model_stack, day_stride=1):
    """Critical Success Index / Threat Score, pixel-wise (no neighborhood
    tolerance, unlike FSS), averaged across THRESHOLDS."""
    idx = np.arange(0, gt_stack.shape[0], day_stride)
    obs, pred = gt_stack[idx], model_stack[idx]
    valid = ~np.isnan(obs) & ~np.isnan(pred)
    obs, pred = obs[valid], pred[valid]
    csis = []
    for thresh in THRESHOLDS:
        hits = np.sum((obs >= thresh) & (pred >= thresh))
        misses = np.sum((obs >= thresh) & (pred < thresh))
        false_alarms = np.sum((obs < thresh) & (pred >= thresh))
        denom = hits + misses + false_alarms
        if denom > 0:
            csis.append(hits / denom)
    return float(np.mean(csis)) if csis else np.nan


# =============================================================================
# 3. LOAD THE WHOLE RECORD (pooled across every year/month - no per-month split)
# =============================================================================
def load_full_record():
    grouped = group_gt_files_by_year(GT_FILES)

    gt_parts, lr_parts = [], []
    model_aligned_parts = {cfg["name"]: [] for cfg in MODELS}

    for year in sorted(grouped.keys()):
        model_datasets = {}
        for cfg in MODELS:
            p = cfg["template"].format(year=year)
            if os.path.exists(p):
                model_datasets[cfg["name"]] = (xr.open_dataset(p), cfg)
            else:
                print(f"  [skip] '{cfg['name']}' missing for {year}: {p}")

        ds_lr_full = xr.open_dataset(LR_PATH)

        for start_date, end_date, gt_path in grouped[year]:
            print(f"Loading {os.path.basename(gt_path)} ({start_date.date()} - {end_date.date()}) ...")
            ds_gt = xr.open_dataset(gt_path)
            gt_da = ds_gt["pr"] * 86400.0
            lat_b = sorted([float(gt_da.lat.min()), float(gt_da.lat.max())])
            lon_b = sorted([float(gt_da.lon.min()), float(gt_da.lon.max())])

            lr_month = lr_time_slice(ds_lr_full, start_date, end_date)
            lr_clip = lr_month.where(
                (lr_month.lat >= lat_b[0]) & (lr_month.lat <= lat_b[1]) &
                (lr_month.lon >= lon_b[0]) & (lr_month.lon <= lon_b[1]), drop=True)
            lr_aligned = lr_clip["pr"].interp(lat=gt_da.lat, lon=gt_da.lon, method="linear")

            n_days = min(len(gt_da.time), len(lr_aligned.time))
            month_aligned = {}
            for name, (ds_full, cfg) in model_datasets.items():
                m_month = year_time_slice(ds_full, year, start_date, end_date)
                m_clip = m_month.where(
                    (m_month.lat >= lat_b[0]) & (m_month.lat <= lat_b[1]) &
                    (m_month.lon >= lon_b[0]) & (m_month.lon <= lon_b[1]), drop=True)
                var = get_model_var(m_clip, cfg)
                aligned = m_clip[var].interp(lat=gt_da.lat, lon=gt_da.lon, method="linear")
                month_aligned[name] = aligned
                n_days = min(n_days, len(aligned.time))

            gt_parts.append(gt_da.isel(time=slice(0, n_days)).values)
            lr_parts.append(lr_aligned.isel(time=slice(0, n_days)).values)
            for name in model_datasets:
                model_aligned_parts[name].append(month_aligned[name].isel(time=slice(0, n_days)).values)

        for ds_full, _ in model_datasets.values():
            ds_full.close()
        ds_lr_full.close()

    if not gt_parts:
        return None

    gt_all = np.concatenate(gt_parts, axis=0)
    lr_all = np.concatenate(lr_parts, axis=0)
    model_aligned_all = {n: np.concatenate(v, axis=0) for n, v in model_aligned_parts.items() if v}
    print(f"\nCombined record: {gt_all.shape[0]} days total, pooled across every year/month.")
    return gt_all, lr_all, model_aligned_all


# =============================================================================
# 4. BUILD THE TABLE
# =============================================================================
def build_rows(gt_all, lr_all, model_aligned_all):
    rows = []
    if INCLUDE_LR_BASELINE:
        print("Computing metrics for baseline '8km ERA5 (Input)' ...")
        bias, pbias, rmse, r, kge = compute_error_metrics(gt_all, lr_all)
        mean_fss = compute_mean_fss(gt_all, lr_all, day_stride=FSS_DAY_STRIDE)
        mean_csi = compute_mean_csi(gt_all, lr_all, day_stride=FSS_DAY_STRIDE)
        rows.append({
            "name": "lr_baseline", "label": "8km ERA5 (Input, baseline)", "color": "#7f7f7f",
            "Bias (mm/day)": bias, "PBIAS (%)": pbias, "RMSE (mm/day)": rmse,
            "Corr (r)": r, "KGE": kge, "Mean FSS": mean_fss, "Mean CSI": mean_csi,
        })

    for cfg in MODELS:
        name = cfg["name"]
        if name not in model_aligned_all:
            print(f"  [skip] '{name}' - no data loaded across the whole record.")
            continue
        print(f"Computing metrics for '{cfg['label']}' ...")
        model_stack = model_aligned_all[name]
        bias, pbias, rmse, r, kge = compute_error_metrics(gt_all, model_stack)
        mean_fss = compute_mean_fss(gt_all, model_stack, day_stride=FSS_DAY_STRIDE)
        mean_csi = compute_mean_csi(gt_all, model_stack, day_stride=FSS_DAY_STRIDE)
        rows.append({
            "name": name, "label": cfg["label"], "color": cfg["color"],
            "Bias (mm/day)": bias, "PBIAS (%)": pbias, "RMSE (mm/day)": rmse,
            "Corr (r)": r, "KGE": kge, "Mean FSS": mean_fss, "Mean CSI": mean_csi,
        })
    return rows


def render_table_image(df, path):
    metric_cols = [c for c in df.columns if c not in ("name", "label", "color")]
    contender_mask = df["name"] != "lr_baseline"

    best_idx = {}
    for col in metric_cols:
        vals = df.loc[contender_mask, col].dropna()
        if vals.empty:
            best_idx[col] = None
            continue
        direction = METRIC_DIRECTIONS[col]
        if direction == "max":
            best_idx[col] = vals.idxmax()
        elif direction == "min":
            best_idx[col] = vals.idxmin()
        else:
            best_idx[col] = vals.abs().idxmin()

    cell_text, cell_colors = [], []
    for i, row in df.iterrows():
        line = [row["label"]]
        colors_row = [to_rgba(row["color"], alpha=0.30)]
        for col in metric_cols:
            v = row[col]
            if pd.isna(v):
                txt = "-"
            elif col == "PBIAS (%)":
                txt = f"{v:+.1f}%"
            elif col in ("Corr (r)", "KGE", "Mean FSS", "Mean CSI"):
                txt = f"{v:.3f}"
            else:
                txt = f"{v:+.2f}" if col == "Bias (mm/day)" else f"{v:.2f}"
            if best_idx.get(col) == i:
                txt = txt + "  \u2605"  # star marker on the best value in this column
            line.append(txt)
            colors_row.append("#eaf7ea" if best_idx.get(col) == i else "white")
        cell_text.append(line)
        cell_colors.append(colors_row)

    col_labels = ["Model"] + metric_cols
    fig_w = 2.8 + 1.55 * len(metric_cols)
    fig_h = 0.85 * len(df) + 1.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=cell_colors,
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 2.0)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    for (r_, c_), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if r_ == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif c_ == 0:
            cell.set_text_props(fontweight="bold", ha="left")
            cell.PAD = 0.03

    direction_note = ("Bias/PBIAS: closer to 0 is better  |  RMSE: lower is better  |  "
                       "Corr/KGE/FSS/CSI: higher is better\n"
                       "\u2605 = best among the downscaling models (baseline row shown for context, not ranked)")
    ax.set_title("Model Comparison - All Years Combined", fontsize=15, fontweight="bold", pad=22)
    ax.text(0.5, -0.02, direction_note, transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#444444")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    loaded = load_full_record()
    if loaded is None:
        print("No data loaded - check GT_FILES/model paths.")
        return
    gt_all, lr_all, model_aligned_all = loaded

    rows = build_rows(gt_all, lr_all, model_aligned_all)
    df = pd.DataFrame(rows)

    csv_path = os.path.join(OUT_DIR, "model_comparison_table.csv")
    df.drop(columns=["color"]).to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(df.drop(columns=["color"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    render_table_image(df, os.path.join(OUT_DIR, "model_comparison_table.png"))


if __name__ == "__main__":
    main()