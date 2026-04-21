"""Track A — Government shutdown mispricing drill-down.

A1 (characterization) — produces:
    1. Price trajectories per market (PNG, grouped by resolved_yes).
    2. Per-market error decomposition table (printed + CSV).
    3. Directional bias: mean market_error with bootstrap 95% CI,
       across snapshots and across markets.
    4. Outlier check: per-market contribution to the 0.62 mean |market_error|.

A2 (signal search, pre-registered) — produces:
    5. Primary-signal tests against `docs/shutdown_hypothesis_2026-04-11.md`:
       days_to_end, pct_lifetime_elapsed, log_volume (Spearman ρ on train/val
       separately, mechanism-predicted directions checked).
    6. Cross-sectional signal test: event_group_price_deviation (val only —
       train has no non-singleton shutdown event groups).
    7. Secondary (descriptive-only) feature correlations, plotted separately
       and explicitly not allowed to drive the confirm/falsify verdict.
    8. Confirm/falsify verdict printed against the pre-registered criteria.

Exploratory script. A1 loads `data/interim/snapshots_enriched.parquet` for
raw characterization; A2 loads `data/processed/{train,val}.parquet` because
it needs the engineered features (log_volume, volume_rank, event_group).
**Test split is held out of A2 entirely** — it is only touched in A3.

Every threshold, filter count, and assumption is printed explicitly
(no-silent-assumptions rule).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data/raw/markets_registry.parquet"
SNAPSHOTS_PATH = ROOT / "data/interim/snapshots_enriched.parquet"
TRAIN_PATH = ROOT / "data/processed/train.parquet"
VAL_PATH = ROOT / "data/processed/val.parquet"
FIG_DIR = ROOT / "outputs/figures"
TABLE_DIR = ROOT / "outputs/tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

TODAY = dt.date.today().isoformat()
THEME = "government_shutdown"
RNG_SEED = 42  # for bootstrap reproducibility
N_BOOTSTRAP = 10_000

# A2 pre-registered thresholds — frozen in docs/shutdown_hypothesis_2026-04-11.md
PRIMARY_RHO_MIN = 0.20   # minimum |Spearman ρ| to count as supported
FALSIFY_RHO_MAX = 0.10   # |ρ| below this on both splits = rejection
PRE_REG_DOC = "docs/shutdown_hypothesis_2026-04-11.md"

# A2 caveat block — printed before and after every A2 result per the pre-reg.
A2_CAVEATS = (
    "A2 CAVEATS (pre-committed, always attached):\n"
    "  - Effective n = 10 markets (2 train / 5 val / 3 test), NOT 717 snapshots.\n"
    "  - Theme was selected on the dependent variable (worst in-sample Brier 0.482).\n"
    "  - Regime window: 2024-2026 shutdown cycles only. Structural claims require\n"
    "    walk-forward validation on post-freeze resolutions before being reported.\n"
    "  - No multiple-comparison correction. ρ values are descriptive, not inferential.\n"
    "  - Structural weakness of the split: train log_volume range is narrow (2 markets,\n"
    "    range ~16.4-17.8); val days_to_end range is short (~1.5-39 days). Train tests\n"
    "    horizon, val tests liquidity and cross-section. They barely overlap.\n"
    "  - Test split is UNTOUCHED by A2. Only A3 may use it."
)


def load_shutdown_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (registry_subset, snapshots_subset) for the shutdown theme."""
    reg = pd.read_parquet(REGISTRY_PATH)
    snap = pd.read_parquet(SNAPSHOTS_PATH)
    print(f"Loaded registry: {len(reg)} markets")
    print(f"Loaded snapshots: {len(snap)} rows")

    sh_reg = reg[reg["theme"] == THEME].copy()
    sh_snap = snap[snap["theme"] == THEME].copy()
    print(f"Filtered to theme='{THEME}': "
          f"{len(sh_reg)} markets, {len(sh_snap)} snapshots")

    # Computed fields — no fill, no drop.
    sh_snap["market_error"] = sh_snap["resolved_yes"].astype(float) - sh_snap["price_yes"]
    sh_snap["abs_market_error"] = sh_snap["market_error"].abs()
    sh_snap = sh_snap.sort_values(["condition_id", "snapshot_ts"]).reset_index(drop=True)

    # Sanity: does overall mean |error| match the expected ~0.62?
    overall_mae = sh_snap["abs_market_error"].mean()
    print(f"Overall mean |market_error| across snapshots: {overall_mae:.4f} "
          f"(expected ~0.62 per plan)")
    return sh_reg, sh_snap


def short_label(question: str, resolved_yes: int, n: int) -> str:
    q = question if len(question) <= 60 else question[:57] + "..."
    return f"[{'YES' if resolved_yes else 'NO '}] {q}  (n={n})"


def plot_price_trajectories(sh_reg: pd.DataFrame, sh_snap: pd.DataFrame) -> Path:
    """One subplot per market, x=snapshot_ts, y=price_yes, color by resolved_yes."""
    markets = (
        sh_snap.groupby("condition_id")
        .agg(n=("snapshot_ts", "size"),
             resolved_yes=("resolved_yes", "max"))
        .reset_index()
        .merge(sh_reg[["condition_id", "question", "end_date"]], on="condition_id")
        .sort_values(["resolved_yes", "end_date"])
        .reset_index(drop=True)
    )
    n_markets = len(markets)
    ncols = 2
    nrows = int(np.ceil(n_markets / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    for i, row in markets.iterrows():
        ax = axes[i]
        mdf = sh_snap[sh_snap["condition_id"] == row["condition_id"]]
        color = "#1b9e77" if row["resolved_yes"] else "#d95f02"
        ax.plot(mdf["snapshot_ts"], mdf["price_yes"], color=color, lw=1.5)
        ax.axhline(float(row["resolved_yes"]), color="black", ls="--", lw=0.8, alpha=0.6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(short_label(row["question"], int(row["resolved_yes"]), int(row["n"])),
                     fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    for j in range(n_markets, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Government shutdown markets — price_yes trajectories\n"
        "Green = resolved YES, Orange = resolved NO, Dashed = true outcome",
        fontsize=12,
    )
    fig.supxlabel("Snapshot timestamp")
    fig.supylabel("price_yes")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = FIG_DIR / f"shutdown_trajectories_{TODAY}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def per_market_error_decomposition(sh_reg: pd.DataFrame, sh_snap: pd.DataFrame) -> pd.DataFrame:
    """For each market: final price, resolution, mean |err|, start vs end |err|."""
    rows = []
    for cid, mdf in sh_snap.groupby("condition_id"):
        mdf = mdf.sort_values("snapshot_ts")
        n = len(mdf)
        # "Start" and "end" windows: first/last 10% of snapshots (min 1)
        k = max(1, int(np.ceil(n * 0.10)))
        start_err = mdf["abs_market_error"].iloc[:k].mean()
        end_err = mdf["abs_market_error"].iloc[-k:].mean()
        final_price = mdf["price_yes"].iloc[-1]
        resolved = int(mdf["resolved_yes"].iloc[0])
        rows.append({
            "condition_id": cid,
            "n_snapshots": n,
            "resolved_yes": resolved,
            "final_price_yes": final_price,
            "mean_abs_error": mdf["abs_market_error"].mean(),
            "mean_signed_error": mdf["market_error"].mean(),
            "abs_error_start_10pct": start_err,
            "abs_error_end_10pct": end_err,
            "abs_error_shrinkage": start_err - end_err,
        })
    df = pd.DataFrame(rows)
    df = df.merge(sh_reg[["condition_id", "question", "end_date", "volume_usd"]],
                  on="condition_id")
    df = df.sort_values("mean_abs_error", ascending=False).reset_index(drop=True)

    print("\n=== A1.2 Per-market error decomposition ===")
    print(f"Note: 'start'/'end' windows = first/last 10% of each market's snapshots "
          f"(min 1 snapshot).")
    with pd.option_context("display.max_colwidth", 60, "display.width", 180):
        print(df[[
            "question", "resolved_yes", "final_price_yes",
            "mean_abs_error", "mean_signed_error",
            "abs_error_start_10pct", "abs_error_end_10pct", "abs_error_shrinkage",
            "n_snapshots",
        ]].to_string(index=False))

    out = TABLE_DIR / f"shutdown_error_decomposition_{TODAY}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")

    # Interpret "shrinkage" in aggregate
    shrunk = (df["abs_error_shrinkage"] > 0.05).sum()
    flat = ((df["abs_error_shrinkage"].abs() <= 0.05)).sum()
    grew = (df["abs_error_shrinkage"] < -0.05).sum()
    print(f"Convergence summary (threshold ±0.05 on |err| shrinkage):")
    print(f"  shrunk toward truth: {shrunk}/{len(df)}")
    print(f"  flat:                {flat}/{len(df)}")
    print(f"  grew away:           {grew}/{len(df)}")
    return df


def bootstrap_ci(values: np.ndarray, stat_fn=np.mean,
                 n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED
                 ) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = stat_fn(values[idx], axis=1)
    return float(stat_fn(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def directional_bias(sh_snap: pd.DataFrame, decomp: pd.DataFrame) -> None:
    """A1.3 — is market_error systematically positive (under-pricing) or negative?"""
    snap_err = sh_snap["market_error"].to_numpy()
    mean_s, lo_s, hi_s = bootstrap_ci(snap_err)

    # Per-market mean signed error
    per_market = decomp["mean_signed_error"].to_numpy()
    mean_m, lo_m, hi_m = bootstrap_ci(per_market)

    print("\n=== A1.3 Directional bias (market_error = resolved - price) ===")
    print(f"Across all 717 snapshots:  mean = {mean_s:+.4f}  "
          f"95% bootstrap CI = [{lo_s:+.4f}, {hi_s:+.4f}]")
    print(f"Across 10 market means:    mean = {mean_m:+.4f}  "
          f"95% bootstrap CI = [{lo_m:+.4f}, {hi_m:+.4f}]")
    print(f"Interpretation: positive = market UNDER-prices YES, "
          f"negative = market OVER-prices YES.")
    print(f"Bootstrap: n_boot={N_BOOTSTRAP}, seed={RNG_SEED}")


def outlier_check(decomp: pd.DataFrame) -> None:
    """A1.4 — is the 0.62 mean driven by a few markets or uniform?"""
    snap_weighted = (decomp["mean_abs_error"] * decomp["n_snapshots"]).sum() / decomp["n_snapshots"].sum()
    market_avg = decomp["mean_abs_error"].mean()
    print("\n=== A1.4 Outlier check ===")
    print(f"Snapshot-weighted mean |err|: {snap_weighted:.4f}")
    print(f"Unweighted per-market mean:   {market_avg:.4f}")
    print(f"Per-market |err| distribution:")
    print(f"  min    = {decomp['mean_abs_error'].min():.4f}")
    print(f"  median = {decomp['mean_abs_error'].median():.4f}")
    print(f"  max    = {decomp['mean_abs_error'].max():.4f}")
    print(f"  std    = {decomp['mean_abs_error'].std():.4f}")

    # Contribution of top-2 markets to total (snapshot-weighted) error mass
    total_err_mass = (decomp["mean_abs_error"] * decomp["n_snapshots"]).sum()
    top2 = decomp.nlargest(2, "mean_abs_error")
    top2_mass = (top2["mean_abs_error"] * top2["n_snapshots"]).sum()
    print(f"Top-2 markets by mean |err| contribute "
          f"{top2_mass / total_err_mass * 100:.1f}% of total (snapshot-weighted) error mass.")

    # Simple outlier figure: bar chart of per-market mean |err|.
    fig, ax = plt.subplots(figsize=(12, 5))
    order = decomp.sort_values("mean_abs_error", ascending=False)
    colors = ["#1b9e77" if r else "#d95f02" for r in order["resolved_yes"]]
    labels = [q if len(q) < 50 else q[:47] + "..." for q in order["question"]]
    ax.bar(range(len(order)), order["mean_abs_error"], color=colors)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("mean |market_error|")
    ax.set_title("Shutdown markets — per-market mean |market_error| "
                 "(green = YES, orange = NO)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"shutdown_per_market_error_{TODAY}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


def load_processed_shutdown() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, val) snapshot subsets for the shutdown theme.

    A2 uses the engineered-feature splits because it needs log_volume,
    volume_rank, and event_group. Test is never loaded here.
    """
    tr = pd.read_parquet(TRAIN_PATH)
    va = pd.read_parquet(VAL_PATH)
    sh_tr = tr[tr["theme"] == THEME].copy()
    sh_va = va[va["theme"] == THEME].copy()
    print(f"A2 train shutdown: {len(sh_tr)} snapshots, "
          f"{sh_tr['condition_id'].nunique()} markets")
    print(f"A2 val   shutdown: {len(sh_va)} snapshots, "
          f"{sh_va['condition_id'].nunique()} markets")
    return sh_tr, sh_va


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Spearman ρ with pairwise-complete deletion. Returns (rho, n_used)."""
    mask = ~(pd.isna(x) | pd.isna(y))
    n = int(mask.sum())
    if n < 5:
        return float("nan"), n
    return float(pd.Series(x[mask]).corr(pd.Series(y[mask]), method="spearman")), n


def plot_signal(ax, x, y, title, xlabel, ylabel) -> None:
    ax.scatter(x, y, s=8, alpha=0.35, color="#555")
    # Rolling-mean overlay (ordered by x)
    order = np.argsort(x)
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    if len(xs) >= 20:
        window = max(10, len(xs) // 15)
        rolled = pd.Series(ys).rolling(window, center=True, min_periods=5).mean()
        ax.plot(xs, rolled, color="#d95f02", lw=1.8, label=f"rolling mean (w={window})")
        ax.legend(loc="best", fontsize=8)
    ax.axhline(0, color="black", ls="--", lw=0.6, alpha=0.6)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3)


def a2_primary_signals(sh_tr: pd.DataFrame, sh_va: pd.DataFrame) -> dict:
    """Test the three pre-registered primary signals.

    Predictions (from docs/shutdown_hypothesis_2026-04-11.md §2):
        days_to_end          -> positive Spearman with market_error
        pct_lifetime_elapsed -> negative Spearman with market_error
        log_volume           -> negative Spearman with abs_market_error
    """
    print("\n=== A2.1 Primary signals (pre-registered) ===")
    print(A2_CAVEATS)

    tests = [
        ("days_to_end", "market_error", "+", "longer horizon → YES under-pricing"),
        ("pct_lifetime_elapsed", "market_error", "-",
         "early in lifetime → YES under-pricing"),
        ("log_volume", "abs_market_error", "-",
         "low liquidity → larger |error|"),
    ]

    results: dict[str, dict] = {}
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for i, (feat, target, expected, mechanism_label) in enumerate(tests):
        rho_tr, n_tr = spearman(sh_tr[feat].to_numpy(), sh_tr[target].to_numpy())
        rho_va, n_va = spearman(sh_va[feat].to_numpy(), sh_va[target].to_numpy())
        pred_sign = +1 if expected == "+" else -1
        tr_ok = np.sign(rho_tr) == pred_sign and abs(rho_tr) >= PRIMARY_RHO_MIN
        va_ok = np.sign(rho_va) == pred_sign and abs(rho_va) >= PRIMARY_RHO_MIN
        tr_rej = abs(rho_tr) < FALSIFY_RHO_MAX
        va_rej = abs(rho_va) < FALSIFY_RHO_MAX
        sign_flip = (np.sign(rho_tr) != np.sign(rho_va)
                     and abs(rho_tr) > FALSIFY_RHO_MAX
                     and abs(rho_va) > FALSIFY_RHO_MAX)

        results[feat] = dict(
            target=target, expected=expected, mechanism=mechanism_label,
            rho_train=rho_tr, n_train=n_tr,
            rho_val=rho_va, n_val=n_va,
            train_supports=tr_ok, val_supports=va_ok,
            train_reject=tr_rej, val_reject=va_rej,
            sign_flip=sign_flip,
        )
        print(f"\n  [{feat}] vs {target}  (mechanism: {mechanism_label})")
        print(f"    expected sign: {expected}")
        print(f"    train ρ = {rho_tr:+.3f}  (n={n_tr})  "
              f"supports={'YES' if tr_ok else 'no'}  "
              f"reject-zone={'YES' if tr_rej else 'no'}")
        print(f"    val   ρ = {rho_va:+.3f}  (n={n_va})  "
              f"supports={'YES' if va_ok else 'no'}  "
              f"reject-zone={'YES' if va_rej else 'no'}")
        if sign_flip:
            print(f"    *** SIGN FLIP between train and val — falsification trigger ***")

        plot_signal(axes[i, 0], sh_tr[feat].to_numpy(), sh_tr[target].to_numpy(),
                    f"TRAIN  {feat} vs {target}  (ρ={rho_tr:+.3f}, n={n_tr})",
                    feat, target)
        plot_signal(axes[i, 1], sh_va[feat].to_numpy(), sh_va[target].to_numpy(),
                    f"VAL    {feat} vs {target}  (ρ={rho_va:+.3f}, n={n_va})",
                    feat, target)

    fig.suptitle("A2 primary signals — pre-registered, mechanism-predicted directions",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG_DIR / f"shutdown_a2_primary_signals_{TODAY}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")
    return results


def a2_cross_sectional(sh_tr: pd.DataFrame, sh_va: pd.DataFrame) -> dict:
    """Test event_group_price_deviation on val (train has only singleton groups).

    Definition (computed here, not from disk): for each snapshot in a
    non-singleton event_group, round snapshot_ts to the day, compute the
    mean price_yes across all same-group markets with snapshots that day,
    then deviation = price_yes - group_day_mean.
    """
    print("\n=== A2.2 Cross-sectional signal: event_group_price_deviation ===")
    print("Note: train shutdown markets are in singleton event groups "
          "(n_markets=1 per group). Cross-sectional test runs on VAL only.")

    def compute_dev(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["snap_day"] = pd.to_datetime(df["snapshot_ts"]).dt.floor("D")
        grp = df.groupby(["event_group", "snap_day"])
        df["group_day_mean_price"] = grp["price_yes"].transform("mean")
        df["group_day_n_markets"] = grp["condition_id"].transform("nunique")
        df["event_group_price_deviation"] = df["price_yes"] - df["group_day_mean_price"]
        return df

    sh_va_x = compute_dev(sh_va)
    eligible = sh_va_x[sh_va_x["group_day_n_markets"] >= 2].copy()
    n_drop = len(sh_va_x) - len(eligible)
    print(f"Dropped {n_drop} val snapshots whose (event_group, day) had only one "
          f"market reporting (deviation undefined). Kept {len(eligible)} snapshots "
          f"across {eligible['condition_id'].nunique()} markets.")

    rho_dev, n_dev = spearman(
        eligible["event_group_price_deviation"].to_numpy(),
        eligible["market_error"].to_numpy(),
    )
    rho_abs, _ = spearman(
        eligible["event_group_price_deviation"].abs().to_numpy(),
        eligible["abs_market_error"].to_numpy(),
    )

    print(f"ρ(deviation, market_error)          = {rho_dev:+.3f}  (n={n_dev})")
    print(f"  mechanism prediction: NEGATIVE — a market priced above its group peers "
          f"on a given day should have *smaller* (more negative) market_error "
          f"(overpriced relative to truth).")
    print(f"ρ(|deviation|, |market_error|)      = {rho_abs:+.3f}  (n={n_dev})")
    print(f"  mechanism prediction: POSITIVE — deviant-from-group markets should "
          f"have larger absolute errors.")

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_signal(
        ax,
        eligible["event_group_price_deviation"].to_numpy(),
        eligible["market_error"].to_numpy(),
        f"VAL  event_group_price_deviation vs market_error  (ρ={rho_dev:+.3f}, n={n_dev})",
        "event_group_price_deviation",
        "market_error",
    )
    fig.tight_layout()
    out = FIG_DIR / f"shutdown_a2_event_group_deviation_{TODAY}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")

    return dict(rho_signed=rho_dev, rho_abs=rho_abs, n=n_dev)


def a2_secondary_signals(sh_tr: pd.DataFrame, sh_va: pd.DataFrame) -> pd.DataFrame:
    """Descriptive-only correlations for demoted features.

    Per the pre-reg, these must NOT drive the verdict. They are reported so
    we can check whether any of them dominates the primaries (that would be a
    data-mining red flag, not a win).
    """
    print("\n=== A2.3 Secondary (descriptive-only) features ===")
    print("These are DEMOTED per pre-reg §2. They are NOT permitted to drive "
          "the confirm/falsify verdict. Reported so we can check whether any "
          "dominates the primaries (data-mining red flag if so).")

    secondaries = [
        "price_volatility_7", "price_momentum_3", "price_momentum_7",
        "price_change", "price_vs_open", "volume_rank",
    ]
    rows = []
    for feat in secondaries:
        if feat not in sh_tr.columns:
            continue
        rho_tr, n_tr = spearman(sh_tr[feat].to_numpy(), sh_tr["market_error"].to_numpy())
        rho_va, n_va = spearman(sh_va[feat].to_numpy(), sh_va["market_error"].to_numpy())
        rho_tr_abs, _ = spearman(sh_tr[feat].to_numpy(), sh_tr["abs_market_error"].to_numpy())
        rho_va_abs, _ = spearman(sh_va[feat].to_numpy(), sh_va["abs_market_error"].to_numpy())
        rows.append(dict(
            feature=feat,
            rho_train_signed=rho_tr, rho_val_signed=rho_va,
            rho_train_abs=rho_tr_abs, rho_val_abs=rho_va_abs,
            n_train=n_tr, n_val=n_va,
        ))
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 160):
        print(df.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    out = TABLE_DIR / f"shutdown_a2_secondary_correlations_{TODAY}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    return df


def a2_verdict(primary: dict, cross: dict, secondary: pd.DataFrame) -> str:
    """Render the confirm/falsify verdict against pre-registered criteria."""
    print("\n=== A2.4 Verdict vs pre-registered criteria ===")

    # Confirmation: all three primaries support on both train AND val
    primary_feats = ["days_to_end", "pct_lifetime_elapsed", "log_volume"]
    all_primary_supports = all(
        primary[f]["train_supports"] and primary[f]["val_supports"]
        for f in primary_feats
    )

    # Falsification triggers
    any_sign_flip = any(primary[f]["sign_flip"] for f in primary_feats)
    any_both_reject = any(
        primary[f]["train_reject"] and primary[f]["val_reject"]
        for f in primary_feats
    )

    # Secondary-dominates check: max |ρ| from secondaries vs primaries
    prim_max_abs_rho = max(
        max(abs(primary[f]["rho_train"]), abs(primary[f]["rho_val"]))
        for f in primary_feats
    )
    sec_max = 0.0
    sec_dominant_feature = None
    for _, row in secondary.iterrows():
        for col in ["rho_train_signed", "rho_val_signed",
                    "rho_train_abs", "rho_val_abs"]:
            v = abs(row[col])
            if not np.isnan(v) and v > sec_max:
                sec_max = v
                sec_dominant_feature = row["feature"]
    secondary_dominates = sec_max > prim_max_abs_rho + 0.05  # margin

    print(f"  All three primaries support mechanism on both splits? "
          f"{'YES' if all_primary_supports else 'no'}")
    print(f"  Any primary has sign flip train↔val? "
          f"{'YES (falsify)' if any_sign_flip else 'no'}")
    print(f"  Any primary below |ρ|={FALSIFY_RHO_MAX} on both splits? "
          f"{'YES (falsify)' if any_both_reject else 'no'}")
    print(f"  Max |ρ| among primaries: {prim_max_abs_rho:.3f}")
    print(f"  Max |ρ| among secondaries: {sec_max:.3f}  "
          f"(feature: {sec_dominant_feature})")
    print(f"  Secondary dominates primaries by >0.05 margin? "
          f"{'YES (falsify)' if secondary_dominates else 'no'}")

    if any_sign_flip or any_both_reject or secondary_dominates:
        verdict = "REJECTED"
        reason = []
        if any_sign_flip:
            reason.append("primary sign flip between splits")
        if any_both_reject:
            reason.append("primary in reject zone on both splits")
        if secondary_dominates:
            reason.append(f"secondary feature '{sec_dominant_feature}' dominates primaries")
        verdict_line = f"VERDICT: MECHANISM {verdict} — {'; '.join(reason)}"
    elif all_primary_supports:
        verdict = "SUPPORTED"
        verdict_line = ("VERDICT: MECHANISM SUPPORTED — all three primary signals "
                        "pointed in the mechanism-predicted direction with |ρ| ≥ "
                        f"{PRIMARY_RHO_MIN} on both train and val.")
    else:
        verdict = "WEAK"
        verdict_line = ("VERDICT: WEAK — no falsification trigger fired, but not "
                        "all primaries cleared the |ρ| ≥ "
                        f"{PRIMARY_RHO_MIN} bar on both splits. "
                        "Treat as inconclusive; do NOT proceed to A3 without "
                        "rewriting the mechanism.")

    print(f"\n{verdict_line}")
    print(f"\n{A2_CAVEATS}")
    print(f"\nPre-registration doc: {PRE_REG_DOC}")
    return verdict_line


def run_a2(sh_tr: pd.DataFrame, sh_va: pd.DataFrame) -> None:
    primary = a2_primary_signals(sh_tr, sh_va)
    cross = a2_cross_sectional(sh_tr, sh_va)
    secondary = a2_secondary_signals(sh_tr, sh_va)
    a2_verdict(primary, cross, secondary)


def main() -> None:
    print(f"Run date: {TODAY}")
    print(f"Theme filter: {THEME}")
    sh_reg, sh_snap = load_shutdown_data()
    plot_price_trajectories(sh_reg, sh_snap)
    decomp = per_market_error_decomposition(sh_reg, sh_snap)
    directional_bias(sh_snap, decomp)
    outlier_check(decomp)
    print("\nA1 characterization complete.")

    print("\n" + "=" * 72)
    print("A2 — Pre-registered signal search")
    print("=" * 72)
    sh_tr, sh_va = load_processed_shutdown()
    run_a2(sh_tr, sh_va)
    print("\nA2 complete. Append the verdict to docs/results_summary.md.")


if __name__ == "__main__":
    main()
