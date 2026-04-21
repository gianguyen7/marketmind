"""B5: Walk-forward validation of the structural hypothesis.

Hypothesis (from docs/shutdown_hypothesis_2026-04-11.md):
    "Long-horizon low-liquidity binary markets under-price YES because the
    marginal trader is capital-constrained and attention-limited."

This script tests the hypothesis on the expanded 4,538-market Gamma dataset
(B3 splits), NOT the 10-market shutdown sample where A2 was underpowered.

Protocol:
    Phase 1 — Signal search on train+val (test is untouched).
    Phase 2 — If signals confirm, freeze a rule and evaluate on test.

Pre-registered primary signals (carried from shutdown_hypothesis doc):
    1. days_to_end  vs  market_error      (+ on YES-resolved markets)
    2. pct_lifetime_elapsed  vs  market_error  (- on YES-resolved markets)
    3. log_volume   vs  |market_error|    (-)

Cross-sectional secondary:
    4. event_group_price_deviation  vs  |market_error|  (+)

Structural population:
    Bottom 25% by volume_usd AND max(days_to_end) > 90 days per market.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


DATE_TAG = datetime.now().strftime("%Y-%m-%d")
FIG_DIR = Path("outputs/figures")
TABLE_DIR = Path("outputs/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    return train, val, test


def structural_population(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to the structural population: bottom-Q25 volume AND long-horizon."""
    market_stats = df.groupby("condition_id").agg(
        volume_usd=("volume_usd", "first"),
        max_days_to_end=("days_to_end", "max"),
    )
    q25_vol = market_stats["volume_usd"].quantile(0.25)
    struct_ids = market_stats[
        (market_stats["volume_usd"] <= q25_vol)
        & (market_stats["max_days_to_end"] > 90)
    ].index
    out = df[df["condition_id"].isin(struct_ids)].copy()
    n_markets = out["condition_id"].nunique()
    print(f"  Structural population: {n_markets} markets, {len(out)} snapshots "
          f"(Q25 volume <= ${q25_vol:,.0f}, max days_to_end > 90)")
    return out


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Spearman rho with p-value, dropping NaN pairs."""
    mask = x.notna() & y.notna()
    if mask.sum() < 5:
        return np.nan, np.nan
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def bootstrap_mean_ci(
    x: pd.Series, n_boot: int = 5000, ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean."""
    rng = np.random.RandomState(42)
    means = [x.sample(len(x), replace=True, random_state=rng.randint(0, 2**31)).mean()
             for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(x.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Phase 1: Signal search (train + val only)
# ---------------------------------------------------------------------------

def phase1_signal_search(
    train: pd.DataFrame, val: pd.DataFrame,
) -> dict:
    """Test primary and cross-sectional signals. Returns results dict."""
    results = {"primary_signals": {}, "cross_sectional": {}, "verdict": None}

    # --- Directional bias check (all markets, not just struct pop) ---
    print("\n" + "=" * 60)
    print("Phase 1: Signal search (train + val, test untouched)")
    print("=" * 60)

    for name, df in [("train", train), ("val", val)]:
        n_m = df["condition_id"].nunique()
        me_mean, me_lo, me_hi = bootstrap_mean_ci(df["market_error"])
        yes_rate = df.groupby("condition_id")["resolved_yes"].first().mean()
        print(f"\n  [{name}] {n_m} markets, {len(df)} snapshots")
        print(f"    Snapshot-level market_error: {me_mean:+.4f} "
              f"(95% CI [{me_lo:+.4f}, {me_hi:+.4f}])")
        print(f"    YES resolution rate: {yes_rate:.3f}")

    # --- Primary signals on structural population ---
    print(f"\n{'─'*60}")
    print("Primary signals on structural population")
    print(f"{'─'*60}")

    train_struct = structural_population(train)
    val_struct = structural_population(val)

    # Also look at full population for comparison
    signals = [
        ("days_to_end", "market_error", "+", "YES-resolved only"),
        ("pct_lifetime_elapsed", "market_error", "-", "YES-resolved only"),
        ("log_volume", "abs_market_error", "-", "all markets"),
    ]

    all_pass = True
    for feat, target, predicted_sign, subset_desc in signals:
        print(f"\n  Signal: {feat} vs {target} (predicted {predicted_sign}, {subset_desc})")

        for sname, sdf in [("train", train_struct), ("val", val_struct)]:
            if subset_desc == "YES-resolved only":
                sub = sdf[sdf["resolved_yes"] == 1]
            else:
                sub = sdf

            if feat not in sub.columns or target not in sub.columns:
                print(f"    [{sname}] SKIP: missing column")
                continue

            rho, p = spearman(sub[feat], sub[target])
            n_markets = sub["condition_id"].nunique()
            n_snap = len(sub)

            sign_match = (predicted_sign == "+" and rho > 0) or \
                         (predicted_sign == "-" and rho < 0)
            status = "OK" if sign_match else "FLIP"

            print(f"    [{sname}] ρ = {rho:+.3f} (p={p:.4f}), "
                  f"n={n_snap} snaps / {n_markets} markets — {status}")

            results["primary_signals"].setdefault(feat, {})[sname] = {
                "rho": rho, "p": p, "n_snaps": n_snap,
                "n_markets": n_markets, "sign_match": sign_match,
            }

            if not sign_match:
                all_pass = False

    # --- Cross-sectional: event_group_price_deviation ---
    print(f"\n{'─'*60}")
    print("Cross-sectional signal: event_group_price_deviation")
    print(f"{'─'*60}")

    # Compute event_group_price_deviation on train+val pooled
    for sname, sdf in [("train", train_struct), ("val", val_struct)]:
        # Mean price_yes per event group at each snapshot time
        if "event_group" in sdf.columns and "price_yes" in sdf.columns:
            group_mean = sdf.groupby("event_group")["price_yes"].transform("mean")
            sdf = sdf.copy()
            sdf["eg_price_dev"] = (sdf["price_yes"] - group_mean).abs()

            # Only multi-market groups
            group_sizes = sdf.groupby("event_group")["condition_id"].transform("nunique")
            multi = sdf[group_sizes > 1]

            if len(multi) > 10:
                rho, p = spearman(multi["eg_price_dev"], multi["abs_market_error"])
                n_m = multi["condition_id"].nunique()
                print(f"  [{sname}] |eg_price_dev| vs |market_error|: "
                      f"ρ = {rho:+.3f} (p={p:.4f}), "
                      f"n={len(multi)} snaps / {n_m} markets")
                results["cross_sectional"][sname] = {"rho": rho, "p": p, "n_markets": n_m}
            else:
                print(f"  [{sname}] Too few multi-market group snapshots ({len(multi)})")

    # --- Also test on ALL markets (not just structural pop) for context ---
    print(f"\n{'─'*60}")
    print("Context: same signals on FULL population (not just structural)")
    print(f"{'─'*60}")

    for feat, target, predicted_sign, subset_desc in signals:
        print(f"\n  {feat} vs {target} ({subset_desc}):")
        for sname, sdf in [("train", train), ("val", val)]:
            if subset_desc == "YES-resolved only":
                sub = sdf[sdf["resolved_yes"] == 1]
            else:
                sub = sdf
            rho, p = spearman(sub[feat], sub[target])
            n_m = sub["condition_id"].nunique()
            sign_match = (predicted_sign == "+" and rho > 0) or \
                         (predicted_sign == "-" and rho < 0)
            status = "OK" if sign_match else "FLIP"
            print(f"    [{sname}] ρ = {rho:+.3f} (p={p:.4f}), {n_m} markets — {status}")

    # --- Verdict ---
    print(f"\n{'='*60}")
    if all_pass:
        print("Phase 1 VERDICT: All primary signals match predicted direction.")
        print("Proceed to Phase 2 (frozen rule evaluation on test).")
        results["verdict"] = "CONFIRMED"
    else:
        flipped = []
        for feat, info in results["primary_signals"].items():
            for split, data in info.items():
                if not data["sign_match"]:
                    flipped.append(f"{feat}/{split}")
        print(f"Phase 1 VERDICT: Sign flip on {', '.join(flipped)}.")
        print("Mechanism NOT confirmed on expanded data.")
        results["verdict"] = "REJECTED"
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Phase 2: Frozen rule evaluation on test
# ---------------------------------------------------------------------------

def phase2_frozen_rule(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    phase1_results: dict,
) -> dict:
    """Freeze a rule from train+val, evaluate on test."""

    print("\n" + "=" * 60)
    print("Phase 2: Frozen rule evaluation on test split")
    print("=" * 60)

    if phase1_results["verdict"] != "CONFIRMED":
        print("\n  Phase 1 did not confirm. Running Phase 2 anyway for completeness")
        print("  but results are EXPLORATORY, not confirmatory.")

    # Define the rule from train+val structural population
    # Rule: on structural-population markets, predict YES with adjusted probability
    #   adjusted_prob = max(price_yes, price_yes + alpha)
    # where alpha is the mean market_error on train structural pop YES-resolved markets
    train_struct = structural_population(train)
    val_struct = structural_population(val)
    test_struct = structural_population(test)

    # Estimate alpha from train YES-resolved structural pop
    train_yes = train_struct[train_struct["resolved_yes"] == 1]
    alpha_train = train_yes["market_error"].mean()

    # Validate on val
    val_yes = val_struct[val_struct["resolved_yes"] == 1]
    alpha_val = val_yes["market_error"].mean()

    print(f"\n  Train struct pop YES markets: {train_yes['condition_id'].nunique()}")
    print(f"  Train alpha (mean market_error on YES): {alpha_train:+.4f}")
    print(f"  Val struct pop YES markets: {val_yes['condition_id'].nunique()}")
    print(f"  Val alpha (mean market_error on YES): {alpha_val:+.4f}")

    # Freeze rule: use train alpha
    FROZEN_ALPHA = alpha_train
    print(f"\n  FROZEN RULE: For structural-pop markets,")
    print(f"    predicted_prob = clip(price_yes + {FROZEN_ALPHA:.4f}, 0, 1)")
    print(f"    For non-structural-pop markets, predicted_prob = price_yes (no adjustment)")

    # Evaluate on all three splits
    results = {}
    for sname, sdf in [("train", train), ("val", val), ("test", test)]:
        sdf = sdf.copy()
        struct_ids = structural_population(sdf)["condition_id"].unique()

        # Naive prediction: price_yes
        sdf["pred_naive"] = sdf["price_yes"]

        # Rule prediction: adjust structural pop
        sdf["pred_rule"] = sdf["price_yes"]
        mask = sdf["condition_id"].isin(struct_ids)
        sdf.loc[mask, "pred_rule"] = (
            sdf.loc[mask, "price_yes"] + FROZEN_ALPHA
        ).clip(0, 1)

        # Brier scores
        brier_naive = ((sdf["pred_naive"] - sdf["resolved_yes"]) ** 2).mean()
        brier_rule = ((sdf["pred_rule"] - sdf["resolved_yes"]) ** 2).mean()

        # Brier on structural pop only
        struct_df = sdf[mask]
        brier_naive_struct = ((struct_df["pred_naive"] - struct_df["resolved_yes"]) ** 2).mean()
        brier_rule_struct = ((struct_df["pred_rule"] - struct_df["resolved_yes"]) ** 2).mean()

        # Market-level Brier (one score per market, then average)
        market_brier_naive = sdf.groupby("condition_id").apply(
            lambda g: ((g["pred_naive"] - g["resolved_yes"]) ** 2).mean()
        )
        market_brier_rule = sdf.groupby("condition_id").apply(
            lambda g: ((g["pred_rule"] - g["resolved_yes"]) ** 2).mean()
        )
        market_brier_naive_mean = market_brier_naive.mean()
        market_brier_rule_mean = market_brier_rule.mean()

        # Structural pop market-level
        struct_market_brier_naive = market_brier_naive.loc[
            market_brier_naive.index.isin(struct_ids)
        ].mean()
        struct_market_brier_rule = market_brier_rule.loc[
            market_brier_rule.index.isin(struct_ids)
        ].mean()

        n_struct = sdf[mask]["condition_id"].nunique()
        n_total = sdf["condition_id"].nunique()

        print(f"\n  [{sname}] {n_total} markets ({n_struct} in structural pop)")
        print(f"    Snapshot Brier (all):    naive={brier_naive:.4f}  rule={brier_rule:.4f}  "
              f"Δ={brier_rule - brier_naive:+.4f}")
        print(f"    Snapshot Brier (struct): naive={brier_naive_struct:.4f}  "
              f"rule={brier_rule_struct:.4f}  Δ={brier_rule_struct - brier_naive_struct:+.4f}")
        print(f"    Market Brier (all):      naive={market_brier_naive_mean:.4f}  "
              f"rule={market_brier_rule_mean:.4f}  "
              f"Δ={market_brier_rule_mean - market_brier_naive_mean:+.4f}")
        print(f"    Market Brier (struct):   naive={struct_market_brier_naive:.4f}  "
              f"rule={struct_market_brier_rule:.4f}  "
              f"Δ={struct_market_brier_rule - struct_market_brier_naive:+.4f}")

        results[sname] = {
            "n_markets": n_total,
            "n_struct": n_struct,
            "brier_naive": float(brier_naive),
            "brier_rule": float(brier_rule),
            "brier_naive_struct": float(brier_naive_struct),
            "brier_rule_struct": float(brier_rule_struct),
            "market_brier_naive": float(market_brier_naive_mean),
            "market_brier_rule": float(market_brier_rule_mean),
            "struct_market_brier_naive": float(struct_market_brier_naive),
            "struct_market_brier_rule": float(struct_market_brier_rule),
        }

    # Simulated P&L: bet $100 on YES for each structural-pop snapshot
    # where rule says prob > price (i.e., always for struct pop since alpha > 0)
    print(f"\n{'─'*60}")
    print("Simulated P&L: $100 bet on YES for each structural-pop market")
    print(f"{'─'*60}")

    for sname, sdf in [("train", train), ("val", val), ("test", test)]:
        struct_ids_set = set(structural_population(sdf)["condition_id"].unique())
        # One bet per market at first available snapshot
        first_snaps = sdf[sdf["condition_id"].isin(struct_ids_set)].groupby(
            "condition_id"
        ).first().reset_index()

        # P&L: bet $100 at price_yes, win $100 if resolved YES, lose $100 * price_yes if NO
        # Expected value per bet = resolved_yes * 100 - price_yes * 100
        # Actually: buy YES at price_yes. Payoff = 100 if YES, 0 if NO. Cost = price_yes * 100.
        first_snaps["pnl"] = first_snaps["resolved_yes"] * 100 - first_snaps["price_yes"] * 100
        total_pnl = first_snaps["pnl"].sum()
        mean_pnl = first_snaps["pnl"].mean()
        n_bets = len(first_snaps)
        n_wins = (first_snaps["resolved_yes"] == 1).sum()
        win_rate = n_wins / n_bets if n_bets > 0 else 0
        avg_price = first_snaps["price_yes"].mean()

        print(f"  [{sname}] {n_bets} bets, win rate {win_rate:.1%}, "
              f"avg entry price {avg_price:.3f}")
        print(f"    Total P&L: ${total_pnl:+,.0f}, Mean P&L/bet: ${mean_pnl:+.2f}")

        results[sname]["pnl_total"] = float(total_pnl)
        results[sname]["pnl_mean"] = float(mean_pnl)
        results[sname]["n_bets"] = n_bets
        results[sname]["win_rate"] = float(win_rate)

    return {"frozen_alpha": float(FROZEN_ALPHA), "splits": results}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_signal_scatter(
    train: pd.DataFrame, val: pd.DataFrame,
) -> None:
    """Scatter plots of primary signals on structural pop."""
    train_struct = structural_population(train)
    val_struct = structural_population(val)

    signals = [
        ("days_to_end", "market_error", "YES-resolved", True),
        ("pct_lifetime_elapsed", "market_error", "YES-resolved", True),
        ("log_volume", "abs_market_error", "All markets", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for col, (feat, target, label, yes_only) in enumerate(signals):
        for row, (sname, sdf) in enumerate([("train", train_struct), ("val", val_struct)]):
            ax = axes[row, col]
            sub = sdf[sdf["resolved_yes"] == 1] if yes_only else sdf

            if len(sub) == 0:
                ax.set_title(f"{sname}: no data")
                continue

            ax.scatter(sub[feat], sub[target], alpha=0.05, s=5, c="steelblue")

            # Rolling mean
            sorted_sub = sub.sort_values(feat)
            rolling = sorted_sub[target].rolling(
                max(50, len(sorted_sub) // 20), min_periods=10, center=True
            ).mean()
            ax.plot(sorted_sub[feat], rolling, c="red", lw=2, label="rolling mean")

            rho, p = spearman(sub[feat], sub[target])
            ax.set_title(f"{sname}: ρ={rho:+.3f} (p={p:.3f})")
            ax.set_xlabel(feat)
            ax.set_ylabel(target)
            if row == 0:
                ax.set_title(f"{feat}\n{sname}: ρ={rho:+.3f} ({label})")
            ax.legend(fontsize=8)

    fig.suptitle(f"B5 Primary Signals — Structural Population ({DATE_TAG})", fontsize=14)
    fig.tight_layout()
    path = FIG_DIR / f"b5_primary_signals_{DATE_TAG}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_rule_evaluation(phase2_results: dict) -> None:
    """Bar chart comparing naive vs rule Brier across splits."""
    splits = phase2_results["splits"]
    names = ["train", "val", "test"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Snapshot-level Brier on structural pop
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.35
    naive_vals = [splits[n]["brier_naive_struct"] for n in names]
    rule_vals = [splits[n]["brier_rule_struct"] for n in names]
    ax.bar(x - w/2, naive_vals, w, label="Naive (price_yes)", color="lightcoral")
    ax.bar(x + w/2, rule_vals, w, label="Rule (adjusted)", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Brier Score (lower = better)")
    ax.set_title("Structural Pop — Snapshot Brier")
    ax.legend()
    for i, (nv, rv) in enumerate(zip(naive_vals, rule_vals)):
        delta = rv - nv
        ax.annotate(f"Δ={delta:+.4f}", (i, max(nv, rv) + 0.002),
                    ha="center", fontsize=9)

    # P&L
    ax = axes[1]
    pnl_vals = [splits[n]["pnl_total"] for n in names]
    colors = ["green" if v > 0 else "red" for v in pnl_vals]
    ax.bar(names, pnl_vals, color=colors, alpha=0.7)
    ax.set_ylabel("Total P&L ($)")
    ax.set_title("Simulated P&L — $100/bet on struct pop YES")
    ax.axhline(0, color="black", lw=0.5)
    for i, v in enumerate(pnl_vals):
        n_bets = splits[names[i]]["n_bets"]
        ax.annotate(f"${v:+,.0f}\n({n_bets} bets)", (i, v),
                    ha="center", va="bottom" if v > 0 else "top", fontsize=9)

    fig.suptitle(f"B5 Rule Evaluation — Frozen α={phase2_results['frozen_alpha']:+.4f} "
                 f"({DATE_TAG})", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"b5_rule_evaluation_{DATE_TAG}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Category breakdown
# ---------------------------------------------------------------------------

def category_breakdown(
    test: pd.DataFrame, frozen_alpha: float,
) -> pd.DataFrame:
    """Per-category Brier and P&L on test structural pop."""
    print(f"\n{'='*60}")
    print("Per-category breakdown on TEST structural population")
    print(f"{'='*60}")

    test_struct = structural_population(test)
    rows = []

    for cat in sorted(test_struct["category"].unique()):
        cat_df = test_struct[test_struct["category"] == cat]
        n_m = cat_df["condition_id"].nunique()
        if n_m < 2:
            continue

        brier_naive = ((cat_df["price_yes"] - cat_df["resolved_yes"]) ** 2).mean()
        pred_rule = (cat_df["price_yes"] + frozen_alpha).clip(0, 1)
        brier_rule = ((pred_rule - cat_df["resolved_yes"]) ** 2).mean()

        # P&L per market (first snapshot)
        first = cat_df.groupby("condition_id").first().reset_index()
        first["pnl"] = first["resolved_yes"] * 100 - first["price_yes"] * 100
        yes_rate = first["resolved_yes"].mean()

        rows.append({
            "category": cat,
            "n_markets": n_m,
            "yes_rate": yes_rate,
            "brier_naive": brier_naive,
            "brier_rule": brier_rule,
            "brier_delta": brier_rule - brier_naive,
            "pnl_total": first["pnl"].sum(),
            "pnl_per_bet": first["pnl"].mean(),
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format="%.4f"))
    path = TABLE_DIR / f"b5_category_breakdown_{DATE_TAG}.csv"
    result.to_csv(path, index=False)
    print(f"\n  Saved {path}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("B5: Walk-Forward Validation of Structural Hypothesis")
    print("=" * 60)
    print(f"\nHypothesis: Long-horizon low-liquidity binary markets under-price YES.")
    print(f"Dataset: B3 expanded splits (4,538 markets, 11 categories).")
    print(f"Date: {DATE_TAG}")

    # Caveats (printed before results, per research plan)
    print(f"\n{'─'*60}")
    print("PRE-COMMITTED CAVEATS (same-sentence rule applies to all numbers below):")
    print("  1. Structural pop is defined empirically (bottom-Q25 volume, days_to_end>90)")
    print("  2. Effective n = market count, not snapshot count")
    print("  3. Test split is temporal (most recent markets), not true walk-forward")
    print("     (no post-freeze live data). This is the best available proxy.")
    print("  4. Category-stratified split means train/test temporal gaps vary by category")
    print("  5. Alpha is estimated from train only; val is for signal confirmation")
    print("  6. No multiple-comparison correction on 3 primary signals")
    print(f"{'─'*60}")

    train, val, test = load_splits()

    # Phase 1: Signal search
    phase1 = phase1_signal_search(train, val)

    # Figures
    print(f"\n{'─'*60}")
    print("Generating figures...")
    plot_signal_scatter(train, val)

    # Phase 2: Frozen rule (run regardless for completeness)
    phase2 = phase2_frozen_rule(train, val, test, phase1)

    plot_rule_evaluation(phase2)
    cat_breakdown = category_breakdown(test, phase2["frozen_alpha"])

    # Save combined results
    all_results = {
        "date": DATE_TAG,
        "hypothesis": "Long-horizon low-liquidity binary markets under-price YES",
        "phase1": phase1,
        "phase2": phase2,
    }
    results_path = TABLE_DIR / f"b5_results_{DATE_TAG}.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n  Saved {results_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("B5 SUMMARY")
    print(f"{'='*60}")
    print(f"  Phase 1 verdict: {phase1['verdict']}")
    test_res = phase2["splits"]["test"]
    print(f"  Frozen alpha: {phase2['frozen_alpha']:+.4f}")
    print(f"  Test Brier (struct pop): naive={test_res['brier_naive_struct']:.4f} "
          f"→ rule={test_res['brier_rule_struct']:.4f} "
          f"(Δ={test_res['brier_rule_struct'] - test_res['brier_naive_struct']:+.4f})")
    print(f"  Test P&L: ${test_res['pnl_total']:+,.0f} over {test_res['n_bets']} bets "
          f"(${test_res['pnl_mean']:+.2f}/bet, win rate {test_res['win_rate']:.1%})")

    if phase1["verdict"] == "CONFIRMED" and \
       test_res["brier_rule_struct"] < test_res["brier_naive_struct"]:
        print("\n  RESULT: Structural hypothesis SUPPORTED on expanded data.")
        print("  The frozen rule improves Brier on the test structural population.")
    elif phase1["verdict"] == "CONFIRMED":
        print("\n  RESULT: Signals confirmed but rule does NOT improve test Brier.")
        print("  The mechanism may be real but the simple alpha-shift rule is insufficient.")
    else:
        print(f"\n  RESULT: Structural hypothesis NOT CONFIRMED on expanded data.")
        print("  The mechanism does not generalize beyond the shutdown theme.")


if __name__ == "__main__":
    main()
