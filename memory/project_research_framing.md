---
name: Research framing and goals
description: MarketMind's core research question, novel contribution, and what the project is NOT about
type: project
---

**Core research question**: "Where and when are prediction markets miscalibrated, and can ML models exploit those systematic errors?"

**Novel contribution**: No published work trains ML models to detect and exploit systematic Polymarket miscalibration patterns. Le (2026) shows calibration varies by domain but nobody built ML to exploit it.

**What this is NOT**: This is NOT a "can ML beat the market?" study. The market price dominates raw outcome prediction. The goal is to characterize miscalibration and correct it.

**Why:** User explicitly corrected the framing — the project is research that utilizes ML models, not ML as secondary benchmarks. Generic "are prediction markets accurate?" is redundant.

**How to apply:** All modeling should target market errors (where the market is wrong), not raw outcomes. Evaluation should use Brier decomposition and be sliced by domain/horizon. Features should capture structural signals the crowd misses.
