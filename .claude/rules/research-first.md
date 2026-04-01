---
name: research-first
description: Prioritize research correctness and scientific rigor over engineering polish
---

# Research First

When making implementation decisions, optimize for:

1. **Correctness of results** over code elegance
2. **Interpretability** over performance — prefer a model you can explain
3. **Statistical validity** over more features — one well-validated experiment beats five sloppy ones
4. **Verifiable claims** — every number in a results table must trace back to code + data

When in doubt between "ship faster" and "validate more carefully," choose validation. A wrong result published is worse than a correct result delayed.

This does NOT mean over-engineer. Keep code simple, but make sure the research logic is sound.
