# Changelog

All notable changes to LeanHEBO will be documented here.

## 0.1.0 - 2026-08-09

- Establish the independent `uv` project and provenance boundary.
- Implement the initial exact-GP LeanHEBO optimization path.
- Fit and checkpoint the HEBO-compatible observed-range input scaler used by the exact GP.
- Preserve append-only observation state, reserved duplicate keys, diagnostics, search history,
  and exact version counters across safe checkpoints.
- Raise a typed exhaustion error instead of silently returning duplicate suggestions.
- Retry numerical fit or posterior failures once with a clean full GP refit.
- Add a pinned, dependency-locked upstream benchmark environment and a fail-closed paired
  comparison harness with actual-work checks and bootstrap confidence intervals.
