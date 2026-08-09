# Code origin

LeanHEBO reimplements the main single-objective exact-GP path of Huawei's HEBO package,
audited at commit `ee6112d39d1a9e9703fecaf9057193e1ec9dae72`.

## Reimplemented components

- mixed-variable design-space encoding and Sobol sampling;
- output standardization and Box-Cox/Yeo-Johnson transformation behavior;
- exact Gaussian-process regression and HEBO-compatible kernel construction;
- the three-objective MACE acquisition calculation;
- HEBO-style batch candidate selection.

## Independently implemented components

- immutable compiled-space metadata and `CandidateBatch` boundary;
- append-only tensor observation storage and incremental duplicate membership;
- persistent GP/model/optimizer lifecycle;
- tensor-native mixed-variable NSGA-II;
- configuration, diagnostics, checkpointing, and optional table adapters.

Files that substantially copy upstream implementation text must retain Huawei copyright
headers. New implementations use SPDX identifiers and refer to `NOTICE.md`.
