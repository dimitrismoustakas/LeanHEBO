# Notice

LeanHEBO is an independent implementation derived in part from the HEBO project by
Huawei Technologies Co., Ltd. It is not affiliated with, sponsored by, or endorsed by
Huawei.

- Upstream project: <https://github.com/huawei-noah/HEBO>
- Audited package: `HEBO/`
- Audited commit: `ee6112d39d1a9e9703fecaf9057193e1ec9dae72`
- Upstream license: MIT

The HEBO method is described by Cowen-Rivers et al., “HEBO: Pushing the Limits of
Sample-Efficient Hyperparameter Optimisation,” *Journal of Artificial Intelligence
Research*, volume 74, 2022.

## Code origin

LeanHEBO reimplements the main single-objective exact-GP path from the audited HEBO
commit. The reimplemented components are:

- mixed-variable design-space encoding and Sobol sampling;
- output standardization and Box-Cox/Yeo-Johnson transformation behavior;
- exact Gaussian-process regression and HEBO-compatible kernel construction;
- the three-objective MACE acquisition calculation;
- HEBO-style batch candidate selection.

LeanHEBO independently implements its compiled-space and candidate boundaries,
observation storage, persistent model lifecycle, mixed-variable NSGA-II, configuration,
diagnostics, checkpointing, and optional table adapters.

Files that substantially copy upstream implementation text retain Huawei copyright
headers. New implementations use SPDX identifiers and refer to this notice.
