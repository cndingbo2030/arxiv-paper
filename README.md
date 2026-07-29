# Posterior Stability and Lifted Discretization for Bayesian Inverse Problems with Learned Ornstein--Uhlenbeck Drift Priors

Companion repository for the manuscript by Bo Ding.

[SSRN abstract 7040660](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7040660)

The repository contains a revised manuscript. The SSRN record may continue to
show an earlier title or PDF until the revised file is uploaded and accepted by
SSRN.

## Scope

The paper studies a prior defined as the terminal law of an
Ornstein--Uhlenbeck process with a learned Cameron--Martin-valued drift. It
establishes:

- local Hellinger stability of the posterior in the observed data;
- a full-space lifted discretization bound controlled by pathwise drift error
  and the covariance tail trace;
- a conditional posterior-contraction criterion that keeps prior mass, tests,
  and identifiability as explicit assumptions; and
- an exact characterization of observational equivalence for bounded linear
  observations.

The manuscript does not claim that an arbitrary trained network is the
reverse-time score of the defining SDE. It also does not claim a universal
posterior-contraction exponent, a nonlinear-forward-map theorem, or sampler
convergence.

## Repository layout

```text
paper/                  LaTeX manuscript and appendices
experiments/            Exploratory one-dimensional code and tests
scripts/                Paper build and packaging scripts
.github/workflows/      LaTeX and Python continuous integration
```

## Build the manuscript

```bash
make paper
make arxiv
```

The first command builds `paper/build/main.pdf`. The second creates a minimal
`arxiv_submission.tar.gz` containing only the manuscript source and
bibliography needed by arXiv.

## Run the code checks

```bash
python -m venv .venv
source .venv/bin/activate
make install
make test
make lint
```

The numerical routines are exploratory implementation checks. They are not
used as evidence for the mathematical theorems.

## Citation

```bibtex
@article{ding2026posterior,
  title   = {Posterior Stability and Lifted Discretization for Bayesian Inverse Problems with Learned Ornstein--Uhlenbeck Drift Priors},
  author  = {Ding, Bo},
  journal = {SSRN},
  year    = {2026},
  url     = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7040660}
}
```
