This repository contains the code to reproduce all the findings in "Bayesian Model Averaging in Causal Instrumental Variable Models", [arxiv:2504.13520](https://arxiv.org/abs/2504.13520).

The applications folder includes all the code to reproduce the empirical examples. In particular, `Carstensen_Gundlach.jl`, `schooling.jl`, and `birthweight.jl` contain the code for the geography or institutions, returns to schooling, and birthweight examples, respectively.
The priors folder includes code to reproduce the plots illustrating the priors implied by the hyperprior on $\nu$.
The simulations folder includes all simulation experiments included in the paper:
  - `simulation_KO2010.jl` includes the code for the setting with many weak instruments, and `simulation_pln.jl` includes the code for the design with a Poisson endogenous variable.
  - `simulation_Kang2016.jl` includes the code for the setting with some invalid instruments.
  - `simulation_MultEnd.jl` includes the code for the setting with multiple endogenous variables and correlated instruments.

The `gIVBMA.jl` package (https://github.com/gregorsteiner/gIVBMA.jl) is required to run the experiments and applications.

