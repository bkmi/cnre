[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7299458.svg)](https://doi.org/10.5281/zenodo.7299458)

# Contrastive Neural Ratio Estimation
In this repository, we estimate the likelihood-to-evidence ratio from simulated data and parameters pairs in order to determine the posterior distribution. It is a so-called simulation-based inference method... also known as likelihood-free inference or implicit likelihood. The algorithm we propose generalizes [amortized approximate likelihood-ratio estimation](https://arxiv.org/abs/1903.04057) (NRE-A) and [sequential ratio estimation](https://arxiv.org/abs/2002.03712) (NRE-B) into a unified framework that we call [Contrastive Neural Ratio Estimation](https://arxiv.org/abs/2210.06170) (NRE-C). The paper which introduces the method was published at [NeurIPS 2022](https://neurips.cc/virtual/2022/poster/54994) by Benjamin Kurt Miller, Christoph Weniger, and Patrick Forré.

### Installation

We recommend using `micromamba` because conda is extremely slow. You can install `micromamba` by [following their guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#automatic-install). Then create the `cnre` environment:

```bash
micromamba env create -f environment.yml
```

If you want to run the examples from `sbibm` that use `julia`, e.g. `Lotka-Voltera`, then you need to install `julia` version `1.5.3` and run the following commands:

```bash
micromamba activate cnre
export JULIA_SYSIMAGE_DIFFEQTORCH="$HOME/.julia_sysimage_diffeqtorch.so"
python -c "from diffeqtorch.install import install_and_test; install_and_test()"
```

## the folders explained
- `cnre/` - contains several experiments.
- `cnre/cnre` - the package itself.
- `cnre/infinite`- the hydra script folder for all experiments, the benchmarking location.
- `cnre/remote` - packages that our repository depends on.

### `cnre/`
There are two legacy jupyter notebooks that show our initial investigations.
- `00 is it real.ipynb` - A simple test of the performance between existing algorithms NRE-A and NRE-B
- `01 basic algorithm.ipynb` - A simple implementation of our algorithm. We check whether increasing the number of contrastive parameters helps on one posterior.

Then there are a few jupyter notebooks which produced plots that we used in the paper.
- `importance sampling diagnostic.ipynb` - Shows that the importance sampling diagnostic fails for NRE-B, but succeeds for NRE-C. Figure 3.
- `loss-summary-diagram.ipynb` - Creates two plots of a joint and product of marginal distribution for the schematic diagram of the loss function. Figure 2.

### `cnre/cnre`
Contains the algorithm and other tools for creating and evaluating the data.

### `cnre/infinite`
Contains the results from our experiments along with the jupyter notebooks we used to create the plots in paper. Since this section uses [hydra](https://hydra.cc/), we have a `config/` folder. The calls made to produce the data are listed in `calls.md`. The raw data from the runs is in the process of being uploaded to Zenodo.

The python files:
- `compiledf.py` - Takes a folder of results from hydra and summarizes them into a metrics `.csv` file.
- `main.py` - Run the experiment in hydra.
- `notebook.py` - Functions for the notebooks.

Jupyter notebooks and their explanations:
- `00 counts.ipynb` - How many experiments are expected? checking for missing experiments. Were the experiments repeated with different seeds enough times? What was the total compute time?
- `01 validation loss.ipynb` - Plot the validation loss. Plot k=1, gamma=1 validation loss for some experiments, no matter which hyperparameters were used to train them.
- `02 relplots.ipynb` - Plot the C2ST versus contrastive parameter count over many gammas.
- `03 conceptual plot.ipynb` - Create the plot which shows the performance across hyperparameters using a spline fit.
- `04 sbibm.ipynb` - Produce the C2ST table results for the [simulation-based inference benchmark](https://arxiv.org/abs/2101.04653).
- `05 mi.ipynb` - Plot the partition function and mutual information lower bounds. Correlate the results with the C2ST.

These notebooks draw from metric summary `.csv` files. Creating the metrics for the simulation-based inference benchmark setting, i.e., `metrics-sbibm*.csv`, required three attempts. That's why there is more than one file for it.
- `metrics-bench.csv` - Section 3.3 Simulation-based inference benchmark
- `metrics-joint.csv` - Section 3.1 Behavior with unlimited data
- `metrics-mi-long.csv` - (not in paper, but longer version of Section 3 Hyperparameter search)
- `metrics-mi.csv` - Section 3 Hyperparameter search
- `metrics-prior.csv` - Section 3.2 Leveraging fast priors (drawing theta)
- `metrics-sameval.csv` -
- `metrics-sbibm-extra-extra.csv` - Section 3 Benchmark
- `metrics-sbibm-extra.csv` - Section 3 Benchmark
- `metrics-sbibm.csv` - Section 3 Benchmark

Each of these metric summaries were generated from [raw data](https://zenodo.org/record/7299588) within a folder with the corresponding name.

### `cnre/remote`
We depend on four packages.
- `diffeqtorch` - Necessary for the simulation-based inference benchmark problems lotka voltera and sir.
- `results` - Necessary to compare our results with the simulation-based inference benchmark.
- `sbi` - Contains alternative simulation-based inference algorithms.
- `sbibm` - Contains the toy problems which we test on in the paper.
