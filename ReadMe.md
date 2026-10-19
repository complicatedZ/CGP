# Coarse-graining metabolic networks via feature learning reveals cross-species growth predictors


## 1. Installation

Clone the repository:

```bash
git clone https://github.com/complicatedZ/CGP.git
cd CGP
```

Create and activate the Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate CGP
```

## 2. Download simulation results

The precomputed flux balance analysis (FBA) solutions are provided separately as a compressed archive (`FBAsolution.tar.gz`) in the [GitHub Releases](https://github.com/complicatedZ/CGP/releases) section.

Download `FBAsolution.tar.gz` from the release assets and place it in the root directory of the repository:

```text
CGP/
└── FBAsolution.tar.gz
```

Extract the archive:

```bash
tar -xzf FBAsolution.tar.gz
```


## 3. Repository structure

```text
CGP/
├── AGORA2_models/
├── Coarse_graining_result/
├── Expression_datasets/
├── FBAsimulation/
├── FBAsolution/                 # Download separately
├── plot_figure/
├── random_walk/
├── helper_function.py
├── universal_model.mat
├── universal_model_reactions&subsystems.xlsx
├── environment.yml
└── ReadMe.md
```

### Directory and file descriptions

| Directory / File | Description |
|---|---|
| `AGORA2_models/` | Selected genome-scale metabolic models from the AGORA2 database used in this study. |
| `Coarse_graining_result/` | Results of metabolic network coarse-graining and sector identification. |
| `Expression_datasets/` | Independent experimental datasets collected from previously published studies for downstream analyses and validation. |
| `FBAsimulation/` | Scripts for simulating microbial growth using FBA and constructing reaction-based graphs from the simulations. |
| `FBAsolution/` | Precomputed FBA solutions used for downstream analyses. Available separately from GitHub Releases. |
| `plot_figure/` | Scripts used to generate figures and visualizations for the study. |
| `random_walk/` | Simulations of 151 randomized metabolic networks constructed using a modified universal metabolic model. |
| `helper_function.py` | Shared helper functions used throughout the computational analyses. |
| `universal_model.mat` | Universal metabolic model constructed from the selected AGORA2 models. |
| `universal_model_reactions&subsystems.xlsx` | Reaction and subsystem annotations for the universal metabolic model. |
| `environment.yml` | Conda environment specification containing the software dependencies. |