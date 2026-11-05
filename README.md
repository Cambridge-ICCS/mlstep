# MLStep: UKCA timestep-halving prediction

Predicts how many times the UKCA chemistry solver halves its timestep in each grid
cell.

## Original MLStep file structure:

- `data.py`	Handles the data. It defines the 267 inputs, reads the NetCDF files, turns ncsteps (1, 2, 4, 8, 16) into labels 0–4, holds the scaling methods, and samples rows for training.

- `model.py`	Defines the neural network, called Student, with one part predicting if a box halves and another predicting how many times. It also saves and loads models.

- `tabm.py`	Provides the network's building blocks. These let 4 small models share most of one network and average their predictions (the TabM-mini design).

- `embeddings.py`	Provides the PLE layer, used only by the ple_all and ple_tracer_residual variants. It turns each chosen input number into a small list of numbers, which can make patterns easier to learn.

- `evaluation.py`	Calculates the scores. It gets predictions from a neural network, computes metrics, picks thresholds, and counts correct, too-low and too-high predictions.

- `precision.py`	Picks the hardware settings. It sets the CPU or GPU device and the number precision used for training.

- `__init__.py`	Empty file. It just marks the folder as a Python package.

## Pipeline flow:

```
raw simulation files
        │
   cache.py (uses data.py)
        │
        ├──► xgb.py ───── teacher targets ──┐
        │                                   ▼
        └──────────────────────────► student.py (uses model.py, tabm.py, embeddings.py)
                                            │
                                     generalization.py (uses evaluation.py)
```


## Dataset

UKCA atmospheric chemistry simulation data one NetCDF file per variable per timestep.
Total 360 timesteps on a 38 × 72 × 96 grid (height, latitude, longitude). 

Stored in RDS: `/rds/project/rds-jzEwvbIgOag/users/jw2423`

**Inputs:** 267 values per row describing a grid cell at one timestep 
- `temp`, `pres` : 2
- `cloud_frac`, `qcl`, `qcf` : 3
- `cell_volume` : 1
- `stratflag` : 1 
- `tracer` : 83
- `residual` : 83
- `photol_rates` : 60
- `wetrt` : 34


**Output:**  `ncsteps` i.e. the number of halvings needed by the solver to converge for each grid cell. Values rage from 0 to 4, which means 1, 2, 4, 8 or 16 sub-steps.

## ML Models compared

**Baseline:** XGBoost 

**Other Models:** TabM-mini (Ensemble)

There are 8 different ways in which TabM-mini is used, based on 8 different combinations of learning methods and input preparation

Learning type: supervised, distilled

Input preparation: physical, stretch128, ple_all, ple_tracer_residual


## Training setup

**Data Split:**  Training 70%, Validation 15%, Test 15%

**Metrics for scoring models:** AP (average precision), RPS (ranked probability score), NLL (negative log-likelihood), Brier score 

**Metrics for final nc step prediction:** precision, recall, confusion matrix

