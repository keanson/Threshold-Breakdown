# The Threshold Breakdown Point

Official python implementation for the paper:
**"The Threshold Breakdown Point"**

## Structure

The codebase is organized as a small Python package under [`src/ftbp`](./src/ftbp), with exploratory and paper-facing notebooks under [`src/ftbp/experiments`](./src/ftbp/experiments) and generated figures under [`src/ftbp/results`](./src/ftbp/results).

- `src/ftbp/estimators.py`: robust score functions and M-estimation utilities
- `src/ftbp/score.py`: M-score tests related breakdown and bound calculations
- `src/ftbp/wald.py`: Wald-style procedures and related calculations
- `src/ftbp/bootstrap.py`: bootstrap-based procedures
- `src/ftbp/two_stage.py`: two-stage procedures
- `src/ftbp/io.py`: data-loading helpers
- `src/ftbp/calcium.csv`: dataset used in the real-data analysis
- `src/ftbp/experiments/`: Jupyter notebooks for experiments and figures
- `src/ftbp/results/`: exported PDF figures

## Setup

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Since the package uses a `src/` layout, run notebooks or scripts from the repository root so they can resolve `src/ftbp` correctly. The notebooks already include path setup cells for that purpose.

## Main Dependencies

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tqdm`
- `jupyterlab`

## Notes

- Generated figures used in the paper are kept in `src/ftbp/results`.
- Although the code is organized in package form, this repository is mainly provided for research reproducibility. The package structure is used to make experiments, dependencies, and workflows easier to maintain and reproduce. The codebase is therefore best understood as paper companion code rather than a polished public library.

