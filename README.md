# Gavel-Replication-Package

## Description

This repository hosts the implementation and datasets for two main components: GAVEL and CNBN. CNBN serves as the baseline for our comparisons, while GAVEL is our newly proposed system designed to improve upon the baseline in specific tasks.

## Repository Structure

- **CNBN** (Baseline): 
  - `gh_action_dataset`: Includes datasets like `all.csv`, `code_block.csv`, and results from multiple 5-fold cross-validation setups.
  - `tools`: Contains scripts such as `dataLoader.py`, `main.py`, and `predictor.py` to facilitate data handling and predictions.

- **GAVEL** (Proposed System): 
  - `CONFS`: Configuration files (`C1.csv` to `C5.csv`) for setting up experimental conditions.
  - `RESULTS`: Contains experimental outputs like `Fold_1_Results.csv`, utilities for converting classification reports to LaTeX (`classification_report2latex.py`), and statistical analysis tools (`stats.py`).
  - `TOOLS`: Includes scripts like `gavel.py` which are central to the GAVEL system.

