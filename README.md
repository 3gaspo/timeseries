# Dataset exploration notebook

This project provides tools to inspect panel time-series datasets, where rows are timestamps and columns are users, sensors, or individual series.

## Files

- `general_utils.py`: data preparation, normalization, train/validation/test splitting, Fourier and gamma feature extraction, dataset statistics, and window sampling.
- `clustering_utils.py`: distance computation, hierarchical clustering, cluster centroids, cluster grouping, and heterogeneity scores.
- `widgets_and_plots.py`: interactive notebook widgets and plots for time series, windows, seasonality, statistics, distances, clustering, and 2D projections.
- `analysis.py`: convenience import file that exposes the utilities from the other modules.
- `dataset_refactored.ipynb`: notebook for loading a dataset, computing statistics, visualizing patterns, comparing splits, and clustering users or sensors.

## Notebook overview

The notebook helps analyze a multivariate time-series dataset from several angles:

1. Load a CSV dataset from Google Drive or a local path.
2. Prepare the data into a clean `pandas.DataFrame`.
3. Display basic dataset information: shape, index type, date range, and column names.
4. Compute descriptive statistics such as sampling rate, missing values, skewness, kurtosis, correlations, distances, lags, and seasonalities.
5. Visualize individual series, trends, aggregates, gamma features, and sampled input/forecast windows.
6. Explore daily, weekly, and yearly seasonality with circular plots.
7. Compare sampled windows using mean/std and alpha/beta statistics.
8. Compare train, validation, and test splits with energy-distance matrices.
9. Cluster users or sensors using raw, Fourier, or gamma representations.
10. Visualize sampled windows with t-SNE, MDS, and UMAP.

## Data loading

In Colab, place the scripts next to the notebook in:

```text
/content/drive/MyDrive/Recherche/Thèse Gaspard/Codes/
