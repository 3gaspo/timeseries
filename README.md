# Dataset exploration notebook

This project provides tools to inspect panel time series datasets.

## Files

- `general_utils.py`: Core data loading, preparation, normalization, statistics, splitting, and window/feature utilities.
- `clustering_utils.py`: Distance, hierarchical clustering, centroid, grouping, and heterogeneity utilities.
- `widgets_and_plots.py`: Interactive widgets and plots for series, windows, seasonality, statistics, distances, clustering, and projections.
- `analysis.py`: Single import entry point for the utility modules.
- `dataset_refactored.ipynb`: Notebook for loading, inspecting, visualizing, comparing, and clustering time-series datasets.

## Notebook overview

The notebook helps analyze a multivariate time-series dataset from several angles:

1. Load a CSV dataset from Google Drive or a local path.
2. Prepare the data into a clean `pandas.DataFrame`.
3. Display basic dataset information: shape, index type, date range, and column names.
4. Compute descriptive statistics such as sampling rate, missing values, skewness, kurtosis, correlations, distances, lags, and seasonalities.
5. Visualize individual series, trends, aggregates, gamma features, and sampled input/forecast windows.
6. Explore daily, weekly, and yearly seasonality with circular plots.
7. Compare sampled windows using mean/std statistics.
8. Compare train, validation, and test splits with energy-distance matrices.
9. Cluster users or sensors using raw, Fourier, or gamma representations.
10. Visualize sampled windows with t-SNE, MDS, and UMAP.
