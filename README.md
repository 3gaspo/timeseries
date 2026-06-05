# Dataset exploration notebook

This project provides tools to inspect panel time series datasets.

## Files

- `general_utils.py`: Core data loading, preparation, normalization, statistics, splitting, and window/feature utilities.
- `clustering_utils.py`: Distance, hierarchical clustering, centroid, grouping, and heterogeneity utilities.
- `widgets_and_plots.py`: Interactive widgets and plots for series, windows, seasonality, statistics, distances, clustering, and projections.
- `dataset_visualization.ipynb`: Notebook for loading, inspecting, visualizing, comparing, and clustering time-series datasets.

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

<img width="675" height="271" alt="widget" src="https://github.com/user-attachments/assets/1c5423d9-37a2-46d1-8a48-48d37feb7a3b" />
<img width="425" height="260" alt="tsne" src="https://github.com/user-attachments/assets/9a2c8392-0024-41c9-93d5-bb83a70a14e2" />
<img width="250" height="175" alt="period" src="https://github.com/user-attachments/assets/ca5f94b0-7b83-4e7c-8f8d-50bf5d356f84" />
<img width="750" height="220" alt="clusters" src="https://github.com/user-attachments/assets/a4fd5c57-811a-45e2-bdec-7cd51da4a25c" />

