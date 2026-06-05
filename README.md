# Dataset notebook refactor

This bundle splits the original monolithic `analysis.py` into importable modules and updates the notebook to use them directly.

## Files

- `general_utils.py`: data preparation, normalization, train/test splitting, descriptive statistics, gamma/Fourier views, and window sampling helpers.
- `clustering_utils.py`: pairwise distances, hierarchical clustering, cluster dictionaries, centroids, and cluster heterogeneity metrics.
- `widgets_and_plots.py`: all notebook-facing plots and interactive widgets, including series/window exploration, seasonality, stats, distances, clustering, t-SNE, MDS, and UMAP.
- `analysis.py`: a small compatibility shim that re-exports the split modules for older code.
- `dataset_refactored.ipynb`: notebook rewritten to import the split modules and to load CSV data with the Google Drive / local `data_path` pattern.

## Notebook summary

The notebook explores panel time-series datasets where rows are dates or time steps and columns are individual series.

1. **Setup** mounts Google Drive when `on_drive=True`, switches to the code directory, imports the split modules, and defines `reload_modules()`.
2. **Data loading** reads `data_path + data_name + ".csv"`, prepares the dataframe with `prepare_pandas_data`, prints raw/prepared shapes and index information, and displays the first rows.
3. **Features** computes dataset metadata with `time_series_features`, including sampling rate, missing values, lag/seasonality summaries, distribution statistics, correlations, distances, and spectral centroid.
4. **Visualization** provides widgets for raw/gamma/trend/aggregate series, individual lookback/horizon windows, constant-window diagnostics, circular seasonality, and 2D user projections.
5. **Stats** samples windows and compares mean/std and alpha/beta distributions for individual users or dataset splits.
6. **Distances** computes energy-distance matrices between train/validation/test splits under raw, standard, and instance normalization.
7. **Clustering** runs hierarchical clustering on raw, Fourier, or gamma views, plots dendrograms, centroids, cluster-level stats, and heterogeneity as the number of clusters varies.
8. **Projections** compares dataset splits with t-SNE and UMAP embeddings built from sampled windows.

## Notes on reused code

I copied the data preparation and forecast-window helpers from `forecast_utils.py` into `general_utils.py` because those functions are cleaner and more complete than the loader embedded in the notebook. In particular, `prepare_pandas_data` supports orientation changes, optional date columns, dropping columns by name or index, custom names, and optional resampling.

I did not copy `plot_prediction` because this notebook is an exploratory dataset-analysis notebook rather than a forecasting-evaluation notebook.

## Colab usage

Upload/copy these `.py` files next to the notebook in:

```text
/content/drive/MyDrive/Recherche/Thèse Gaspard/Codes/
```

Then set:

```python
on_drive = True
data_name = "electricity"
```

The dataset CSV should be available at:

```text
/content/drive/MyDrive/Recherche/Thèse Gaspard/Datasets/electricity.csv
```

For local use, set `on_drive = False` and keep the CSV in the same directory as the notebook, or edit `data_path`.
