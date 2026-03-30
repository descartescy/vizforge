<p align="center">
  <h1 align="center">PlotCraft</h1>
  <p align="center">
    <strong>Reproduce journal-quality charts with just one line of code</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/plotcraft/"><img src="https://img.shields.io/pypi/v/plotcraft.svg" alt="PyPI version"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="Python"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
    <a href="https://github.com/descartescy/plotcraft/stargazers"><img src="https://img.shields.io/github/stars/descartescy/plotcraft?style=social" alt="GitHub stars"></a>
  </p>
</p>

---

## Introduction

**PlotCraft** is a Python visualization library built on top of Matplotlib, focused on creating chart types that are **not easily achievable** with Seaborn or Matplotlib alone.

When reading academic papers, have you ever come across beautifully crafted yet complex charts that you wanted to reproduce, only to find it would require hundreds of lines of code? PlotCraft was created to solve exactly this problem. We encapsulate classic chart styles from high-quality journals into concise Python functions, allowing you to generate publication-ready visualizations with just a few lines of code.

> 🚀 Continuously updated! If there's a chart you'd like to reproduce, feel free to submit an [Issue](https://github.com/descartescy/plotcraft/issues).

## Features

- 🎯 **Journal-Quality Output** — All chart styles follow visualization standards from high-impact-factor journal publications
- 🔧 **Ready to Use** — Quickly generate complex charts through a concise API, with no need to write extensive Matplotlib configurations
- 🎨 **Highly Customizable** — Every function returns a `(Figure, Axes)` tuple for easy further fine-tuning
- 📊 **Covers Multiple Scenarios** — From data distribution comparisons to model evaluation, from correlation analysis to clinical decision curves

## Installation

```bash
pip install plotcraft
```

## Dependencies

PlotCraft is built on the following commonly used scientific computing libraries:

- Python >= 3.12
- matplotlib >= 3.10.8
- numpy >= 2.4.3
- pandas >= 3.0.1
- scipy >= 1.17.1
- scikit-learn >= 1.8.0

## Chart Gallery

### 1. Train/Test Lift Comparison Histogram — `train_test_lift`

Displays the distributions of training and test sets in a stacked lift manner with dual Y-axis scales corresponding to the baselines of each dataset, providing an intuitive comparison of distribution differences.

```python
from plotcraft.draw import train_test_lift
import numpy as np
import matplotlib.pyplot as plt

train_data = np.arange(21, 100, dtype=int)
sigma, mu = 15, 60
y = np.exp(-(train_data - mu) ** 2 / (2 * sigma ** 2))
train_count = (y * 50 + 10).astype(int)

test_data = train_data.copy()
test_count = train_count.copy()

fig, ax = train_test_lift(
    [train_data, train_count],
    [test_data, test_count],
    paired=False,
)
ax.set_xlabel('Length', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11, labelpad=35)
plt.show()
```

**Key Parameters:** `paired` (whether to use paired format), `colors` (bar colors), `labels` (legend labels), `yticks_interval` (Y-axis tick interval), `offset` (test set lift height)

---

### 2. Triangular Heatmap — `triangular_heatmap`

Draws the lower triangle of a correlation coefficient matrix using diamond-shaped cells, with support for custom colormaps and value annotations — an elegant alternative to traditional square heatmaps.

```python
from plotcraft.draw import triangular_heatmap
import numpy as np
import pandas as pd
from scipy import stats

n_samples, n_vars = 200, 20
data = np.random.randn(n_samples, n_vars)
cols = [f"Var{i+1}" for i in range(n_vars)]
df = pd.DataFrame(data, columns=cols)

n = n_vars
corr = np.ones((n, n))
for i in range(n):
    for j in range(i + 1, n):
        r, _ = stats.spearmanr(df.iloc[:, i], df.iloc[:, j])
        corr[i, j] = r
        corr[j, i] = r

corr_df = pd.DataFrame(corr, index=cols, columns=cols)

fig, ax = triangular_heatmap(
    corr_df,
    annot=True,
    annot_kws={'size': 7.2},
    linewidths=0.5,
    ticks_size=8,
)
plt.show()
```

**Key Parameters:** `annot` (whether to display values), `annot_kws` (annotation style), `cmap` (colormap), `vmin/vmax` (color range), `norm` (normalization method)

---

### 3. ROC Curve with Inset Zoom — `enlarged_roc_curve`

Plots one or more ROC curves with automatic AUC calculation, supporting an inset zoom subplot within a specified region for fine-grained comparison of multi-model classification performance.

```python
from plotcraft.draw import enlarged_roc_curve
import numpy as np

arr = np.load('data/true_score.npy')
data_list = [[arr[i], arr[i+1]] for i in range(0, arr.shape[0], 2)]

fig, ax = enlarged_roc_curve(
    *data_list,
    labels=[f'model{i}' for i in range(len(data_list))],
    enlarged=True,
    to_enlarge_frame_location=[0.01, 0.80, 0.15, 0.98],
    enlarged_frame_location=[0.3, 0.5, 0.4, 0.4],
    enlarged_frame_xticks=[0.045, 0.08, 0.115],
    enlarged_frame_yticks=[0.9, 0.93, 0.96],
)
plt.show()
```

**Key Parameters:** `enlarged` (whether to enable zoom), `to_enlarge_frame_location` (zoom region coordinates), `enlarged_frame_location` (inset subplot position), `calculate` (whether to compute AUC)

---

### 4. PR Curve with Inset Zoom — `enlarged_pr_curve`

Usage is identical to `enlarged_roc_curve`, designed for plotting and zooming into Precision-Recall curves.

```python
from plotcraft.draw import enlarged_pr_curve
import numpy as np

arr = np.load('data/true_score.npy')
data_list = [[arr[i], arr[i+1]] for i in range(0, arr.shape[0], 2)]

fig, ax = enlarged_pr_curve(
    *data_list,
    labels=[f'model{i}' for i in range(len(data_list))],
    enlarged=True,
    to_enlarge_frame_location=[0.82, 0.75, 0.97, 0.93],
    enlarged_frame_location=[0.3, 0.5, 0.4, 0.4],
    enlarged_frame_xticks=[0.858, 0.895, 0.93],
    enlarged_frame_yticks=[0.795, 0.84, 0.885],
)
plt.show()
```

---

### 5. Predicted vs. Actual Scatter Plot — `correlation_graph_between_prediction_and_reality`

Plots a scatter chart of actual values versus predicted values, accompanied by a \(y = x\) reference line and a correlation coefficient (Pearson R by default), providing an at-a-glance assessment of regression model quality.

```python
from plotcraft.draw import correlation_graph_between_prediction_and_reality
import numpy as np

real = np.random.randn(1000)
pred = real + np.random.randn(1000) * 0.5

fig, ax = correlation_graph_between_prediction_and_reality(real, pred)
plt.show()
```

**Key Parameters:** `correlation` (custom correlation function; defaults to `scipy.stats.pearsonr`)

---

### 6. Decision Curve Analysis — `dca_curve`

Plots a Decision Curve Analysis (DCA) chart, calculating Standardized Net Benefit across different risk thresholds, with support for bootstrap confidence intervals and a cost-benefit ratio secondary axis. The methodology references the R package `dcurves`.

```python
from plotcraft.draw import dca_curve
import numpy as np
import pandas as pd

arr = np.load('data/true_score.npy')
datas = [
    pd.DataFrame(
        np.array([arr[i], arr[i+1]]).T,
        columns=['true', 'pred']
    )
    for i in range(0, arr.shape[0], 2)
]

# Basic usage
fig, ax = dca_curve(
    *datas,
    dataframe_cols=['true', 'pred'],
    thresholds=np.arange(0.01, 1.01, 0.01),
)
plt.show()

# With confidence intervals
fig, ax = dca_curve(
    datas[0],
    dataframe_cols=['true', 'pred'],
    thresholds=np.arange(0.01, 1.01, 0.01),
    confidence_intervals=0.95,
)
plt.show()
```

**Key Parameters:** `thresholds` (risk threshold grid), `confidence_intervals` (confidence level), `bootstraps` (number of bootstrap iterations), `policy` (`"opt-in"` / `"opt-out"`), `cost_benefit_axis` (whether to display the cost-benefit axis)

---

## API Conventions

All plotting functions follow a unified set of design conventions:

1. **Return Value**: All functions uniformly return a `(Figure, Axes)` tuple, making it easy to further customize titles, labels, save outputs, etc.
2. **Data Format**: The `paired` parameter flexibly supports both paired format `[[x1, y1], ...]` and separated format `[[x_vals], [y_vals]]`
3. **DataFrame Support**: Most functions accept both NumPy arrays and Pandas DataFrames as input
4. **Optional Zoom**: ROC/PR curves enable an inset zoom subplot with a single `enlarged=True` toggle

## Project Roadmap

- [x] Train/Test Distribution Lift Comparison Chart
- [x] Triangular Correlation Heatmap
- [x] ROC Curve with Inset Zoom
- [x] PR Curve with Inset Zoom
- [x] Predicted vs. Actual Correlation Scatter Plot
- [x] Decision Curve Analysis (DCA)
- [ ] More journal chart types (continuously updated…)

## Contributing

PlotCraft is a continuously growing project. If you've seen a beautifully crafted yet hard-to-reproduce chart in a paper, we'd love to hear about it!

**How to Submit:**

1. Go to [GitHub Issues](https://github.com/descartescy/plotcraft/issues)
2. Describe the chart type you'd like
3. Ideally, attach an image sample or paper reference

We will evaluate feasibility and implement it in future releases. Pull Requests are also very welcome!

## License

This project is open-sourced under the [MIT License](LICENSE).

## Citation

If PlotCraft has been helpful for your research, please consider citing it in your paper:

```
@software{plotcraft,
  title={PlotCraft: Journal-Quality Visualization Made Simple},
  author={descartescy},
  url={https://github.com/descartescy/plotcraft},
  year={2025}
}
```

