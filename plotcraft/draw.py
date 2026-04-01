import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from typing import Union, List, Optional, Literal, Callable
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from .utils import floor_significant_digits, calculate_nb, _threshold_to_cost_benefit
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as patches
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy import stats
import warnings
import matplotlib.ticker as ticker
from scipy.special import logit as qlogis, expit as plogis
from scipy.stats import norm, chi2
from scipy.optimize import minimize
import sympy as sp
from matplotlib.lines import Line2D


def train_test_lift(
        train:Union[List[List], np.ndarray],
        test:Union[List[List], np.ndarray],
        paired:bool=True,
        colors:Optional[List[str]]=None,
        labels:Optional[List[str]]=None,
        yticks_interval:Optional[int|float]=None,
        axis_range:Optional[List[Optional[int|float]]]=None,
        offset:Optional[int|float]=None
) -> tuple[Figure,Axes]:
    """
    Plot lifted histogram comparison between training and test distributions.

    Visualize two groups of data (training vs test) as bar charts, with the test
    bars lifted vertically for clear separation. Dual Y-axis ticks are drawn on
    the left and right to match each distribution’s baseline. Suitable for length
    distribution, value count, or density comparison in data analysis pipelines.

    Parameters
    ----------
    train : list of lists or np.ndarray
        Training data, either as paired [[x1, y1], ...] or separated [x_vals, y_vals].
    test : list of lists or np.ndarray
        Test data, in the same format as training data.
    paired : bool, default=True
        If True, input arrays are treated as paired points: [[x1, y1], [x2, y2], ...].
        If False, inputs are separated coordinates: ([x1, x2, ...], [y1, y2, ...]).
    colors : list of str, optional
        Two-element color list for training and test bars.
        Defaults to muted dark pink and deep blue.
    labels : list of str, optional
        Legend labels for training and test sets. Default: ["Train", "Test"].
    yticks_interval : int or float, optional
        Step interval for Y-axis ticks. If None, computed automatically from data range.
    axis_range : list of int/float/None, optional
        Axis limits in the form [X_min, X_max, Y_min, Y_max].
        Use None to auto-compute a given limit.
    offset : int or float, optional
        Vertical offset to lift test bars. If None, set to half the tick interval.

    Returns
    -------
    Figure
        The figure object containing the plot.
    Axes
        Matplotlib Axes object containing the finished plot for further styling.

    Examples
    --------
    >>> from plotcraft.draw import train_test_lift
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> train_data = np.arange(21, 100,dtype=int)
    >>> sigma, mu = 15, 60
    >>> y = np.exp(-(train_data - mu) ** 2 / (2 * sigma ** 2))
    >>> train_count = (y * 50 + 10).astype(int)
    >>> test_data = train_data.copy()
    >>> test_count = train_count.copy()
    >>> fig, ax = train_test_lift([train_data,train_count],[test_data,test_count],paired=False)
    >>> ax.set_xlabel('Length', fontsize=11)
    >>> ax.set_ylabel('Frequency', fontsize=11, labelpad=35)
    >>> plt.show()
    """
    train = np.array(train)
    test = np.array(test)
    if paired:
        train_x = train[:,0]
        train_y = train[:,1]
        test_x = test[:,0]
        test_y = test[:,1]
    else:
        train_x, train_y = train
        test_x, test_y = test

    if axis_range is None:
        X_min = min(min(train_x), min(test_x))
        X_max = max(max(train_x), max(test_x))
        Y_min = 0
        Y_max = max(max(train_y), max(test_y))
    else:
        X_min, X_max, Y_min, Y_max = axis_range
        if X_min is None:
            X_min = min(min(train_x), min(test_x))
        if X_max is None:
            X_max = max(max(train_x), max(test_x))
        if Y_min is None:
            Y_min = 0
        if Y_max is None:
            Y_max = max(max(train_y), max(test_y))

    if labels is None:
        labels = ["Train", "Test"]

    if colors is None:
        colors = ['#E0726D', '#5187B0']

    if yticks_interval is None:
        yticks_interval = floor_significant_digits((Y_max - Y_min)/4, 2)
    tick_vals = np.arange(Y_min,Y_max,yticks_interval)

    if offset is None:
        offset = yticks_interval / 2

    fig, ax = plt.subplots()

    ax.bar(train_x, train_y, alpha=0.5,
           color=colors[0], edgecolor='white', linewidth=0.5, label=labels[0])

    ax.bar(test_x, test_y, bottom=offset, alpha=0.5,
           color=colors[1], edgecolor='white', linewidth=0.5, label=labels[1])

    ax.set_xlim(X_min-1, X_max+1)
    ax.set_ylim(Y_min, Y_max + offset)
    ax.set_yticks([])

    ax.axhline(y=offset, color='#888888', linestyle='--', linewidth=1.5, dashes=(5, 2), alpha=0.8)

    blend = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    for i, v in enumerate(tick_vals):
        ax.text(-0.03, v, f'{v:.2f}', transform=blend,
                fontsize=8, color=colors[0], va='center', ha='right')
        ax.plot([-0.02, 0], [v, v], color=colors[0], linewidth=0.8,
                clip_on=False, transform=blend)

        if i:
            ax.text(0.03, v + offset, f'{v:.2f}', transform=blend,
                    fontsize=8, color=colors[1], va='center', ha='left')
            ax.plot([0, 0.02], [v + offset, v + offset], color=colors[1],
                    linewidth=0.8, clip_on=False, transform=blend)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.legend(frameon=True, fontsize=9, loc='upper right')
    plt.subplots_adjust(left=0.18)
    return fig, ax


def triangular_heatmap(
        data: pd.DataFrame | np.ndarray,
        annot: bool = True,
        annot_kws: Optional[dict] = None,
        linewidths: float | int = 1.5,
        linecolor: str = 'white',
        ticks_size: int | float = 9,
        vmin: float | int = -1,
        vmax: float | int = 1,
        cmap: str | plt.Colormap = None,
        norm: Normalize = None
) -> tuple[Figure,Axes]:
    """
    Draw a heatmap of a triangle.

    This function creates a triangular heatmap using diamond-shaped cells to visualize
    the lower triangular part of a square correlation matrix. It supports custom color
    mapping, value annotations, and styling of cell borders and labels.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Square matrix (n×n) containing correlation values. Only the lower triangular
        part of the matrix will be visualized. If a DataFrame is provided, column names
        will be used as variable labels; if a numpy array is provided, labels will be
        automatically generated as Var1, Var2, ..., Varn.

    annot : bool, default=True
        Whether to display numerical values inside each diamond cell.

    annot_kws : dict or None, default=None
        Keyword arguments for customizing the annotation text. Supported keys:
        - 'size': Font size of the annotation (default: 20)
        - 'color': Fixed text color; if not specified, text color will be white for
          values with absolute value > 0.60, otherwise dark gray (#222222)
        - 'fontweight': Font weight (default: 'bold')
        - 'fontfamily': Font family (default: None, inherits global settings)

    linewidths : float or int, default=1.5
        Width of the border lines between diamond cells.

    linecolor : str, default='white'
        Color of the border lines between diamond cells.

    ticks_size : float or int, default=9
        Font size of the variable name labels on the triangular axes.

    vmin : float or int, default=-1
        Minimum value for color normalization. Values less than or equal to vmin
        will be mapped to the bottom color of the colormap.

    vmax : float or int, default=1
        Maximum value for color normalization. Values greater than or equal to vmax
        will be mapped to the top color of the colormap.

    cmap : str or matplotlib.colors.Colormap, default=None
        Colormap used for mapping correlation values to colors. If None, 'RdBu_r'
        (red-blue reversed) will be used.

    norm : matplotlib.colors.Normalize, default=None
        Normalization object to scale data values to the [0, 1] range for colormap
        mapping. If None, a basic Normalize instance with vmin and vmax will be used.
        Other options include CenteredNorm or TwoSlopeNorm for asymmetric scaling.

    Returns
    -------
    Figure
        The figure object containing the plot.
    Axes
        Matplotlib Axes object containing the finished plot for further styling.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy import stats
    >>> from plotcraft.draw import triangular_heatmap
    >>> n_samples, n_vars = 200, 20
    >>> data = np.random.randn(n_samples, n_vars)
    >>> cols = [f"Var{i+1}" for i in range(n_vars)]
    >>> df = pd.DataFrame(data, columns=cols)
    >>> n = n_vars
    >>> corr = np.ones((n, n))
    >>> for i in range(n):
    ...     for j in range(i + 1, n):
    ...         r, _ = stats.spearmanr(df.iloc[:, i], df.iloc[:, j])
    ...         corr[i, j] = r
    ...         corr[j, i] = r
    >>> corr_df = pd.DataFrame(corr, index=cols, columns=cols)
    >>> fig, ax = triangular_heatmap(
    ...     corr_df,
    ...     annot=True,
    ...     annot_kws={'size': 7.2},
    ...     linewidths=0.5,
    ...     linecolor='white',
    ...     ticks_size=8,
    ...     vmax=1,
    ...     vmin=-1,
    ... )
    >>> plt.show()
    """

    assert vmax > vmin
    if isinstance(data, pd.DataFrame):
        columns = list(data.columns)
        corr = data.values
    else:
        corr = np.asarray(data)
        columns = [f"Var{i+1}" for i in range(corr.shape[0])]

    n = corr.shape[0]
    assert corr.shape == (n, n), "data 必须是方阵"

    _annot_kws = {'size': 20, 'fontweight': 'bold', 'fontfamily': None, 'color': None}
    if annot_kws:
        _annot_kws.update(annot_kws)

    def to_canvas(row, col):
        cx = 2 * (n - 1) - (row + col)
        cy = row - col
        return cx, cy

    half = 1.0

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    if cmap is None:
        cmap = 'RdBu_r'
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if norm is None:
        norm_c = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm_c = norm

    for row in range(n):
        for col in range(row + 1):
            val   = corr[row, col]
            color = cmap(norm_c(val))
            cx, cy = to_canvas(row, col)

            diamond = patches.Polygon(
                [(cx, cy+half), (cx+half, cy), (cx, cy-half), (cx-half, cy)],
                closed=True,
                facecolor=color,
                edgecolor=linecolor,
                linewidth=linewidths,
                zorder=2,
            )
            ax.add_patch(diamond)

            if annot:
                if _annot_kws['color'] is not None:
                    txt_color = _annot_kws['color']
                else:
                    txt_color = 'white' if abs(val) > 0.60 else '#222222'

                txt_kws = dict(
                    ha='center', va='center', zorder=3,
                    fontsize=_annot_kws['size'],
                    color=txt_color,
                    fontweight=_annot_kws['fontweight'],
                )
                if _annot_kws['fontfamily']:
                    txt_kws['fontfamily'] = _annot_kws['fontfamily']

                ax.text(cx, cy, f'{val:.2f}', **txt_kws)

    t = n * 0.005 + 0.6
    offset = 0.18
    sq2    = np.sqrt(2)

    for i in range(n):
        cx, cy = to_canvas(i, 0)
        lx  = cx + half * t + offset / sq2
        ly  = cy + half * (1 - t) + offset / sq2
        ax.text(lx, ly, columns[i],
                ha='left', va='bottom',
                fontsize=ticks_size, rotation=45,
                rotation_mode='anchor', zorder=4)

        cx2, cy2 = to_canvas(n - 1, i)
        lx2 = cx2 - half * t - offset / sq2
        ly2 = cy2 + half * (1 - t) + offset / sq2
        ax.text(lx2, ly2, columns[i],
                ha='right', va='bottom',
                fontsize=ticks_size, rotation=-45,
                rotation_mode='anchor', zorder=4)

    sm = ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.01, shrink=0.65, aspect=22)
    cbar.set_ticks(np.linspace(vmin,vmax,9))
    cbar.ax.tick_params(labelsize=8.5)
    cbar.outline.set_linewidth(0.5)

    ax.set_xlim(-half - 3.0, 2*(n-1) + half + 3.0)
    ax.set_ylim(-half - 0.5, (n-1) + half + 2.5)

    plt.tight_layout()
    return fig, ax

def enlarged_roc_curve(
        *true_score_pairs: List[List] | np.ndarray | pd.DataFrame,
        dataframe_cols: List[str] = None,
        colors:Optional[List[str]]=None,
        labels:Optional[List[str]]=None,
        paired:bool=False,
        calculate:bool=True,
        plot_kwargs:dict=None,
        enlarged:bool=False,
        to_enlarge_frame_location:List[int|float]=None,
        enlarged_frame_location:List[int|float]=None,
        enlarged_frame_xticks:List[int|float]=None,
        enlarged_frame_yticks:List[int|float]=None,
        enlarged_frame_transparent:bool=True,
        legend_kwargs:dict=None
) -> tuple[Figure,Axes]:
    """
    Plot ROC curves with optional local zoom-in functionality.

    Convenience function to draw ROC curves for one or multiple models,
    compute AUC scores, and add an inset axes to magnify a region of interest
    in the ROC space (typically low FPR, high TPR).

    Parameters
    ----------
    *true_score_pairs : sequence of array-like | dataframe
        Each argument is a pair (y_true, y_score). Multiple pairs can be
        passed to compare PR curves across models.

    dataframe_cols : list of str, default=None
        If you input "dataframe", please enter a one-dimensional list of length 2, like[true_column_name, score_column_name].
        If it is None, then the default list will be ["true", "score"].

    colors : list of str, default=None
        List of colors for each ROC curve. Length must match the number
        of model pairs provided.

    labels : list of str, default=None
        List of labels for each ROC curve. Length must match the number
        of model pairs provided.

    paired : bool, default=False
        If True, each input pair is expected to be an N x 2 array
        where each row is [y_true, score].
        If False, each input pair is interpreted as two 1D arrays:
        [y_true_array, score_array].

    calculate : bool, default=True
        Whether to compute and display AUC in the legend label.

    plot_kwargs : dict, default=None
        Keyword arguments passed to ax.plot() for ROC curves,
        e.g., linewidth, linestyle, alpha.

    enlarged : bool, default=False
        Whether to create an inset axes with a zoomed view of a subregion.

    to_enlarge_frame_location : list of float, length 4
        Region in main axes to magnify, specified as [x1, y1, x2, y2]
        in [0,1] coordinates, where (x1,y1) is lower-left and (x2,y2) upper-right.

    enlarged_frame_location : list of float, length 4
        Position of the inset axes within the main axes, in relative coordinates:
        [x1, y1, x2, y2] lower-left to upper-right.

    enlarged_frame_xticks : array-like, default=None
        Custom tick positions for the x-axis of the inset plot.

    enlarged_frame_yticks : array-like, default=None
        Custom tick positions for the y-axis of the inset plot.

    enlarged_frame_transparent : bool, default=True
        Whether to make the background of the inset plot transparent.

    legend_kwargs : dict, default=None
        Keyword arguments passed to ax.legend(), e.g., fontsize, loc.

    Returns
    -------
    Figure
        The figure object containing the plot.
    Axes
        Matplotlib Axes object containing the finished plot for further styling.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from plotcraft.draw import enlarged_roc_curve
    >>> arr = np.load('examples/data/true_score.npy')
    >>> data_list = [[arr[i], arr[i+1]] for i in range(0, arr.shape[0], 2)]
    >>> fig, ax = enlarged_roc_curve(
    ...     *data_list,
    ...     labels=[f'model{i}' for i in range(len(data_list))],
    ...     enlarged=True,
    ...     to_enlarge_frame_location=[0.01, 0.80, 0.15, 0.98],
    ...     enlarged_frame_location=[0.3, 0.5, 0.4, 0.4],
    ...     enlarged_frame_xticks=[0.045, 0.08, 0.115],
    ...     enlarged_frame_yticks=[0.9, 0.93, 0.96]
    ... )
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=(8,8))

    ax.plot([0, 1], [0, 1], color="lightgray", linestyle="--")

    fpr_list, tpr_list = [], []
    for i, true_score_pair in enumerate(true_score_pairs):
        if isinstance(true_score_pair, pd.DataFrame):
            if dataframe_cols is None:
                dataframe_cols = ['true', 'score']
            y_true, score = true_score_pair[dataframe_cols[0]].values, true_score_pair[dataframe_cols[1]].values
        else:
            true_score_pair = np.array(true_score_pair)
            if paired:
                y_true, score = true_score_pair[:, 0], true_score_pair[:, 1]
            else:
                y_true, score = true_score_pair
        fpr, tpr, _ = roc_curve(y_true, score)
        if calculate:
            roc_auc = auc(fpr, tpr)
            add_str = f"(AUC = {roc_auc:.3f})"
        else:
            add_str = ""
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        parameters = {}
        if colors is not None:
            parameters['color'] = colors[i]
        if labels is not None:
            parameters['label'] = labels[i] + add_str
        if plot_kwargs is not None:
            parameters.update(plot_kwargs)
        else:
            parameters['linewidth'] = 2

        ax.plot(fpr, tpr, **parameters)

    ax.spines[["top", "left"]].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.set_xlabel("False positive rate", fontsize=22, labelpad=10)
    ax.set_ylabel("True positive rate", fontsize=22, labelpad=20)
    ax.set_title("ROC curve", fontsize=22, pad=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if labels is not None:
        if legend_kwargs is None:
            legend_kwargs = {'fontsize':12}
        ax.legend(loc="lower right",**legend_kwargs)
    ax.grid(False)

    if enlarged:
        assert to_enlarge_frame_location is not None
        assert enlarged_frame_location is not None
        x1, y1, x2, y2 = to_enlarge_frame_location
        assert 0 <= x1 < x2 <=1
        assert 0 <= y1 < y2 <=1
        axins = ax.inset_axes(enlarged_frame_location,
                              xlim=(x1, x2), ylim=(y1, y2))

        if enlarged_frame_transparent:
            axins.patch.set_alpha(0.0)

        for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
            parameters = {}
            if colors is not None:
                parameters['color'] = colors[i]
            if plot_kwargs is not None:
                parameters.update(plot_kwargs)
            else:
                parameters['linewidth'] = 2
            axins.plot(fpr, tpr, **parameters)

        axins.yaxis.tick_right()
        if enlarged_frame_xticks is not None:
            axins.set_xticks(enlarged_frame_xticks)
        if enlarged_frame_yticks is not None:
            axins.set_yticks(enlarged_frame_yticks)
        axins.grid(False)

        ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)

    plt.tight_layout()
    return fig, ax

def enlarged_pr_curve(
        *true_score_pairs: List[List] | np.ndarray | pd.DataFrame,
        dataframe_cols:List[str]=None,
        colors:Optional[List[str]]=None,
        labels:Optional[List[str]]=None,
        paired:bool=False,
        calculate:bool=True,
        plot_kwargs:dict=None,
        enlarged:bool=False,
        to_enlarge_frame_location:List[int|float]=None,
        enlarged_frame_location:List[int|float]=None,
        enlarged_frame_xticks:List[int|float]=None,
        enlarged_frame_yticks:List[int|float]=None,
        enlarged_frame_transparent:bool=True,
        legend_kwargs:dict=None
) -> tuple[Figure, Axes]:
    """
    Plot PR curves with optional local zoom-in functionality.

    Convenience function to draw PR curves for one or multiple models,
    compute AUC scores, and add an inset axes to magnify a region of interest
    in the PR space (typically high Recall, high Precision).

    Parameters
    ----------
    *true_score_pairs : sequence of array-like | dataframe
        Each argument is a pair (y_true, y_score). Multiple pairs can be
        passed to compare PR curves across models.

    dataframe_cols : list of str, default=None
        If you input "dataframe", please enter a one-dimensional list of length 2, like[true_column_name, score_column_name].
        If it is None, then the default list will be ["true", "score"].

    colors : list of str, default=None
        List of colors for each PR curve. Length must match the number
        of model pairs provided.

    labels : list of str, default=None
        List of labels for each PR curve. Length must match the number
        of model pairs provided.

    paired : bool, default=False
        If True, each input pair is expected to be an N x 2 array
        where each row is [y_true, score].
        If False, each input pair is interpreted as two 1D arrays:
        [y_true_array, score_array].

    calculate : bool, default=True
        Whether to compute and display AUC in the legend label.

    plot_kwargs : dict, default=None
        Keyword arguments passed to ax.plot() for PR curves,
        e.g., linewidth, linestyle, alpha.

    enlarged : bool, default=False
        Whether to create an inset axes with a zoomed view of a subregion.

    to_enlarge_frame_location : list of float, length 4
        Region in main axes to magnify, specified as [x1, y1, x2, y2]
        in [0,1] coordinates, where (x1,y1) is lower-left and (x2,y2) upper-right.

    enlarged_frame_location : list of float, length 4
        Position of the inset axes within the main axes, in relative coordinates:
        [x1, y1, x2, y2] lower-left to upper-right.

    enlarged_frame_xticks : array-like, default=None
        Custom tick positions for the x-axis of the inset plot.

    enlarged_frame_yticks : array-like, default=None
        Custom tick positions for the y-axis of the inset plot.

    enlarged_frame_transparent : bool, default=True
        Whether to make the background of the inset plot transparent.

    legend_kwargs : dict, default=None
        Keyword arguments passed to ax.legend(), e.g., fontsize, loc.

    Returns
    -------
    Figure
        The figure object containing the plot.
    Axes
        Matplotlib Axes object containing the finished plot for further styling.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> arr = np.load('./data/true_score.npy')
    >>> data_list = [[arr[i], arr[i+1]] for i in range(0, arr.shape[0], 2)]
    >>> fig, ax = enlarged_pr_curve(*data_list,
    ...     labels=[f'model{i}' for i in range(len(datas))],
    ...     enlarged=True,
    ...     to_enlarge_frame_location=[0.82,0.75,0.97,0.93],
    ...     enlarged_frame_location=[0.3, 0.5, 0.4, 0.4],
    ...     enlarged_frame_xticks=[0.858,0.895,0.93],
    ...     enlarged_frame_yticks=[0.795, 0.84, 0.885]
    ... )
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    precision_list, recall_list = [], []
    for i, true_score_pair in enumerate(true_score_pairs):
        if isinstance(true_score_pair, pd.DataFrame):
            if dataframe_cols is None:
                dataframe_cols = ['true', 'score']
            y_true, score = true_score_pair[dataframe_cols[0]].values, true_score_pair[dataframe_cols[1]].values
        else:
            true_score_pair = np.array(true_score_pair)
            if paired:
                y_true, score = true_score_pair[:, 0], true_score_pair[:, 1]
            else:
                y_true, score = true_score_pair
        precision, recall, _ = precision_recall_curve(y_true, score)
        if calculate:
            AP = average_precision_score(y_true, score)
            add_str = f"(AUC = {AP:.3f})"
        else:
            add_str = ""
        precision_list.append(precision)
        recall_list.append(recall)
        parameters = {}
        if colors is not None:
            parameters['color'] = colors[i]
        if labels is not None:
            parameters['label'] = labels[i] + add_str
        if plot_kwargs is not None:
            parameters.update(plot_kwargs)
        else:
            parameters['linewidth'] = 2

        ax.plot(recall, precision, **parameters)

    ax.spines[["top", "right"]].set_visible(False)

    ax.set_xlabel("Recall", fontsize=22, labelpad=10)
    ax.set_ylabel("Precision", fontsize=22, labelpad=20)
    ax.set_title("PR curve", fontsize=22, pad=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if labels is not None:
        if legend_kwargs is None:
            legend_kwargs = {'fontsize': 12}
        ax.legend(loc="lower left", **legend_kwargs)
    ax.grid(False)

    if enlarged:
        assert to_enlarge_frame_location is not None
        assert enlarged_frame_location is not None
        x1, y1, x2, y2 = to_enlarge_frame_location
        assert 0 <= x1 < x2 <= 1
        assert 0 <= y1 < y2 <= 1
        axins = ax.inset_axes(enlarged_frame_location,
                              xlim=(x1, x2), ylim=(y1, y2))
        if enlarged_frame_transparent:
            axins.patch.set_alpha(0.0)

        for i, (recall, precision) in enumerate(zip(recall_list, precision_list)):
            parameters = {}
            if colors is not None:
                parameters['color'] = colors[i]
            if plot_kwargs is not None:
                parameters.update(plot_kwargs)
            else:
                parameters['linewidth'] = 2
            axins.plot(recall, precision, **parameters)

        if enlarged_frame_xticks is not None:
            axins.set_xticks(enlarged_frame_xticks)
        if enlarged_frame_yticks is not None:
            axins.set_yticks(enlarged_frame_yticks)
        axins.grid(False)

        ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)

    plt.tight_layout()
    return fig, ax

def correlation_graph_between_prediction_and_reality(
        real:np.ndarray|List|pd.Series,
        pred:np.ndarray|List|pd.Series,
        correlation:Optional[callable]=None
) -> tuple[Figure, Axes]:
    """
    Scatter plot of true vs. predicted values with correlation coefficient.

    Generates a scatter plot to visualize the relationship between real (true) values and
    predicted values, with a diagonal reference line (y=x) and the correlation coefficient
    displayed in the top-left corner.

    Parameters
    ----------
    real : np.ndarray or List or pd.Series
        Ground truth (real) values. Will be flattened to 1D if input is multi-dimensional.
    pred : np.ndarray or List or pd.Series
        Predicted values. Must have the same length as `real`. Will be flattened to 1D if
        input is multi-dimensional.
    correlation : callable, default=None
        Function to compute the correlation coefficient. If None, uses `scipy.stats.pearsonr`.
        If a callable is provided, it must take two arrays (real, pred) as input and return a
        tuple (value, ...), where the first element is the correlation coefficient to display.
        The signature should match `scipy.stats.pearsonr`.

    Returns
    -------
    fig : Figure
        Matplotlib Figure object containing the plot.
    ax : Axes
        Matplotlib Axes object containing the plot.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from plotcraft.draw import correlation_graph_between_prediction_and_reality
    >>> real = np.random.randn(1000)
    >>> pred = real + np.random.randn(1000) * 0.5
    >>> fig, ax = correlation_graph_between_prediction_and_reality(real, pred)
    >>> plt.show()

    >>> real = np.random.randn(1000)
    >>> pred = real.copy()
    >>> fig, ax = correlation_graph_between_prediction_and_reality(real, pred)
    >>> plt.show()
    """
    fig = plt.figure(figsize=(8, 8))
    real = np.array(real).ravel()
    pred = np.array(pred).ravel()
    plt.scatter(real, pred, color='#3388dd', alpha=0.6, s=40, edgecolors='none')
    range_min = min(min(real), min(pred))
    range_max = max(max(real), max(pred))
    dis = range_max - range_min
    range_min -= dis * 0.05
    range_max += dis * 0.05
    plt.xlim(range_min, range_max)
    plt.ylim(range_min, range_max)
    plt.plot([range_min, range_max], [range_min, range_max], '--', color='grey')
    if correlation is None:
        correlation = stats.pearsonr
    r_value, _ = correlation(real, pred)
    r_text = f"R = {r_value:.2f}"
    ax = plt.gca()
    ax.text(0.02, 0.98, r_text,
            transform=ax.transAxes,
            fontsize=32, fontweight='bold', color='#bb2222',
            va='top', ha='left')
    plt.tight_layout()
    return fig,ax


def dca_curve(
        *dataframes: pd.DataFrame,
        dataframe_cols: List[str],
        thresholds: Optional[np.ndarray | List[str]] = None,
        confidence_intervals: Optional[float] = None,
        bootstraps: int = 500,
        policy: str = "opt-in",
        study_design: str = "cohort",
        population_prevalence: float | None = None,
        random_state: int = 42,
        model_names: List[str] = None,
        cost_benefit_axis: bool = True,
        colors: Optional[List[str]] = None,
) -> tuple[Figure, Axes]:
    """
    Plot Decision Curve Analysis (DCA) for one or more prediction models.

    Compute and visualize the standardized net benefit (sNB) across a range
    of risk thresholds, along with the "Treat All" and "Treat None"
    reference strategies.  Optionally adds bootstrap confidence intervals
    and a secondary cost:benefit ratio axis.

    This implementation mirrors the methodology of the R ``dcurves``
    package (``decision_curve``).

    Parameters
    ----------
    *dataframes : sequence of pandas.DataFrame
        One or more DataFrames, each containing at least the outcome column
        and the predicted probability column specified in `dataframe_cols`.
        All DataFrames must share the same column names given by
        `dataframe_cols`.  When multiple DataFrames are supplied, each is
        treated as a separate model; the "Treat All" / "Treat None"
        reference curves are drawn only from the first DataFrame.

    dataframe_cols : list of str, length = 2
        Column names to use.  ``dataframe_cols[0]`` is the binary outcome
        variable (coded 0/1) and ``dataframe_cols[1]`` is the predicted
        probability of the outcome (values in [0, 1]).

    thresholds : array-like of float, default=None
        Risk-threshold grid on which net benefit is evaluated.
        Each element must lie in [0, 1].  If None, defaults to
        ``np.arange(0.01, 1.01, 0.01)``.

    confidence_intervals : float, default=None
        If not None, a value in (0, 1) giving the confidence level for
        bootstrap confidence intervals (e.g. 0.95 for 95 % CIs).
        When None, no confidence intervals are computed.

    bootstraps : int, default=500
        Number of bootstrap resamples used to estimate confidence intervals.
        Ignored when ``confidence_intervals`` is None.

    policy : {'opt-in', 'opt-out'}, default='opt-in'
        Clinical policy direction.

        - ``'opt-in'``:  patients are treated only if their predicted risk
          exceeds the threshold (the standard DCA scenario).
        - ``'opt-out'``: patients are treated by default and only opt out
          of treatment when their predicted risk falls below the threshold.

    study_design : {'cohort', 'case-control'}, default='cohort'
        Study design from which the data originate.  When
        ``'case-control'``, the ``population_prevalence`` parameter is
        required to re-calibrate net-benefit calculations.

    population_prevalence : float or None, default=None
        Known disease prevalence in the target population.
        Required when ``study_design='case-control'``; ignored (with a
        warning) when ``study_design='cohort'``.

    random_state : int, default=42
        Seed for the random number generator used in bootstrap resampling.
        Pass an int for reproducible confidence intervals across multiple
        function calls.

    model_names : list of str or None, default=None
        Display names for each model in the legend.  Must have the same
        length as the number of DataFrames.  If None, defaults to
        ``['model 0', 'model 1', ...]``.

    cost_benefit_axis : bool, default=True
        If True, a secondary x-axis is drawn showing the cost:benefit
        ratio that corresponds to each threshold value.

    colors : list of str or None, default=None
        Matplotlib-compatible color specifications for the model curves.
        If None, the current ``axes.prop_cycle`` colors are used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object containing the DCA plot.

    ax : matplotlib.axes.Axes
        The primary Axes object of the plot, which can be used for further
        customisation.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from plotcraft.draw import dca_curve

    >>> array = np.load('./data/true_score.npy')
    >>> datas = [
    ...     pd.DataFrame(
    ...         np.array([array[i], array[i + 1]]).T,
    ...         columns=['true', 'pred'],
    ...     )
    ...     for i in range(0, array.shape[0], 2)
    ... ]

    >>> fig, ax = dca_curve(
    ...     *datas,
    ...     dataframe_cols=['true', 'pred'],
    ...     thresholds=np.arange(0.01, 0.11, 0.01),
    ... )
    >>> plt.show()

    >>> fig, ax = dca_curve(
    ...     *datas,
    ...     dataframe_cols=['true', 'pred'],
    ...     thresholds=np.arange(0.01, 1.01, 0.01),
    ... )
    >>> plt.show()

    >>> fig, ax = dca_curve(
    ...     datas[0],
    ...     dataframe_cols=['true', 'pred'],
    ...     thresholds=np.arange(0.01, 1.01, 0.01),
    ...     confidence_intervals=0.95,
    ... )
    >>> plt.show()
    """
    assert len(dataframe_cols) == 2
    real_col, score_col = dataframe_cols
    if thresholds is None:
        thresholds = np.arange(0.01, 1.01, 0.01)
    else:
        thresholds = np.array(thresholds)
        assert (0 <= thresholds).all()
        assert (thresholds <= 1).all()
    if confidence_intervals is not None:
        assert 0 < confidence_intervals < 1
    assert isinstance(bootstraps, int)
    assert policy in ("opt-in", "opt-out")
    assert study_design in ("cohort", "case-control")
    opt_in = policy == "opt-in"

    if study_design == "case-control":
        if population_prevalence is None:
            raise ValueError("In a case-control study, population prevalence needs to be provided.")
        casecontrol_rho = population_prevalence
    else:
        if population_prevalence is not None:
            warnings.warn("When study_design is set to 'cohort', the population_prevalence will be ignored.")
        casecontrol_rho = None

    if model_names is None:
        model_names = [f"model {i}" for i in range(len(dataframes))]

    rng = np.random.default_rng(random_state)

    def _calculate(real, score, thresholds, casecontrol_rho, opt_in, B_ind, confidence_intervals,
                   calculate_all_none=False):
        if not calculate_all_none:
            nb_df = calculate_nb(
                real, score, thresholds=thresholds,
                casecontrol_rho=casecontrol_rho, opt_in=opt_in,
            )
            nb_df["model"] = "pred"
            if B_ind is not None:
                alpha = 1 - confidence_intervals
                boot_snb = np.zeros((len(thresholds), bootstraps))
                boot_nb = np.zeros((len(thresholds), bootstraps))

                for b in range(bootstraps):
                    idx = B_ind[:, b]
                    real_b = real[idx]
                    score_b = score[idx]
                    try:
                        tmp = calculate_nb(
                            real_b, score_b, thresholds=thresholds,
                            casecontrol_rho=casecontrol_rho, opt_in=opt_in,
                        )
                        boot_snb[:, b] = tmp["sNB"].values
                        boot_nb[:, b] = tmp["NB"].values
                    except Exception:
                        boot_snb[:, b] = np.nan
                        boot_nb[:, b] = np.nan

                nb_df["sNB_lower"] = np.nanquantile(boot_snb, alpha / 2, axis=1)
                nb_df["sNB_upper"] = np.nanquantile(boot_snb, 1 - alpha / 2, axis=1)
                nb_df["NB_lower"] = np.nanquantile(boot_nb, alpha / 2, axis=1)
                nb_df["NB_upper"] = np.nanquantile(boot_nb, 1 - alpha / 2, axis=1)

            return nb_df
        else:
            nb_pred = _calculate(
                real, score, thresholds=thresholds,
                casecontrol_rho=casecontrol_rho, opt_in=opt_in, B_ind=B_ind, confidence_intervals=confidence_intervals
            )
            nb_pred["model"] = "pred"
            nb_all = _calculate(
                real, np.ones_like(real), thresholds=thresholds,
                casecontrol_rho=casecontrol_rho, opt_in=opt_in, B_ind=B_ind, confidence_intervals=confidence_intervals
            )
            nb_all["model"] = "All"
            nb_none = _calculate(
                real, np.zeros_like(real), thresholds=thresholds,
                casecontrol_rho=casecontrol_rho, opt_in=opt_in, B_ind=B_ind, confidence_intervals=confidence_intervals
            )
            nb_none["model"] = "None"
            return pd.concat([nb_pred, nb_all, nb_none], ignore_index=True)

    ans = []
    for i, dataframe in enumerate(dataframes):
        assert isinstance(dataframe, pd.DataFrame)
        dataframe = dataframe[dataframe_cols].copy()
        real = dataframe[real_col].values

        B_ind = None
        if confidence_intervals is not None:
            n = len(dataframe)
            B_ind = np.zeros((n, bootstraps), dtype=int)
            if study_design == "cohort":
                for b in range(bootstraps):
                    B_ind[:, b] = rng.integers(0, n, size=n)
            else:
                idx_pos = np.where(real == 1)[0]
                idx_neg = np.where(real == 0)[0]
                for b in range(bootstraps):
                    s_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
                    s_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
                    B_ind[:, b] = np.concatenate([s_pos, s_neg])

        score = dataframe[score_col].values

        if not i:
            nb_df = _calculate(
                real, score, thresholds=thresholds,
                casecontrol_rho=casecontrol_rho, opt_in=opt_in, B_ind=B_ind, confidence_intervals=confidence_intervals,
                calculate_all_none=True
            )
            nb_df["cost_benefit_ratio"] = np.tile(
                _threshold_to_cost_benefit(thresholds, policy),
                3,
            )
        else:
            nb_df = _calculate(real, score, thresholds=thresholds,
                               casecontrol_rho=casecontrol_rho, opt_in=opt_in, B_ind=B_ind,
                               confidence_intervals=confidence_intervals)
            nb_df["cost_benefit_ratio"] = np.tile(
                _threshold_to_cost_benefit(thresholds, policy),
                1,
            )
        ans.append(nb_df)

    color_map = {"All": "grey", "None": "black"}
    lw_map = {"All": 0.2, "None": 1.2}
    if colors is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    else:
        default_colors = colors
    fig, ax = plt.subplots()

    for i, df in enumerate(ans):
        if not i:
            labels = ['All', 'None', 'pred']
            for label in labels:
                sub = df[df["model"] == label].sort_values("threshold")
                t = sub["threshold"].values
                snb = sub["sNB"].values
                color = color_map.get(label, default_colors[i])
                lw = lw_map.get(label, 1.0)
                ax.plot(t, snb, color=color, lw=lw, label=model_names[i] if label == 'pred' else label)
                if confidence_intervals is not None and "sNB_lower" in sub.columns and label != "None":
                    if label == "pred":
                        ax.plot(t, sub["sNB_lower"].values, color=color, lw=0.5, linestyle="-")
                        ax.plot(t, sub["sNB_upper"].values, color=color, lw=0.5, linestyle="-")
                    else:
                        ax.plot(t, sub["sNB_lower"].values, color=color, lw=0.2, linestyle="-")
                        ax.plot(t, sub["sNB_upper"].values, color=color, lw=0.2, linestyle="-")
        else:
            labels = ['pred']
            for label in labels:
                sub = df[df["model"] == label].sort_values("threshold")
                t = sub["threshold"].values
                snb = sub["sNB"].values
                color = color_map.get(label, default_colors[i])
                lw = lw_map.get(label, 1.0)
                ax.plot(t, snb, color=color, lw=lw, label=model_names[i] if label == 'pred' else label, zorder=3)
                if confidence_intervals is not None and "sNB_lower" in sub.columns and label != "None":
                    ax.plot(t, sub["sNB_lower"].values, color=color, lw=1, linestyle="-")
                    ax.plot(t, sub["sNB_upper"].values, color=color, lw=1, linestyle="-")
    ax.set_xlim(thresholds[0] - 0.005, thresholds[-1] + 0.005)
    ax.set_ylim(-0.05, 1.0)

    x_step = round((thresholds[-1] - thresholds[0]) / 4, 2)
    x_ticks = np.arange(
        round(x_step, 2),
        thresholds[-1] + 1e-9,
        x_step,
    )
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_bounds(0, 1.0)

    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["bottom"].set_bounds(x_ticks[0], x_ticks[-1])
    ax.tick_params(axis="x", direction="out", length=5, pad=8,
                   bottom=True, top=False)
    ax.set_xlabel("High Risk Threshold", fontsize=11, labelpad=12)
    ax.set_ylabel("Standardized Net Benefit", fontsize=11)
    ax.legend(loc="upper right", frameon=True,
              framealpha=0.9, edgecolor="black", fontsize=9)

    if cost_benefit_axis:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())

        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        ax2.spines["bottom"].set_position(("outward", 70))
        ax2.spines[["top", "right", "left"]].set_visible(False)
        ax2.spines["bottom"].set_visible(True)
        ax2.tick_params(axis="x", direction="out", length=5, pad=4,
                        bottom=True, top=False)

        t_lo, t_hi = thresholds[0], thresholds[-1]

        candidate_pts = []
        candidate_labels = []

        if policy == "opt-in":
            candidate_pts.append(1 / 2)
            candidate_labels.append(f"1:1")
            for K in range(5, 500, 5):
                pt = 1.0 / (1.0 + K)
                if t_lo <= pt <= t_hi:
                    candidate_pts.append(pt)
                    candidate_labels.append(f"1:{K}")
            for K in range(5, 500, 5):
                pt = K / (1.0 + K)
                if t_lo <= pt <= t_hi:
                    candidate_pts.append(pt)
                    candidate_labels.append(f"{K}:1")
        else:
            candidate_pts.append(1 / 2)
            candidate_labels.append(f"1:1")
            for K in range(5, 500, 5):
                pt = K / (1.0 + K)
                if t_lo <= pt <= t_hi:
                    candidate_pts.append(pt)
                    candidate_labels.append(f"1:{K}")
            for K in range(5, 500, 5):
                pt = 1.0 / (1.0 + K)
                if t_lo <= pt <= t_hi:
                    candidate_pts.append(pt)
                    candidate_labels.append(f"{K}:1")

        if candidate_pts:
            order = np.argsort(candidate_pts)
            all_pts = np.array(candidate_pts)[order]
            all_labels = [candidate_labels[i] for i in order]
        else:
            all_pts = np.array([])
            all_labels = []

        if len(all_pts) <= 5:
            sel_pts = all_pts
            sel_labels = all_labels
        else:
            left_t, right_t = all_pts[-1], all_pts[0]
            ideal_mid = np.linspace(left_t, right_t, 5)[1:-1]
            mid_pts = all_pts[1:-1]
            mid_labels_pool = all_labels[1:-1]
            chosen_idx = []
            for target in ideal_mid:
                dists = np.abs(mid_pts - target)
                for ci in chosen_idx:
                    dists[ci] = np.inf
                chosen_idx.append(int(np.argmin(dists)))
            chosen_idx.sort()
            sel_pts = np.concatenate([
                [all_pts[-1]],
                mid_pts[chosen_idx],
                [all_pts[0]],
            ])
            sel_labels = (
                    [all_labels[-1]]
                    + [mid_labels_pool[ci] for ci in chosen_idx]
                    + [all_labels[0]]
            )

        ax2.set_xticks(sel_pts)
        ax2.set_xticklabels(sel_labels, fontsize=9)
        ax2.spines["bottom"].set_bounds(sel_pts.min(), sel_pts.max())
        ax2.set_xlabel("Cost:Benefit Ratio", fontsize=11, labelpad=10)

    plt.tight_layout()

    return fig, ax

def calibration_curve(
        real:np.ndarray | List[int] | pd.DataFrame,
        pred:Optional[np.ndarray | List[int|float] | pd.DataFrame]=None,
        logit_vals:Optional[np.ndarray | List[int|float]| pd.DataFrame]=None,
        logistic_cal:bool=True,
        nonparametric_cal:bool=True,
        legendloc:Optional[str | tuple[float | int] | Literal[False]]=None,
        statloc:Optional[tuple[float | int] | Literal[False]]=None,
        riskdist:Literal["predicted","calibrated"]="predicted",
        cex:float | int=0.7,
) -> tuple[Figure, Axes, dict[str, np.ndarray]]:
    r"""
    Validate predicted probabilities against binary outcomes with a calibration plot.

    Python implementation of Frank Harrell's ``val.prob`` in the R ``rms``
    package.  Computes discrimination and calibration statistics for a set
    of predicted probabilities and observed binary outcomes, and produces a
    calibration plot that includes a logistic recalibration curve, a lowess
    nonparametric smooth, and a spike histogram of the risk distribution.

    The calibration plot overlays three references:

    - **Ideal line** (the 45° diagonal): perfect calibration.
    - **Logistic calibration curve**: a two-parameter (intercept + slope)
      logistic recalibration of the original predictions.  An intercept of
      0 and slope of 1 indicates no systematic miscalibration.
    - **Nonparametric (lowess) curve**: a flexible smooth that reveals
      local departures from ideal calibration.

    Parameters
    ----------
    real : array-like of shape (n_samples,)
        Observed binary outcomes.  Each element must be 0 or 1.

    pred : array-like of shape (n_samples,), default=None
        Predicted probabilities, each in the interval [0, 1].
        Exactly one of ``pred`` and ``logit_vals`` must be provided.
        If ``pred`` is given, logit values are derived internally via
        the logit (log-odds) transform.

    logit_vals : array-like of shape (n_samples,), default=None
        Predicted values on the logit (log-odds) scale.  If provided
        instead of ``pred``, predicted probabilities are obtained via
        the inverse-logit (expit) transform.  Exactly one of ``pred``
        and ``logit_vals`` must be provided.

    logistic_cal : bool, default=True
        If ``True``, plot the two-parameter logistic recalibration curve

    logistic_cal : bool, default=True
        If ``True``, plot the two-parameter Nonparametric curve

    legendloc : str, tuple of float, or False, default=None
        Location of the legend on the calibration plot.  Accepts any
        value valid for :func:`matplotlib.axes.Axes.legend` ``loc``
        parameter.  If ``None``, the legend is placed automatically
        (``loc="best"``).  Set to ``False`` to suppress the legend
        entirely.

    statloc : tuple of float (x, y) or False, default=None
        Coordinates (in data space) for the top-left corner of the
        statistics text block.  If ``None``, the text is placed at the
        upper-left of the axes in axes-fraction coordinates (0.02, 0.98).
        Set to ``False`` to suppress the statistics text entirely.

    riskdist : {"predicted", "calibrated"}, default="predicted"
        Which risk distribution to display as a spike histogram along
        the bottom of the calibration plot.

        - ``"predicted"`` : use the original predicted probabilities.
        - ``"calibrated"`` : use probabilities recalibrated through the
          fitted logistic model.

    cex : float, default=0.7
        Character expansion factor that controls the font size of the
        legend and statistics text.  The effective font size is
        ``cex * 10`` points, consistent with the R ``cex`` convention.

    Returns
    -------
    fig : matplotlib.figure.Figure

    ax : matplotlib.axes.Axes

    stats : dict
        A dictionary of discrimination and calibration statistics with
        the following keys:

        ============  =====================================================
        Key           Description
        ============  =====================================================
        Dxy           Somers' Dxy rank correlation between
                      predicted probabilities and observed outcomes
                      (\(= 2C - 1\)).
        C (ROC)       Concordance statistic (area under the ROC curve).
        R2            Nagelkerke–Cox–Snell R2 index.
        D             Discrimination index, \((LR_{\chi^{2}} - 1) / n\),
                      where \(LR_{\chi^{2}}\) is the likelihood-ratio
                      \(\chi^{2}\) statistic comparing the model to a
                      null (intercept-only) model.
        D:Chi-sq      Likelihood-ratio \(\chi^{2}\) statistic.
        D:p           p-value for the likelihood-ratio test.
        U             Unreliability index,
                      \((U_{\chi^{2}} - 2) / n\).
        U:Chi-sq      Unreliability \(\chi^{2}\) statistic (deviance
                      difference between original predictions and the
                      logistic recalibration model).
        U:p           p-value for the unreliability test.
        Q             Quality index, D - U.
        Brier         Brier score, \(\frac{1}{n}\sum(y_i - \hat{p}_i)^{2}\).
        Intercept     Intercept alpha of the logistic recalibration.
                      Ideally 0.
        Slope         Slope beta of the logistic recalibration.
                      Ideally 1.
        Emax          Maximum absolute calibration error (from the lowess
                      smooth).
        E90           90th-percentile absolute calibration error.
        Eavg          Mean absolute calibration error.
        S:z           Spiegelhalter's z-statistic for testing
                      calibration-in-the-large.
        S:p           Two-sided p-value for Spiegelhalter's test.
        ============  =====================================================

    Raises
    ------
    ValueError
        If neither ``pred`` nor ``logit_vals`` is provided, or if the
        lengths of ``pred`` (or ``logit_vals``) and ``real`` differ.

    Examples
    --------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> df = pd.read_csv("all.csv")
    >>> fig, ax, result = val_prob(df["true"], df["pred"])
    >>> plt.show()
    >>> print(result)
    """

    def _logistic_fit(X, y):
        k = X.shape[1]
        beta0 = np.zeros(k)

        def neg_ll(beta):
            eta = X @ beta
            ll = np.sum(y * eta - np.logaddexp(0, eta))
            return -ll

        def grad(beta):
            eta = X @ beta
            pr = plogis(eta)
            return -X.T @ (y - pr)

        res = minimize(neg_ll, beta0, jac=grad, method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-12})
        coef = res.x
        deviance = 2 * res.fun
        return coef, deviance

    def _roc_auc(y, p):
        y = np.asarray(y, dtype=int)
        p = np.asarray(p, dtype=float)
        pos = p[y == 1]
        neg = p[y == 0]
        n1, n0 = len(pos), len(neg)
        if n1 == 0 or n0 == 0:
            return 0.5
        all_vals = np.concatenate([pos, neg])
        order = np.argsort(all_vals)
        sorted_vals = all_vals[order]
        ranks = np.empty(len(all_vals))
        i = 0
        while i < len(sorted_vals):
            j = i
            while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            ranks[i:j] = avg_rank
            i = j
        inv_order = np.argsort(order)
        ranks_orig = ranks[inv_order]
        auc = (np.sum(ranks_orig[:n1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
        return auc

    def _lowess(x, y, frac=2.0 / 3.0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        k = int(np.ceil(frac * n))
        order = np.argsort(x)
        x_s = x[order]
        y_s = y[order]
        y_hat = np.zeros(n)

        for i in range(n):
            dists = np.abs(x_s - x_s[i])
            idx = np.argsort(dists)[:k]
            max_dist = dists[idx[-1]]
            if max_dist == 0:
                max_dist = 1.0
            u = dists[idx] / max_dist
            w = np.maximum((1 - u ** 3) ** 3, 0)

            xw = x_s[idx]
            yw = y_s[idx]
            sw = np.sum(w)
            if sw == 0:
                y_hat[i] = np.mean(yw)
                continue
            mx = np.sum(w * xw) / sw
            my = np.sum(w * yw) / sw
            ss_xx = np.sum(w * (xw - mx) ** 2)
            if ss_xx == 0:
                y_hat[i] = my
            else:
                slope = np.sum(w * (xw - mx) * (yw - my)) / ss_xx
                y_hat[i] = my + slope * (x_s[i] - mx)

        return x_s, y_hat

    if pred is None and logit_vals is None:
        raise ValueError("Either pred or logit_vals must be provided.")

    if pred is not None:
        pred = np.asarray(pred, dtype=float).ravel()
        logit_vals = qlogis(pred)
    else:
        logit_vals = np.asarray(logit_vals, dtype=float).ravel()
        pred = plogis(logit_vals)

    real = np.asarray(real, dtype=float).ravel()
    if len(pred) != len(real):
        raise ValueError("The lengths of pred (or logit) and real are not consistent.")

    assert riskdist in ("predicted", "calibrated")

    def _spi(pv, yv):
        z = np.sum((yv - pv) * (1 - 2 * pv)) / np.sqrt(np.sum((1 - 2 * pv) ** 2 * pv * (1 - pv)))
        pval = 2 * norm.sf(np.abs(z))
        return z, pval

    nma = ~(np.isnan(pred) | np.isnan(real))
    logit_vals = logit_vals[nma]
    real = real[nma]
    pred = pred[nma]
    n = len(real)

    if len(np.unique(pred)) == 1:
        P = np.mean(real)
        Intc = qlogis(P)
        D = -1.0 / n
        L01 = -2.0 * np.nansum(real * logit_vals - np.logaddexp(0, logit_vals))
        L_cal = -2.0 * np.nansum(real * Intc - np.logaddexp(0, Intc))
        U_chisq = L01 - L_cal
        U_p = 1 - chi2.cdf(U_chisq, 1)
        U = (U_chisq - 1) / n
        Q = D - U
        spi_z, spi_p = _spi(pred, real)
        return {
            "Dxy": 0.0, "C (ROC)": 0.5, "R2": 0.0, "D": D,
            "D:Chi-sq": 0.0, "D:p": 1.0, "U": U, "U:Chi-sq": U_chisq,
            "U:p": U_p, "Q": Q, "Brier": np.mean((real - pred[0]) ** 2),
            "Intercept": Intc, "Slope": 0.0,
            "Emax": 0.0, "E90": 0.0, "Eavg": abs(pred[0] - P),
            "S:z": spi_z, "S:p": spi_p,
        }

    finite_mask = np.isfinite(logit_vals)
    nm = int(np.sum(~finite_mask))
    if nm > 0:
        warnings.warn(
            f"{nm} observations were excluded from the logistic calibration "
            "because they had a probability of either 0 or 1."
        )

    logit_f = logit_vals[finite_mask]
    y_f = real[finite_mask]
    p_f = pred[finite_mask]
    n_f = len(y_f)

    X_recal = np.column_stack([np.ones_like(logit_f), logit_f])
    recal_coef, recal_deviance = _logistic_fit(X_recal, y_f)
    recal_intercept, recal_slope = recal_coef[0], recal_coef[1]

    C = _roc_auc(y_f, p_f)
    Dxy = 2 * (C - 0.5)

    p_bar_f = np.mean(y_f)
    L0_f = -2.0 * np.sum(
        y_f * np.log(np.maximum(p_bar_f, 1e-15))
        + (1 - y_f) * np.log(np.maximum(1 - p_bar_f, 1e-15))
    )
    L1_f = -2.0 * np.sum(y_f * logit_f - np.logaddexp(0, logit_f))

    lr = L0_f - L1_f
    p_lr = 1 - chi2.cdf(lr, 1) if lr > 0 else 1.0

    L0_ll_f = -L0_f / 2.0
    R2_num = 1 - np.exp(-lr / n_f)
    R2_den = 1 - np.exp(2 * L0_ll_f / n_f)
    R2 = R2_num / R2_den if R2_den != 0 else 0.0

    D = (lr - 1) / n_f

    L01 = -2.0 * np.nansum(real * logit_vals - np.logaddexp(0, logit_vals))
    U_chisq = L01 - recal_deviance
    p_U = 1 - chi2.cdf(U_chisq, 2) if U_chisq > 0 else 1.0
    U = (U_chisq - 2) / n_f
    Q = D - U

    B = np.mean((pred - real) ** 2)

    sm_x, sm_y = _lowess(pred, real, frac=2.0 / 3.0)
    cal_smooth = np.interp(pred, sm_x, sm_y)
    er = np.abs(pred - cal_smooth)
    eavg = np.mean(er)
    emax = np.max(er)
    e90 = float(np.percentile(er, 90))

    spi_z, spi_p = _spi(pred, real)

    stats = {
        "Dxy": Dxy, "C (ROC)": C, "R2": R2, "D": D,
        "D:Chi-sq": lr, "D:p": p_lr, "U": U, "U:Chi-sq": U_chisq,
        "U:p": p_U, "Q": Q, "Brier": B,
        "Intercept": recal_intercept, "Slope": recal_slope,
        "Emax": emax, "E90": e90, "Eavg": eavg,
        "S:z": spi_z, "S:p": spi_p,
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Probability")
    ax.set_aspect("equal")

    ax.plot((0, 1), (0, 1), linewidth=6, color="0.85", label="Ideal", zorder=1)

    if logistic_cal:
        logit_seq = np.linspace(-7, 7, 200)
        prob_seq = plogis(logit_seq)
        pred_prob = plogis(recal_intercept + recal_slope * logit_seq)
        ax.plot(prob_seq, pred_prob, "k-", linewidth=1,
                label="Logistic calibration", zorder=3)
    if nonparametric_cal:
        ax.plot(sm_x, sm_y, "k:", linewidth=1,
                label="Nonparametric", zorder=2)

    if legendloc is not False:
        if legendloc is None:
            ax.legend(fontsize=cex * 10, frameon=False, loc="best")
        else:
            ax.legend(fontsize=cex * 10, frameon=False, loc=legendloc)

    if statloc is not False:
        dostats_keys = [
            "Dxy", "C (ROC)", "R2", "D", "U", "Q",
            "Brier", "Intercept", "Slope", "Emax", "E90", "Eavg",
            "S:z", "S:p",
        ]
        stat_lines = "\n".join(
            f"{k:<12s} {stats[k]:>.3f}" if stats[k] > 0 else f"{k:<12s}{stats[k]:>.3f}" for k in dostats_keys
        )
        if statloc is None:
            ax.text(0.02, 0.98, stat_lines,
                    transform=ax.transAxes,
                    fontsize=cex * 10, family="monospace",
                    verticalalignment="top")
        else:
            ax.text(statloc[0], statloc[1], stat_lines,
                    fontsize=cex * 10, family="monospace",
                    verticalalignment="top")

    if riskdist:
        if riskdist == "calibrated":
            x_dist = plogis(recal_intercept + recal_slope * logit_vals)
            x_dist = np.where(pred == 0, 0.0, x_dist)
            x_dist = np.where(pred == 1, 1.0, x_dist)
        else:
            x_dist = pred.copy()

        bins = np.linspace(0, 1, 101)
        x_in = x_dist[(x_dist >= 0) & (x_dist <= 1)]
        hist_vals, bin_edges = np.histogram(x_in, bins=bins)
        j = hist_vals > 0
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        max_h = hist_vals.max() if hist_vals.max() > 0 else 1
        spike_h = 0.15 * hist_vals[j] / max_h
        ax.vlines(bin_centers[j], 0, spike_h,
                  linewidth=0.5, color="black", zorder=1)

    plt.tight_layout()

    return fig, ax, stats


def calibration_curves(
        *dataframes: pd.DataFrame,
        dataframe_cols: List[str],
        logistic_cal: bool = True,
        nonparametric_cal: bool = True,
        legendloc: Optional[str | tuple[float | int] | Literal[False]] = None,
        model_names: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        cex: float | int = 0.7,
) -> tuple[Figure, Axes, List[dict[str, float]]]:
    r"""
    Plot calibration curves for one or more prediction models.

    Extends the single-model ``calibration_curve`` (Frank Harrell's
    ``val.prob`` methodology) to support multiple models on a single plot.
    Each model is drawn in a distinct colour, with its logistic
    recalibration curve and lowess nonparametric smooth sharing the same
    colour.  The 45° ideal line is rendered as a thin dashed line.

    Parameters
    ----------
    *dataframes : sequence of pandas.DataFrame
        One or more DataFrames, each containing at least the outcome column
        and the predicted-probability column specified in ``dataframe_cols``.
        All DataFrames must share the same column names given by
        ``dataframe_cols``.  Each DataFrame is treated as a separate model.

    dataframe_cols : list of str, length = 2
        Column names to use.  ``dataframe_cols[0]`` is the binary outcome
        variable (coded 0/1) and ``dataframe_cols[1]`` is the predicted
        probability of the outcome (values in [0, 1]).

    logistic_cal : bool, default=True
        If ``True``, plot the two-parameter logistic recalibration curve
        for every model.

    nonparametric_cal : bool, default=True
        If ``True``, plot the lowess nonparametric calibration curve for
        every model.

    legendloc : str, tuple of float, or False, default=None
        Location of the legend.  Accepts any value valid for
        :func:`matplotlib.axes.Axes.legend` ``loc`` parameter.
        If ``None``, the legend is placed automatically (``"best"``).
        Set to ``False`` to suppress the legend entirely.

    model_names : list of str or None, default=None
        Display names for each model in the legend.  Must have the same
        length as the number of DataFrames.  If ``None``, defaults to
        ``['Model 0', 'Model 1', ...]``.

    colors : list of str or None, default=None
        Matplotlib-compatible colour specifications for the model curves.
        If ``None``, the current ``axes.prop_cycle`` colours are used.
        For a given model, the logistic recalibration curve and the
        nonparametric smooth share the same colour.

    cex : float, default=0.7
        Character expansion factor controlling the font size of the
        legend.  The effective font size is ``cex * 10`` points.

    Returns
    -------
    fig : matplotlib.figure.Figure

    ax : matplotlib.axes.Axes

    all_stats : list of dict
        A list of dictionaries (one per model), each containing the
        following discrimination and calibration statistics:

        ============  =====================================================
        Key           Description
        ============  =====================================================
        Dxy           Somers' Dxy rank correlation
                      (\(= 2C - 1\)).
        C (ROC)       Concordance statistic (AUC-ROC).
        R2            Nagelkerke–Cox–Snell \(R^{2}\).
        D             Discrimination index,
                      \((LR_{\chi^{2}} - 1) / n\).
        D:Chi-sq      Likelihood-ratio \(\chi^{2}\) statistic.
        D:p           p-value for the likelihood-ratio test.
        U             Unreliability index,
                      \((U_{\chi^{2}} - 2) / n\).
        U:Chi-sq      Unreliability \(\chi^{2}\) statistic.
        U:p           p-value for the unreliability test.
        Q             Quality index, \(D - U\).
        Brier         Brier score,
                      \(\frac{1}{n}\sum(y_i - \hat{p}_i)^{2}\).
        Intercept     Logistic recalibration intercept (ideally 0).
        Slope         Logistic recalibration slope (ideally 1).
        Emax          Maximum absolute calibration error.
        E90           90th-percentile absolute calibration error.
        Eavg          Mean absolute calibration error.
        S:z           Spiegelhalter's z-statistic.
        S:p           Two-sided p-value for Spiegelhalter's test.
        ============  =====================================================

    Raises
    ------
    ValueError
        If ``dataframe_cols`` does not have exactly two elements, or if the
        number of ``model_names`` does not match the number of DataFrames.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pandas as pd
    >>> from plotcraft.draw import calibration_curves
    >>> array = np.load('./data/true_score.npy')
    >>> datas = [pd.DataFrame(np.array([array[i],array[i+1]]).T,columns=['true','pred']) for i in range(0,array.shape[0],2) if i != 14]
    >>> fig, ax, all_stats = calibration_curves(*datas,dataframe_cols=['true','pred'])
    >>> plt.show()
    >>> print(all_stats)
    >>> fig, ax, all_stats = calibration_curves(*datas, dataframe_cols=['true', 'pred'], logistic_cal=False)
    >>> plt.show()
    >>> fig, ax, all_stats = calibration_curves(*datas, dataframe_cols=['true', 'pred'], nonparametric_cal=False)
    >>> plt.show()
    """

    if len(dataframe_cols) != 2:
        raise ValueError("dataframe_cols must have exactly 2 elements: [outcome_col, pred_col].")
    real_col, pred_col = dataframe_cols

    if len(dataframes) == 0:
        raise ValueError("At least one DataFrame must be provided.")

    if model_names is None:
        model_names = [f"Model {i}" for i in range(len(dataframes))]
    elif len(model_names) != len(dataframes):
        raise ValueError(
            f"Length of model_names ({len(model_names)}) does not match "
            f"the number of DataFrames ({len(dataframes)})."
        )

    if colors is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    else:
        default_colors = colors

    def _logistic_fit(X, y):
        k = X.shape[1]
        beta0 = np.zeros(k)

        def neg_ll(beta):
            eta = X @ beta
            return -np.sum(y * eta - np.logaddexp(0, eta))

        def grad(beta):
            eta = X @ beta
            pr = plogis(eta)
            return -X.T @ (y - pr)

        res = minimize(neg_ll, beta0, jac=grad, method="L-BFGS-B",
                       options={"maxiter": 200, "ftol": 1e-12})
        return res.x, 2 * res.fun

    def _roc_auc(y, p):
        y = np.asarray(y, dtype=int)
        p = np.asarray(p, dtype=float)
        pos, neg = p[y == 1], p[y == 0]
        n1, n0 = len(pos), len(neg)
        if n1 == 0 or n0 == 0:
            return 0.5
        all_vals = np.concatenate([pos, neg])
        order = np.argsort(all_vals)
        sorted_vals = all_vals[order]
        ranks = np.empty(len(all_vals))
        i = 0
        while i < len(sorted_vals):
            j = i
            while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
                j += 1
            ranks[i:j] = (i + 1 + j) / 2.0
            i = j
        ranks_orig = ranks[np.argsort(order)]
        return (np.sum(ranks_orig[:n1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0)

    def _lowess(x, y, frac=2.0 / 3.0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        k = int(np.ceil(frac * n))
        order = np.argsort(x)
        x_s, y_s = x[order], y[order]
        y_hat = np.zeros(n)
        for i in range(n):
            dists = np.abs(x_s - x_s[i])
            idx = np.argsort(dists)[:k]
            max_dist = max(dists[idx[-1]], 1.0) if dists[idx[-1]] == 0 else dists[idx[-1]]
            u = dists[idx] / max_dist
            w = np.maximum((1 - u ** 3) ** 3, 0)
            xw, yw = x_s[idx], y_s[idx]
            sw = np.sum(w)
            if sw == 0:
                y_hat[i] = np.mean(yw)
                continue
            mx = np.sum(w * xw) / sw
            my = np.sum(w * yw) / sw
            ss_xx = np.sum(w * (xw - mx) ** 2)
            if ss_xx == 0:
                y_hat[i] = my
            else:
                slope = np.sum(w * (xw - mx) * (yw - my)) / ss_xx
                y_hat[i] = my + slope * (x_s[i] - mx)
        return x_s, y_hat

    def _spi(pv, yv):
        z = np.sum((yv - pv) * (1 - 2 * pv)) / np.sqrt(
            np.sum((1 - 2 * pv) ** 2 * pv * (1 - pv))
        )
        return z, 2 * norm.sf(np.abs(z))

    def _compute_stats(real, pred):
        """Compute all calibration / discrimination statistics for one model."""
        logit_vals = qlogis(pred)

        nma = ~(np.isnan(pred) | np.isnan(real))
        logit_vals = logit_vals[nma]
        real = real[nma]
        pred = pred[nma]
        n = len(real)

        if len(np.unique(pred)) == 1:
            P = np.mean(real)
            Intc = qlogis(P)
            D_val = -1.0 / n
            L01 = -2.0 * np.nansum(real * logit_vals - np.logaddexp(0, logit_vals))
            L_cal = -2.0 * np.nansum(real * Intc - np.logaddexp(0, Intc))
            U_chisq = L01 - L_cal
            U_p = 1 - chi2.cdf(U_chisq, 1)
            U_val = (U_chisq - 1) / n
            spi_z, spi_p = _spi(pred, real)
            return (
                {
                    "Dxy": 0.0, "C (ROC)": 0.5, "R2": 0.0, "D": D_val,
                    "D:Chi-sq": 0.0, "D:p": 1.0,
                    "U": U_val, "U:Chi-sq": U_chisq, "U:p": U_p,
                    "Q": D_val - U_val,
                    "Brier": np.mean((real - pred[0]) ** 2),
                    "Intercept": Intc, "Slope": 0.0,
                    "Emax": 0.0, "E90": 0.0, "Eavg": abs(pred[0] - P),
                    "S:z": spi_z, "S:p": spi_p,
                },
                None,
                None,
                None,
                None,
            )

        finite_mask = np.isfinite(logit_vals)
        nm = int(np.sum(~finite_mask))
        if nm > 0:
            warnings.warn(
                f"{nm} observations excluded from logistic calibration "
                "(probability of exactly 0 or 1)."
            )

        logit_f = logit_vals[finite_mask]
        y_f = real[finite_mask]
        p_f = pred[finite_mask]
        n_f = len(y_f)

        X_recal = np.column_stack([np.ones_like(logit_f), logit_f])
        recal_coef, recal_deviance = _logistic_fit(X_recal, y_f)
        recal_intercept, recal_slope = recal_coef[0], recal_coef[1]

        C = _roc_auc(y_f, p_f)
        Dxy = 2 * (C - 0.5)

        p_bar_f = np.mean(y_f)
        L0_f = -2.0 * np.sum(
            y_f * np.log(np.maximum(p_bar_f, 1e-15))
            + (1 - y_f) * np.log(np.maximum(1 - p_bar_f, 1e-15))
        )
        L1_f = -2.0 * np.sum(y_f * logit_f - np.logaddexp(0, logit_f))

        lr = L0_f - L1_f
        p_lr = 1 - chi2.cdf(lr, 1) if lr > 0 else 1.0
        L0_ll_f = -L0_f / 2.0
        R2_num = 1 - np.exp(-lr / n_f)
        R2_den = 1 - np.exp(2 * L0_ll_f / n_f)
        R2 = R2_num / R2_den if R2_den != 0 else 0.0
        D_val = (lr - 1) / n_f

        L01 = -2.0 * np.nansum(real * logit_vals - np.logaddexp(0, logit_vals))
        U_chisq = L01 - recal_deviance
        p_U = 1 - chi2.cdf(U_chisq, 2) if U_chisq > 0 else 1.0
        U_val = (U_chisq - 2) / n_f
        Q = D_val - U_val
        B = np.mean((pred - real) ** 2)

        sm_x, sm_y = _lowess(pred, real, frac=2.0 / 3.0)
        cal_smooth = np.interp(pred, sm_x, sm_y)
        er = np.abs(pred - cal_smooth)

        spi_z, spi_p = _spi(pred, real)

        stats = {
            "Dxy": Dxy, "C (ROC)": C, "R2": R2, "D": D_val,
            "D:Chi-sq": lr, "D:p": p_lr,
            "U": U_val, "U:Chi-sq": U_chisq, "U:p": p_U,
            "Q": Q, "Brier": B,
            "Intercept": recal_intercept, "Slope": recal_slope,
            "Emax": np.max(er), "E90": float(np.percentile(er, 90)),
            "Eavg": np.mean(er),
            "S:z": spi_z, "S:p": spi_p,
        }
        return stats, recal_intercept, recal_slope, sm_x, sm_y

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Probability")
    ax.set_aspect("equal")

    ax.plot((0, 1), (0, 1), linestyle="--", linewidth=1,
            color="grey", label="Ideal", zorder=1)

    all_stats: List[dict] = []

    for idx, df in enumerate(dataframes):
        assert isinstance(df, pd.DataFrame), f"Argument {idx} is not a DataFrame."
        real_arr = np.asarray(df[real_col], dtype=float).ravel()
        pred_arr = np.asarray(df[pred_col], dtype=float).ravel()

        pred_arr = np.clip(pred_arr, 1e-15, 1 - 1e-15)

        color = default_colors[idx % len(default_colors)]
        name = model_names[idx]

        stats, r_int, r_slope, sm_x, sm_y = _compute_stats(real_arr, pred_arr)
        all_stats.append(stats)

        if logistic_cal and r_int is not None:
            logit_seq = np.linspace(-7, 7, 200)
            prob_seq = plogis(logit_seq)
            pred_prob = plogis(r_int + r_slope * logit_seq)
            ax.plot(prob_seq, pred_prob, "-", color=color, linewidth=1,
                    label=f"{name} — Logistic", zorder=3)

        if nonparametric_cal and sm_x is not None:
            ax.plot(sm_x, sm_y, ":", color=color, linewidth=1,
                    label=f"{name} — Nonparametric", zorder=2)

    if legendloc is not False:
        loc = "best" if legendloc is None else legendloc
        handles, labels = [], []

        handles.append(Line2D([0], [0], linestyle="--", linewidth=1, color="grey"))
        labels.append("Ideal")

        if logistic_cal:
            handles.append(Line2D([0], [0], linestyle="-", linewidth=1, color="black"))
            labels.append("Logistic")
        if nonparametric_cal:
            handles.append(Line2D([0], [0], linestyle=":", linewidth=1, color="black"))
            labels.append("Nonparametric")

        for idx_m, name in enumerate(model_names):
            c = default_colors[idx_m % len(default_colors)]
            handles.append(Line2D([0], [0], linestyle="-", linewidth=2, color=c))
            labels.append(name)

        ax.legend(handles, labels, fontsize=cex * 10, frameon=False, loc=loc)

    plt.tight_layout()
    return fig, ax, all_stats


def plot_function_with_asymptote(
        f: sp.Expr,
        var: sp.Symbol,
        x_range: tuple[float | int, float | int] = (-10, 10),
        n_points: int = 1000,
        curve_color: Optional[str] = None,
        asymptote_color: Optional[str] = None,
        add_asymptote: Optional[List[Callable]] = None,
        verbose:bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot a SymPy symbolic function with automatic asymptote detection.

    Automatically computes the continuous domain, detects vertical /
    horizontal / oblique asymptotes, and renders the curve with proper
    discontinuity handling (jump detection & segment splitting).

    Parameters
    ----------
    f : sympy.Expr
        The symbolic expression to plot, e.g. ``sin(x) / x``.

    var : sympy.Symbol, default=x
        The free variable in ``f``.

    x_range : tuple of (float, float), default=(-10, 10)
        The horizontal viewing window ``(x_min, x_max)``.

    n_points : int, default=1000
        Total number of sample points distributed across all valid
        intervals.  More points yield a smoother curve at the cost of
        longer computation.

    curve_color : str or None, default=None
        Matplotlib color string for the main curve.  When ``None``,
        defaults to ``"red"``.

    asymptote_color : str or None, default=None
        Matplotlib color string for every asymptote line (both
        auto-detected and manually supplied).  When ``None``, defaults
        to ``"grey"``.

    add_asymptote : list of callable or None, default=None
        Extra asymptote curves that cannot be found automatically.
        Each element must be a callable with signature
        ``func(x_array) -> y_array`` (NumPy-compatible).  The function
        is allowed to return ``np.inf`` / ``np.nan`` at isolated
        points; those points are filtered out and the line is split
        into continuous segments automatically.

    verbose: bool, default=True
        if verbose is true, print some details

    Returns
    -------
    fig : matplotlib.figure.Figure

    ax : matplotlib.axes.Axes

    Notes
    -----
    - The continuous domain is computed via
      ``sympy.calculus.util.continuous_domain``.
    - Vertical asymptotes are located by combining three strategies:
      excluded points of the domain, ``sympy.singularities``, and
      roots of the denominator after ``cancel``.  Candidates are then
      verified numerically.
    - Horizontal / oblique asymptotes are found by evaluating
      ``lim_{x->±∞} f/x`` and ``lim_{x->±∞} (f - kx)``.
    - Large jumps between consecutive sample points are detected with
      a median-based threshold so that branches separated by a
      vertical asymptote are never connected by a spurious line.

    Examples
    --------
    >>> from sympy import symbols, tan, sqrt, sin
    >>> x = symbols("x")

    Plot ``tan(x)`` with automatically detected vertical asymptotes:

    >>> f = tan(x)
    >>> plot_function(f, x, x_range=(-10, 10), n_points=1000)

    Plot ``sqrt(x² - 1)`` whose domain excludes ``(-1, 1)``:

    >>> f = sqrt(x**2 - 1)
    >>> plot_function(f, x, x_range=(-10, 10), n_points=1000)

    Plot ``sin(x)/x`` and manually add the curvilinear asymptote
    ``y = 1/x`` (which has its own singularity at ``x = 0``):

    >>> f = sin(x) / x
    >>> plot_function(
    ...     f, x,
    ...     x_range=(-10, 10),
    ...     n_points=1000,
    ...     add_asymptote=[lambda t: 1 / t],
    ... )
    """
    def enumerate_set_points(s, x_lo, x_hi, max_n=200):
        pts = []
        if isinstance(s, sp.FiniteSet):
            for p in s:
                try:
                    v = float(p)
                    if x_lo - 0.01 <= v <= x_hi + 0.01:
                        pts.append(v)
                except (TypeError, ValueError):
                    pass
        elif isinstance(s, sp.Union):
            for part in s.args:
                pts.extend(enumerate_set_points(part, x_lo, x_hi, max_n))
        elif isinstance(s, sp.ImageSet):
            lam = s.lamda
            n_var = lam.variables[0]
            expr = lam.expr
            for i in range(-max_n, max_n + 1):
                try:
                    v = float(expr.subs(n_var, i))
                    if x_lo - 0.01 <= v <= x_hi + 0.01:
                        pts.append(v)
                except (TypeError, ValueError, OverflowError):
                    pass
        elif isinstance(s, sp.Complement):
            pts.extend(enumerate_set_points(s.args[1], x_lo, x_hi, max_n))
        return sorted(set(round(p, 12) for p in pts))

    def get_excluded_points(domain, x_lo, x_hi):
        pts = []
        if isinstance(domain, sp.Complement):
            pts.extend(enumerate_set_points(domain.args[1], x_lo, x_hi))
        elif isinstance(domain, sp.Union):
            for part in domain.args:
                pts.extend(get_excluded_points(part, x_lo, x_hi))
        return sorted(set(pts))

    def domain_to_intervals(domain, x_lo, x_hi):
        view = sp.Interval(x_lo, x_hi)
        clipped = domain.intersect(view)
        return _set_to_intervals(clipped, x_lo, x_hi)

    def _set_to_intervals(s, x_lo, x_hi):
        if s == sp.S.EmptySet:
            return []
        if isinstance(s, sp.Interval):
            lo = max(float(s.start) if s.start != -sp.oo else x_lo, x_lo)
            hi = min(float(s.end) if s.end != sp.oo else x_hi, x_hi)
            return [(lo, hi)] if lo < hi else []
        if isinstance(s, sp.Union):
            intervals = []
            for part in s.args:
                intervals.extend(_set_to_intervals(part, x_lo, x_hi))
            return sorted(intervals)
        if isinstance(s, sp.FiniteSet):
            return []
        if isinstance(s, sp.Complement):
            base_intervals = _set_to_intervals(s.args[0], x_lo, x_hi)
            excluded_pts = enumerate_set_points(s.args[1], x_lo, x_hi)
            if not excluded_pts:
                return base_intervals
            result = []
            for lo, hi in base_intervals:
                cuts = sorted([p for p in excluded_pts if lo < p < hi])
                bounds = [lo] + cuts + [hi]
                for i in range(len(bounds) - 1):
                    a, b = bounds[i], bounds[i + 1]
                    if b - a > 1e-12:
                        result.append((a, b))
            return result
        return [(x_lo, x_hi)]

    def find_vertical_asymptotes(f, var, x_range=(-10, 10), domain=None):
        x_lo, x_hi = x_range
        candidates = set()

        if domain is not None:
            excluded = get_excluded_points(domain, x_lo, x_hi)
            candidates.update(excluded)

        try:
            from sympy import singularities as _sing
            sing = _sing(f, var)
            pts = enumerate_set_points(sing, x_lo, x_hi)
            candidates.update(pts)
        except Exception:
            pass

        try:
            d = sp.denom(sp.cancel(f))
            for c in sp.solve(d, var):
                try:
                    if c.is_real:
                        cv = float(c)
                        if x_lo <= cv <= x_hi:
                            candidates.add(cv)
                except (TypeError, ValueError):
                    pass
        except Exception:
            pass

        if not candidates:
            return []

        f_np = sp.lambdify(var, f, modules=["numpy"])
        v_asym = []
        for c in candidates:
            eps = 1e-8
            try:
                with np.errstate(all="ignore"):
                    y_left = abs(float(f_np(c - eps)))
                    y_right = abs(float(f_np(c + eps)))
                if y_left > 1e6 or y_right > 1e6:
                    v_asym.append(c)
            except Exception:
                v_asym.append(c)

        return sorted(set(round(v, 10) for v in v_asym))

    def find_oblique_asymptotes(f, var):
        results = []
        for direction, inf_val in [("+inf", sp.oo), ("-inf", -sp.oo)]:
            try:
                k = sp.limit(f / var, var, inf_val)
                if k in (sp.oo, -sp.oo, sp.zoo, sp.nan) or not k.is_finite:
                    continue
                b = sp.limit(f - k * var, var, inf_val)
                if b in (sp.oo, -sp.oo, sp.zoo, sp.nan) or not b.is_finite:
                    continue
                results.append((float(k), float(b), direction))
            except Exception:
                pass
        if (len(results) == 2
                and np.isclose(results[0][0], results[1][0])
                and np.isclose(results[0][1], results[1][1])):
            return [(results[0][0], results[0][1], "both")]
        return results

    def sample_segments(f, intervals, var, n=1000, v_asym=None):
        if v_asym is None:
            v_asym = []

        va_set = set(round(v, 10) for v in v_asym)

        fine = []
        for lo, hi in intervals:
            cuts = sorted([a for a in v_asym if lo + 1e-9 < a < hi - 1e-9])
            bounds = [lo] + cuts + [hi]
            for i in range(len(bounds) - 1):
                a, b = bounds[i], bounds[i + 1]
                eps = (b - a) * 0.002
                a_adj = a + eps if round(a, 10) in va_set else a
                b_adj = b - eps if round(b, 10) in va_set else b
                if b_adj - a_adj > 1e-12:
                    fine.append((a_adj, b_adj))

        total_len = sum(hi - lo for lo, hi in fine)
        if total_len == 0:
            return []

        f_np = sp.lambdify(var, f, modules=["numpy"])
        segments = []

        for lo, hi in fine:
            seg_n = max(6, int(n * (hi - lo) / total_len))
            seg_x = np.linspace(lo, hi, seg_n)
            with np.errstate(all="ignore"):
                seg_y = np.asarray(f_np(seg_x), dtype=float)
            mask = np.isfinite(seg_y)
            if mask.any():
                sx, sy = seg_x[mask], seg_y[mask]
                if len(sy) > 1:
                    segments.extend(_split_at_jumps(sx, sy))
                else:
                    segments.append((sx, sy))
        return segments

    def _split_at_jumps(xs, ys, factor=50):
        diffs = np.abs(np.diff(ys))
        median_diff = np.median(diffs) if len(diffs) > 0 else 1
        threshold = max(median_diff * factor, 5)
        jumps = np.where(diffs > threshold)[0]
        segs = []
        prev = 0
        for j in jumps:
            if j + 1 - prev >= 2:
                segs.append((xs[prev:j + 1], ys[prev:j + 1]))
            prev = j + 1
        if len(xs) - prev >= 2:
            segs.append((xs[prev:], ys[prev:]))
        return segs if segs else [(xs, ys)]


    domain = sp.calculus.util.continuous_domain(f, var, sp.S.Reals)

    intervals = domain_to_intervals(domain, x_range[0], x_range[1])

    texts = [f"function: f(x) = {f}", f"domain of definition: {domain}", f"Drawing interval({len(intervals)} segment): {intervals[:8]}{'...' if len(intervals) > 8 else ''}"]


    v_asym = find_vertical_asymptotes(f, var, x_range, domain=domain)
    o_asym = find_oblique_asymptotes(f, var)

    if v_asym:
        labels = [f"x={round(a, 4)}" for a in v_asym]
        texts.append(f"vertical asymptotes (with a total of {len(v_asym)} lines): {labels[:10]}{'...' if len(labels) > 10 else ''}")
    else:
        texts.append("vertical asymptotes: None")

    for k, b, d in o_asym:
        kind = "horizontal" if k == 0 else "oblique"
        sign = "+" if b >= 0 else "-"
        eq = f"y = {k}x {sign} {abs(b)}" if k != 0 else f"y = {b}"
        texts.append(f"{kind}asymptotes: {eq}  (direction: {d})")

    segments = sample_segments(f, intervals, var, n_points, v_asym)
    if not segments:
        texts.append("No valid points could be sampled within the given range.")
        return None, None

    all_y = np.concatenate([s[1] for s in segments])
    y_lo, y_hi = np.percentile(all_y, 2), np.percentile(all_y, 98)
    y_margin = (y_hi - y_lo) * 0.3 + 0.5
    y_view = (y_lo - y_margin, y_hi + y_margin)

    fig, ax = plt.subplots(figsize=(10, 6))
    if curve_color is None:
        curve_color = "red"
    if asymptote_color is None:
        asymptote_color = "grey"

    for i, (sx, sy) in enumerate(segments):
        ax.plot(sx, sy, color=curve_color, linewidth=2,
                label="f(x)" if i == 0 else None)

    for i, va in enumerate(v_asym):
        ax.axvline(va, color=asymptote_color, linestyle="--", linewidth=1, alpha=0.6)

    for i, (k, b, d) in enumerate(o_asym):
        xl = np.linspace(x_range[0], x_range[1], 300)
        yl = k * xl + b
        ax.plot(xl, yl, color=asymptote_color, linestyle="-.", linewidth=1.2,
                alpha=0.8)

    if add_asymptote:
        xl = np.linspace(x_range[0], x_range[1], 300)
        for i, func in enumerate(add_asymptote):
            try:
                with np.errstate(all="ignore"):
                    yl = np.asarray(func(xl), dtype=float)
                mask = np.isfinite(yl)
                if not mask.any():
                    texts.append(f"additional asymptote {i + 1}: there are no valid points within the given range, so skipping.")
                    continue
                seg_indices = np.split(np.where(mask)[0],
                                       np.where(np.diff(np.where(mask)[0]) > 1)[0] + 1)
                for idx_chunk in seg_indices:
                    if len(idx_chunk) < 2:
                        continue
                    sx, sy = xl[idx_chunk], yl[idx_chunk]
                    sub_segs = _split_at_jumps(sx, sy)
                    for ssx, ssy in sub_segs:
                        ax.plot(ssx, ssy, color=asymptote_color, linestyle="-.", linewidth=1.2,
                                alpha=0.8)
            except Exception as e:
                texts.append(f"An error occurred when drawing the supplementary asymptote {i + 1}: {e}")

    if verbose:
        print('\n'.join(texts))

    ax.set_xlim(x_range)
    ax.set_ylim(y_view)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.4)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)

    asymptote_handle = Line2D([], [], color=asymptote_color, linestyle="--",
                              linewidth=1, alpha=0.6, label="asymptote")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(asymptote_handle)
    labels.append("asymptote")
    ax.legend(handles=handles, labels=labels, loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

if __name__ == '__main__':
    array = np.load('./data/true_score.npy')
    fig, ax, result = calibration_curve(array[0], array[1])
    plt.show()
    print(result)