import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

colorblind_codes = {
    "orange": "#f28e2b",
    "red": "#e15759",
    "green": "#59a14f",
    "blue": "#4e79a7",
    "grey": "#bab0ac"
}

def gradient_cmap(end_color="#000000", final_c = (1, 1, 1), r=False):
    colors = [final_c, end_color] if not r else [end_color, final_c]
    positions = [0, 1]
    custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(positions, colors)))
    return custom_cmap

def two_sided_gradient_cmap(end_color="#000000", r=False):
    colors = [(1, 1, 1), end_color, (1, 1, 1)] if not r else [end_color, (1, 1, 1) , end_color]
    positions = [0, 0.5, 1]
    custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(positions, colors)))
    return custom_cmap


figsizes = {
    "neurips": {"full": (5.5, 1.7), "half": (2.25, 2.1)},
    "ICML": {"full": (6.5, 1.7), "half": (2.25, 2.1)},
    "ICLR": {"full": (5.5, 1.7), "half": (2.25, 2.1)},
    "a1poster": {"full": (9.5, 3.2), "half": (4.25, 3.2)},
    "a0poster": {"full": (9.5, 3.2), "half": (4.25, 3.2)},
    "keynote": {"full": (12.5, 5.2), "half": (4.25, 3.2)},
}

sns_contexts = {
    "neurips": {
        "font_scale": 0.85,
        "context": "paper",
        "style": "whitegrid",
        "palette": "colorblind",
        "font": "serif"
    },
    "ICLR": {
        "font_scale": 0.92,
        "context": "paper",
        "style": "whitegrid",
        "palette": "colorblind",
        "font": "serif"
    },
    "ICML": {
        "font_scale": 0.92,
        "context": "paper",
        "style": "whitegrid",
        "palette": "colorblind",
        "font": "serif"
    },
    "a1poster": {
        "font_scale": 0.75,
        "context": "talk",
        "style": "whitegrid",
        "palette": "colorblind",
        "font": "serif"
    },
    "a0poster": {
        "font_scale": 0.9,
        "context": "talk",
        "style": "whitegrid",
        "palette": "colorblind",
        "font": "serif"
    },
    "keynote": {
        "font_scale": 1.4,
        "context": "talk",
        "style": "whitegrid",
        "palette": "colorblind",
        "font": "serif"
    },
}

def make_full_page_figure(style, n_rows=1, n_cols=1, relative_figsize=(1, 1), **kwargs):
    sns.set_theme(**sns_contexts[style])
    figsize = figsizes[style]["full"]
    figsize = (figsize[0] * relative_figsize[0], figsize[1] * relative_figsize[1])
    f, axs = plt.subplots(n_rows, n_cols, figsize=figsize, **kwargs)
    return f, axs

def plot_boxplot_comparison(results, nominal_coverage=None): 
    box_data = []
    box_labels = []
    box_colors = []
    approach_colors = {
        "BPS": colorblind_codes["green"],
        "APS": colorblind_codes["orange"],            
        "BPS_nom": colorblind_codes["blue"],
        "APS_nom": colorblind_codes["red"],}

    for approach in results.keys():
        box_data.append(results[approach])
        box_colors.append(approach_colors[approach])
        box_labels.append(f"{approach}")
    make_full_page_figure(style="neurips")
    bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True, showmeans=True)
    # Apply colors
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    for median in bp['medians']:
        # median.set_color('black')
        median.set_visible(False)
    # Apply color to mean markers
    for mean in bp['means']:
        mean.set_markerfacecolor('black')
        mean.set_markeredgecolor('black')
        mean.set_marker('_')  
    if nominal_coverage is not None:
        plt.axhline(nominal_coverage, color=colorblind_codes["grey"], linestyle='--', label=f"Nominal ({nominal_coverage})")
    plt.xticks(rotation=90)
    plt.grid(axis="y")

def plot_boxplot_comparison_ax(results, ax, nominal_coverage=None): 
    box_data = []
    box_labels = []
    box_colors = []
    approach_colors = {
        "BPS": colorblind_codes["green"],
        "APS": colorblind_codes["orange"],            
        "BPS_nom": colorblind_codes["blue"],
        "APS_nom": colorblind_codes["red"],}

    for approach in results.keys():
        box_data.append(results[approach])
        box_colors.append(approach_colors[approach])
        box_labels.append(f"{approach}")
    bp = ax.boxplot(box_data, patch_artist=True, showmeans=True)
    # Apply colors
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    for median in bp['medians']:
        # median.set_color('black')
        median.set_visible(False)
    # Apply color to mean markers
    for mean in bp['means']:
        mean.set_markerfacecolor('black')
        mean.set_markeredgecolor('black')
        mean.set_marker('_')  
    if nominal_coverage is not None:
        ax.axhline(nominal_coverage, color=colorblind_codes["grey"], linestyle='--', label=f"Nominal ({nominal_coverage})")
    ax.set_xticklabels(box_labels, rotation=90)
