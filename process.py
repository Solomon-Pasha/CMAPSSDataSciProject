import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

relevantColumns = ["sensor measurement3", "sensor measurement17", "sensor measurement2", "sensor measurement21",
"sensor measurement20", "sensor measurement15", "sensor measurement8", "sensor measurement13", "sensor measurement7", "sensor measurement4",
"sensor measurement12", "sensor measurement11", "sensor measurement14", "sensor measurement9"]

def remainingLife(data : pd.DataFrame):
    retVal = data.copy()
    engines = data.groupby("unit number")
    max_cycle = engines["time in cycles"].transform("max")      # each row gets its engine's max
    retVal["RUL"] = max_cycle - data["time in cycles"]
    return retVal

# def displayRemainingVsVars(data: pd.DataFrame, engines : list[int]):
#      data["unit number"].isin(engines)
#      fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 12), sharex=True)
#      sensors = data[relevantColumns]
#      for combination in zip(sensors, axes):
#          col, ax = combination
#          ax.plot()
         
         
def displayRemainingVsVars(
    data: pd.DataFrame,
    engines: list[int],
    sensor_cols: list[str] | None = relevantColumns,
    labels: dict[str, str] | None = None,
    smooth: int | None = 10,
    ncols: int = 4,
):
    """Plot each sensor against cycles remaining (RUL) for a few engines.
 
    data        - training data with "unit number", "RUL" and the sensor columns
    engines     - engine IDs to overlay, e.g. [1, 2, 3, 4, 5]
    sensor_cols - the sensors to plot (your 14 kept sensors)
    labels      - optional {column: readable name} for chart titles
    smooth      - rolling-average window in cycles; None to show raw data only
    """
    labels = labels or {}
 
    # Step 1/2: keep only the chosen engines' rows
    rows = data[data["unit number"].isin(engines)]
 
    # Step 3: one small chart per sensor, sharing the x-axis
    nrows = math.ceil(len(sensor_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True)
    axes = axes.flat
 
    for sensor, ax in zip(sensor_cols, axes):
        # Step 5: one line per engine
        for eng, g in rows.groupby("unit number"):
            if smooth:
                # Step 6: faint raw line, bold smoothed line in the same colour
                raw = ax.plot(g["RUL"], g[sensor], alpha=0.25, linewidth=0.7)
                ax.plot(g["RUL"], g[sensor].rolling(smooth).mean(),
                        color=raw[0].get_color(), linewidth=1.4, label=f"Engine {eng}")
            else:
                ax.plot(g["RUL"], g[sensor], alpha=0.7, linewidth=0.8, label=f"Engine {eng}")
 
        # Step 7: labels
        ax.set_title(labels.get(sensor, sensor), fontsize=10)
        ax.tick_params(labelsize=8)
 
    # Step 4: time runs left to right towards failure (RUL = 0 on the right).
    # The x-axis is shared, so inverting one inverts them all.
    axes_list = fig.axes
    axes_list[0].invert_xaxis()
 
    # Hide unused grid spaces (e.g. 14 sensors in a 4x4 grid leaves 2 spare)
    for ax in axes_list[len(sensor_cols):]:
        ax.set_visible(False)
 
    # Label the x-axis on the lowest visible chart in each column only.
    # With sharex=True, matplotlib hides tick labels except on the bottom row,
    # so re-enable them on charts sitting above a hidden space.
    n = len(sensor_cols)
    for i, ax in enumerate(axes_list[:n]):
        if i >= n - ncols:
            ax.set_xlabel("Cycles remaining", fontsize=9)
            ax.tick_params(labelbottom=True)
 
    # One shared legend instead of one per chart
    handles, names = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower right", fontsize=9)
 
    title = "Sensor trends towards failure"
    if smooth:
        title += f" ({smooth}-cycle rolling average over raw data)"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
 
    # Step 8: return the figure; display or save it in the notebook
    return fig

def getCorrelationHeatmap(data: pd.DataFrame):
    corr = data[relevantColumns].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    names = [c.replace("sensor measurement", "s") for c in corr.columns]
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(names)), names)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Correlation between kept sensors")
    return fig

def getStrongestCorrelations(data: pd.DataFrame):
    corr = data[relevantColumns].corr()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)    # upper triangle, excluding the diagonal
    pairs = corr.where(mask).stack()                        # one row per unique pair
    return pairs.abs().sort_values(ascending=False).head(10)


def lastCycle(data: pd.DataFrame, cap = 125):
    temp = data.groupby("unit number").tail(1).copy()
    temp["RUL_true_capped"] = data["RUL_true"].clip(upper=cap)
    temp["error"] = (temp["RUL_pred"] - temp["RUL_true"])
    return temp
    
def plotLastCyclePredictions(results: dict, cap: int = 125, ncols: int = 2):
    """Predicted vs true RUL at each test engine's last cycle, one chart per model.

    Points above the diagonal are over-predictions: the model thinks the engine
    has more life left than it really does, which is the costly mistake.
    """
    rmse = lambda e: np.sqrt((e ** 2).mean())
    n = len(results)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                            sharex=True, sharey=True, squeeze=False)
    axes = axes.flat

    for (name, df), ax in zip(results.items(), axes):
        last = lastCycle(df, cap)
        over = last["error"] > 0

        # Shade the over-prediction region and draw the "perfect prediction" line
        ax.fill_between([0, cap], [0, cap], cap * 1.15, color="tab:red", alpha=0.06)
        ax.plot([0, cap], [0, cap], color="grey", linestyle="--", linewidth=1)

        ax.scatter(last.loc[~over, "RUL_true"], last.loc[~over, "RUL_pred"],
                s=18, color="tab:blue", label="Under-prediction (safe side)")
        ax.scatter(last.loc[over, "RUL_true"], last.loc[over, "RUL_pred"],
                s=18, color="tab:red", label="Over-prediction (risky)")

        ax.set_title(f"{name}\nRMSE {rmse(last['error']):.1f} cycles, "
                    f"{over.mean():.0%} over-predicted", fontsize=10)
        ax.set_xlim(0, cap * 1.05)
        ax.set_ylim(0, cap * 1.15)
        ax.set_aspect("equal")

    for ax in list(axes)[n:]:
        ax.set_visible(False)

    for ax in fig.axes:
        if ax.get_visible():
            ax.set_xlabel(f"True RUL (capped at {cap})", fontsize=9)
            ax.set_ylabel("Predicted RUL", fontsize=9)
            ax.tick_params(labelbottom=True, labelleft=True)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Predicted vs true remaining life at each test engine's last cycle", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig