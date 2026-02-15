import os
import matplotlib

matplotlib.use("Agg")  # fixes Mac backend issues

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

# Global plot defaults (modern + readable)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (9, 5.5),
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# Modern palette (clean, consistent)
PALETTE = {
    "blue": "#4E79A7",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
    "teal": "#76B7B2",
    "purple": "#B07AA1",
    "gray": "#9D9D9D",
}

TABLE_CLASSES = "table table-striped table-hover table-bordered align-middle"


class AnalysisService:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        os.makedirs("static/plots", exist_ok=True)

        self.questions = {
            1: self._question_1,
            2: self._question_2,
            3: self._question_3,
            4: self._question_4,
            5: self._question_5,
        }

    def _new_plot(self):
        plt.clf()

    def _finish_plot(self, title: str, xlabel: str = "", ylabel: str = ""):
        plt.title(title, pad=10)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        ax = plt.gca()
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.grid(axis="x", visible=False)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()

    def _label_bars(self, labels=None):
        ax = plt.gca()
        for container in ax.containers:
            if labels is None:
                ax.bar_label(container, fmt="%.0f", padding=3)
            else:
                ax.bar_label(container, labels=labels, padding=3)

    def run_question(self, question_id: int):
        func = self.questions.get(question_id)
        if not func:
            return "Error", "Question not found", ""
        return func()

    def _question_1(self):
        title = "Overall Survival Rate"

        # Text result
        survival_rate = self.df["survived"].mean() * 100
        result_text = f"Overall Survival Rate: {survival_rate:.2f}%"

        # Plot
        self._new_plot()
        counts = self.df["survived"].value_counts().sort_index()
        ax = counts.plot(kind="bar", color=[PALETTE["red"], PALETTE["green"]])

        ax.set_xticklabels(["No (0)", "Yes (1)"], rotation=0)
        self._label_bars()
        self._finish_plot("Survival Count", ylabel="Passengers")

        filename = "survival_overall.png"
        plt.savefig(f"static/plots/{filename}", bbox_inches="tight")

        return title, result_text, filename

    def _question_2(self):
        title = "Survival Rate by Sex"

        # Table result
        result_df = self.df.groupby("sex")["survived"].mean().reset_index()
        result_df["survived"] = (result_df["survived"] * 100).round(2).astype(str) + "%"

        # Plot
        self._new_plot()
        plot_data = self.df.groupby("sex")["survived"].mean().sort_index()
        ax = plot_data.plot(kind="bar", color=[PALETTE["blue"], PALETTE["orange"]])

        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0, 1)
        ax.set_xticklabels([t.get_text().capitalize() for t in ax.get_xticklabels()], rotation=0)

        percent_labels = [f"{v*100:.1f}%" for v in plot_data.values]
        self._label_bars(labels=percent_labels)

        self._finish_plot("Survival Rate by Sex", ylabel="Survival Rate")

        filename = "survival_by_sex.png"
        plt.savefig(f"static/plots/{filename}", bbox_inches="tight")

        return title, result_df.to_html(index=False, classes=TABLE_CLASSES), filename

    def _question_3(self):
        title = "Survival Rate by Class"

        # Prefer "class" column (First/Second/Third). If missing, fallback to pclass.
        group_col = "class" if "class" in self.df.columns else "pclass"

        result_df = self.df.groupby(group_col)["survived"].mean().reset_index()
        result_df["survived"] = (result_df["survived"] * 100).round(2).astype(str) + "%"

        # Plot
        self._new_plot()
        plot_data = self.df.groupby(group_col)["survived"].mean()

        if group_col == "class":
            order = ["First", "Second", "Third"]
            plot_data = plot_data.reindex([x for x in order if x in plot_data.index])
            colors = [PALETTE["purple"], PALETTE["teal"], PALETTE["orange"]]
        else:
            plot_data = plot_data.sort_index()
            colors = [PALETTE["purple"], PALETTE["teal"], PALETTE["orange"]][:len(plot_data)]

        ax = plot_data.plot(kind="bar", color=colors)

        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0, 1)
        ax.set_xticklabels([str(t.get_text()) for t in ax.get_xticklabels()], rotation=0)

        percent_labels = [f"{v*100:.1f}%" for v in plot_data.values]
        self._label_bars(labels=percent_labels)

        self._finish_plot("Survival Rate by Class", ylabel="Survival Rate")

        filename = "survival_by_class.png"
        plt.savefig(f"static/plots/{filename}", bbox_inches="tight")

        return title, result_df.to_html(index=False, classes=TABLE_CLASSES), filename

    def _question_4(self):
        title = "Age Distribution"

        age_col = self.df["age"]
        missing_count = int(age_col.isna().sum())
        mean_age = float(age_col.mean())

        # Plot
        self._new_plot()
        plt.hist(age_col.dropna(), bins=22, color=PALETTE["blue"], alpha=0.85)

        self._finish_plot("Age Distribution", xlabel="Age", ylabel="Passengers")

        filename = "age_distribution.png"
        plt.savefig(f"static/plots/{filename}", bbox_inches="tight")

        result_text = (
            f"The mean age is {mean_age:.2f} years. "
            f"There are {missing_count} missing values."
        )
        return title, result_text, filename

    def _question_5(self):
        title = "Passengers by Embarkation Port"

        counts_df = self.df["embarked"].value_counts().reset_index()
        counts_df.columns = ["Port", "Passenger Count"]
        counts_df = counts_df.sort_values("Passenger Count", ascending=False)

        # Plot
        self._new_plot()
        plt.bar(counts_df["Port"], counts_df["Passenger Count"], color=PALETTE["teal"])
        self._label_bars()
        self._finish_plot("Passengers by Embarkation Port", xlabel="Port (C, Q, S)", ylabel="Passengers")

        filename = "embarked_counts.png"
        plt.savefig(f"static/plots/{filename}", bbox_inches="tight")

        return title, counts_df.to_html(index=False, classes=TABLE_CLASSES), filename
