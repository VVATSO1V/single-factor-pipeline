"""Generate an HTML single-factor analysis report from pipeline CSV outputs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - fallback for partially installed envs
    sns = None


REQUIRED_FILES = [
    "summary.csv",
    "ic_series.csv",
    "quantile_daily_returns.csv",
    "quantile_summary.csv",
    "long_short_nav.csv",
    "turnover.csv",
    "yearly_performance.csv",
    "data_quality.csv",
    "run_config.json",
]

INTEGER_COLUMNS = {
    "year",
    "return_window",
    "n_dates",
    "factor_rows",
    "factor_dates",
    "factor_stocks",
    "market_panel_rows",
    "market_panel_dates",
    "market_panel_stocks",
    "eligible_rows",
    "excluded_not_in_universe",
    "excluded_st",
    "excluded_suspended",
    "excluded_listing_days",
    "excluded_open_limit",
    "ic_days",
    "long_count",
    "short_count",
    "quantiles",
    "min_listing_days",
}
INTEGER_PARAMETERS = {"quantiles", "min_listing_days"}
PALETTE = {
    "raw": "#D55E00",
    "neutralized": "#0072B2",
}


def _set_plot_style() -> None:
    if sns is not None:
        sns.set_theme(
            style="whitegrid",
            context="talk",
            font="Microsoft YaHei",
            rc={
                "axes.edgecolor": "#CBD5E1",
                "axes.linewidth": 0.8,
                "grid.color": "#E5E7EB",
                "grid.linewidth": 0.8,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
            },
        )
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def _read_csv(output_dir: Path, filename: str) -> pd.DataFrame:
    path = output_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"missing pipeline output file: {path}")
    return pd.read_csv(path)


def _read_config(output_dir: Path) -> dict:
    path = output_dir / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing pipeline output file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _is_integer_like(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    return False


def _format_value(value: object, column: str | None = None) -> str:
    if isinstance(value, (list, tuple)):
        return html.escape(", ".join(str(item) for item in value))
    if isinstance(value, dict):
        return html.escape(json.dumps(value, ensure_ascii=False))
    if pd.isna(value):
        return ""
    if column in INTEGER_COLUMNS and _is_integer_like(value):
        return f"{int(value):,}"
    if isinstance(value, (int, float)):
        if _is_integer_like(value) and column in INTEGER_COLUMNS:
            return f"{int(value):,}"
        return f"{value:,.4f}"
    return html.escape(str(value))


def _format_config_value(parameter: object, value: object) -> str:
    parameter_name = str(parameter)
    if parameter_name in INTEGER_PARAMETERS and _is_integer_like(value):
        return f"{int(value):,}"
    if parameter_name == "mad_width" and isinstance(value, (int, float)):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return html.escape(", ".join(str(item) for item in value))
    if isinstance(value, dict):
        return html.escape(json.dumps(value, ensure_ascii=False))
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return html.escape(str(value))


def _frame_to_html(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    if columns is not None:
        available = [column for column in columns if column in frame.columns]
        frame = frame[available]
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(lambda value: _format_value(value, column))
    return display.to_html(index=False, escape=False, classes="report-table")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _variant_color(variant: str) -> str:
    return PALETTE.get(variant, "#4B5563")


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.3f}") -> None:
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            labels.append("" if pd.isna(value) else fmt.format(value))
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)


def _plot_ic_series(ic_series: pd.DataFrame, figures_dir: Path) -> None:
    data = ic_series.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["return_window"] == 1]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for variant, group in data.groupby("variant", sort=True):
        group = group.sort_values("date")
        color = _variant_color(variant)
        ax.plot(
            group["date"],
            group["rank_ic"],
            color=color,
            alpha=0.12,
            linewidth=0.7,
            label=f"{variant} daily",
        )
        ax.plot(
            group["date"],
            group["rank_ic"].rolling(60, min_periods=10).mean(),
            color=color,
            linewidth=2.2,
            label=f"{variant} 60D mean",
        )
    ax.axhline(0, color="#64748B", linewidth=0.9, linestyle="--")
    ax.set_title("1D Rank IC Series with 60D Rolling Mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rank IC")
    ax.legend(ncol=2, fontsize=9)
    _save_figure(fig, figures_dir / "ic_series.png")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for variant, group in data.groupby("variant", sort=True):
        group = group.sort_values("date")
        ax.plot(
            group["date"],
            group["rank_ic"].cumsum(),
            color=_variant_color(variant),
            label=variant,
            linewidth=2.0,
        )
    ax.axhline(0, color="#64748B", linewidth=0.9, linestyle="--")
    ax.set_title("Cumulative 1D Rank IC")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Rank IC")
    ax.legend()
    _save_figure(fig, figures_dir / "cumulative_ic.png")


def _plot_quantile_returns(
    quantile_daily: pd.DataFrame,
    figures_dir: Path,
) -> None:
    data = quantile_daily[quantile_daily["return_window"] == 1]
    grouped = (
        data.groupby(["variant", "quantile"], as_index=False)["mean_return"].mean()
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for variant, group in grouped.groupby("variant", sort=True):
        ax.plot(
            group["quantile"],
            group["mean_return"],
            marker="o",
            color=_variant_color(variant),
            linewidth=2.2,
            label=variant,
        )
    ax.axhline(0, color="#64748B", linewidth=0.9, linestyle="--")
    ax.set_title("Average 1D Return by Quantile")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Average Return")
    ax.legend()
    _save_figure(fig, figures_dir / "quantile_returns.png")


def _plot_long_short_nav(long_short_nav: pd.DataFrame, figures_dir: Path) -> None:
    data = long_short_nav.copy()
    data["date"] = pd.to_datetime(data["date"])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for variant, group in data.groupby("variant", sort=True):
        ax.plot(
            group["date"],
            group["long_short_nav"],
            color=_variant_color(variant),
            label=variant,
            linewidth=2.0,
        )
    ax.set_title("Long-Short NAV")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend()
    _save_figure(fig, figures_dir / "long_short_nav.png")


def _plot_turnover(turnover: pd.DataFrame, figures_dir: Path) -> None:
    data = turnover.copy()
    data["date"] = pd.to_datetime(data["date"])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for variant, group in data.groupby("variant", sort=True):
        group = group.sort_values("date")
        color = _variant_color(variant)
        ax.plot(
            group["date"],
            group["factor_rank_turnover"],
            color=color,
            alpha=0.14,
            linewidth=0.7,
            label=f"{variant} daily",
        )
        ax.plot(
            group["date"],
            group["factor_rank_turnover"].rolling(60, min_periods=10).mean(),
            color=color,
            linewidth=2.2,
            label=f"{variant} 60D mean",
        )
    ax.set_title("Factor Rank Turnover with 60D Rolling Mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")
    ax.legend(ncol=2, fontsize=9)
    _save_figure(fig, figures_dir / "turnover.png")


def _plot_yearly_performance(yearly: pd.DataFrame, figures_dir: Path) -> None:
    data = yearly[yearly["return_window"] == 1].copy()
    data["year"] = data["year"].astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    pivot_ic = data.pivot(index="year", columns="variant", values="rank_ic_mean")
    pivot_ret = data.pivot(index="year", columns="variant", values="long_short_return")
    colors = [_variant_color(column) for column in pivot_ic.columns]
    pivot_ic.plot(kind="bar", ax=axes[0], color=colors, width=0.78)
    axes[0].set_title("Yearly 1D Rank IC Mean")
    axes[0].set_ylabel("Rank IC")
    _annotate_bars(axes[0], "{:.3f}")
    colors = [_variant_color(column) for column in pivot_ret.columns]
    pivot_ret.plot(kind="bar", ax=axes[1], color=colors, width=0.78)
    axes[1].set_title("Yearly Long-Short Return")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Return")
    _annotate_bars(axes[1], "{:.2f}")
    _save_figure(fig, figures_dir / "yearly_performance.png")


def _generate_figures(outputs: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    _set_plot_style()
    _plot_ic_series(outputs["ic_series"], figures_dir)
    _plot_quantile_returns(outputs["quantile_daily_returns"], figures_dir)
    _plot_long_short_nav(outputs["long_short_nav"], figures_dir)
    _plot_turnover(outputs["turnover"], figures_dir)
    _plot_yearly_performance(outputs["yearly_performance"], figures_dir)


def _figure_tag(filename: str, caption: str) -> str:
    escaped_caption = html.escape(caption)
    return (
        f'<figure class="figure-block"><img src="figures/{filename}" alt="{escaped_caption}">'
        f"<figcaption>{escaped_caption}</figcaption></figure>"
    )


def _summary_sentence(summary: pd.DataFrame, config: dict) -> str:
    factor_name = html.escape(str(config.get("factor_name", "unknown factor")))
    raw_1d = summary[
        (summary["variant"] == "raw") & (summary["return_window"] == 1)
    ]
    if raw_1d.empty:
        return f"本报告汇总因子 {factor_name} 的标准单因子测试结果。"
    row = raw_1d.iloc[0]
    return (
        f"本报告汇总因子 {factor_name} 的标准单因子测试结果。"
        f"raw 版本 1 日 Rank IC 均值为 {_format_value(row['rank_ic_mean'])}，"
        f"ICIR 为 {_format_value(row['rank_icir'])}，"
        f"IC 为正比例为 {_format_value(row['ic_positive_ratio'])}。"
    )


def _build_html(
    outputs: dict[str, pd.DataFrame],
    config: dict,
) -> str:
    css = """
body { font-family: Arial, "Microsoft YaHei", sans-serif; margin: 32px; color: #1f2933; line-height: 1.55; }
h1 { margin-bottom: 8px; }
h2 { border-bottom: 1px solid #d8dee4; padding-bottom: 6px; margin-top: 32px; }
.summary-box { background: #f6f8fa; border: 1px solid #d8dee4; padding: 16px; border-radius: 6px; }
.report-table { border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 13px; }
.report-table th, .report-table td { border: 1px solid #d8dee4; padding: 6px 8px; text-align: right; }
.report-table th:first-child, .report-table td:first-child { text-align: left; }
.report-table th { background: #f6f8fa; }
.figure-block { margin: 22px auto 30px; text-align: center; }
.figure-block img { display: block; max-width: min(100%, 1120px); margin: 0 auto; border: 1px solid #d8dee4; border-radius: 6px; background: white; }
.figure-block figcaption { color: #57606a; font-size: 13px; margin-top: 8px; text-align: left; max-width: 1120px; margin-left: auto; margin-right: auto; }
code { background: #f6f8fa; padding: 1px 4px; border-radius: 3px; }
"""
    config_table = pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in config.items()]
    )
    config_table["value"] = config_table.apply(
        lambda row: _format_config_value(row["parameter"], row["value"]),
        axis=1,
    )
    turnover = outputs["turnover"].groupby("variant", as_index=False).agg(
        factor_rank_turnover=("factor_rank_turnover", "mean"),
        long_turnover=("long_turnover", "mean"),
        short_turnover=("short_turnover", "mean"),
    )
    final_nav = (
        outputs["long_short_nav"]
        .sort_values("date")
        .groupby("variant", as_index=False)
        .tail(1)
    )
    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>单因子分析报告</title>
  <style>{css}</style>
</head>
<body>
  <h1>单因子分析报告</h1>
  <div class="summary-box">{_summary_sentence(outputs["summary"], config)}</div>

  <h2>1. 运行参数</h2>
  {_frame_to_html(config_table)}

  <h2>2. 数据质量与样本过滤</h2>
  {_frame_to_html(outputs["data_quality"])}

  <h2>3. IC 分析</h2>
  {_frame_to_html(outputs["summary"], ["variant", "return_window", "rank_ic_mean", "rank_ic_std", "rank_icir", "rank_icir_annualized", "pearson_ic_mean", "pearson_icir", "ic_positive_ratio", "n_dates"])}
  {_figure_tag("ic_series.png", "1 日 Rank IC 时间序列")}
  {_figure_tag("cumulative_ic.png", "1 日累计 Rank IC")}

  <h2>4. 分组收益分析</h2>
  {_frame_to_html(outputs["quantile_summary"], ["variant", "return_window", "monotonicity", "top_group_return", "bottom_group_return", "top_bottom_spread"])}
  {_figure_tag("quantile_returns.png", "1 日各分组平均收益")}

  <h2>5. 多空组合净值</h2>
  {_frame_to_html(final_nav, ["variant", "date", "long_nav", "short_nav", "long_short_nav"])}
  {_figure_tag("long_short_nav.png", "多空组合净值曲线")}
  <p>注：当前多空净值为诊断型结果，不包含交易成本、滑点、冲击成本和融券约束。</p>

  <h2>6. 换手率分析</h2>
  {_frame_to_html(turnover)}
  {_figure_tag("turnover.png", "因子排序换手率")}

  <h2>7. 分年度表现与衰减</h2>
  {_frame_to_html(outputs["yearly_performance"], ["year", "variant", "return_window", "rank_ic_mean", "rank_icir", "ic_positive_ratio", "long_short_return", "long_short_sharpe", "long_short_max_drawdown", "long_short_win_rate", "factor_rank_turnover", "long_turnover", "short_turnover"])}
  {_figure_tag("yearly_performance.png", "年度 IC 与多空收益")}

  <h2>8. 结论提示</h2>
  <p>阅读报告时建议同时关注 IC 均值、ICIR、分组单调性、多空净值、换手率和年度表现。若中性化后仍然有效，说明因子可能具有更独立的选股信息；若换手率过高，后续需要加入交易成本进一步检验。</p>
</body>
</html>
"""
    return report


def generate_report(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    for filename in REQUIRED_FILES:
        if not (output_path / filename).exists():
            raise FileNotFoundError(f"missing pipeline output file: {output_path / filename}")

    outputs = {
        "summary": _read_csv(output_path, "summary.csv"),
        "ic_series": _read_csv(output_path, "ic_series.csv"),
        "quantile_daily_returns": _read_csv(output_path, "quantile_daily_returns.csv"),
        "quantile_summary": _read_csv(output_path, "quantile_summary.csv"),
        "long_short_nav": _read_csv(output_path, "long_short_nav.csv"),
        "turnover": _read_csv(output_path, "turnover.csv"),
        "yearly_performance": _read_csv(output_path, "yearly_performance.csv"),
        "data_quality": _read_csv(output_path, "data_quality.csv"),
    }
    config = _read_config(output_path)
    figures_dir = output_path / "figures"
    _generate_figures(outputs, figures_dir)
    html_text = _build_html(outputs, config)
    report_path = output_path / "report.html"
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML report from single-factor pipeline outputs."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report_path = generate_report(parse_args().output_dir)
    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()
