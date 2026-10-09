"""Demand-volatility sensitivity analysis for ReCellular inventory planning.

This script does not read or modify the Excel workbook. It uses only the
parameters specified for this experiment. All outputs are simulated model
results, not real operating data.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import highspy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results"


DEMAND_MEAN = 1000
DEMAND_SDS = [100, 150, 250, 350, 500]
TRAINING_SIZE = 5_000
VALIDATION_SIZE = 10_000
TRAINING_SEED = 42
VALIDATION_SEED = 2026

ACQUISITION = np.array([50.0, 35.0, 5.0])
REMANUFACTURING = np.array([30.0, 50.0, 85.0])
SELLING_PRICE = 120.0


def excel_round_and_truncate(raw_demand: np.ndarray) -> np.ndarray:
    """Replicate MAX(0, ROUND(raw demand, 0))."""
    rounded = np.sign(raw_demand) * np.floor(np.abs(raw_demand) + 0.5)
    return np.maximum(0, rounded).astype(int)


def generate_demand(z: np.ndarray, sd: int) -> np.ndarray:
    return excel_round_and_truncate(DEMAND_MEAN + sd * z)


def excel_logic(demand: np.ndarray, q: np.ndarray) -> dict[str, float]:
    high = np.minimum(demand, q[0])
    remaining = np.maximum(demand - high, 0)
    medium = np.minimum(remaining, q[1])
    remaining = np.maximum(remaining - medium, 0)
    low = np.minimum(remaining, q[2])
    sold = np.column_stack([high, medium, low])

    profit = (
        SELLING_PRICE * sold.sum(axis=1)
        - ACQUISITION @ q
        - sold @ REMANUFACTURING
    )
    total_inventory = int(q.sum())
    total_sold = sold.sum(axis=1)
    return {
        "average_profit": float(profit.mean()),
        "average_unmet_demand": float(np.maximum(demand - total_sold, 0).mean()),
        "average_leftover_inventory": float((total_inventory - total_sold).mean()),
    }


def optimize(training_demand: np.ndarray) -> tuple[np.ndarray, str, float, float]:
    """Solve an exact, demand-aggregated equivalent of the scenario MILP."""
    unique_demand, counts = np.unique(training_demand, return_counts=True)
    probabilities = counts / len(training_demand)

    model = highspy.Highs()
    model.setOptionValue("output_flag", False)
    model.setOptionValue("mip_rel_gap", 0.0)

    q = model.addVariables(
        3,
        lb=0,
        ub=int(training_demand.max()),
        obj=(-ACQUISITION).tolist(),
        type=highspy.HighsVarType.kInteger,
        name=["Q_H", "Q_M", "Q_L"],
        out_array=True,
    )

    # Identical-demand scenarios are aggregated with their exact frequencies.
    # R can be continuous: for integer Q and demand, the recourse LP has an
    # integral extreme point, so this is equivalent to integer unit sales.
    for s, (demand, probability) in enumerate(zip(unique_demand, probabilities)):
        sales = []
        for g, grade in enumerate(("H", "M", "L")):
            r = model.addVariable(
                lb=0,
                ub=int(demand),
                obj=float(probability * (SELLING_PRICE - REMANUFACTURING[g])),
                name=f"R_{grade}_{s + 1}",
            )
            model.addConstr(r <= q[g])
            sales.append(r)
        model.addConstr(sum(sales) <= int(demand))

    model.setMaximize()
    model.run()

    status = model.modelStatusToString(model.getModelStatus())
    info = model.getInfo()
    q_star = np.rint(model.getSolution().col_value[:3]).astype(int)
    training_metrics = excel_logic(training_demand, q_star)

    if status != "Optimal":
        raise RuntimeError(f"HiGHS did not prove optimality: {status}; gap={info.mip_gap}")
    if abs(training_metrics["average_profit"] - info.objective_function_value) > 1e-6:
        raise AssertionError("MILP objective does not match the original profit logic.")

    return q_star, status, float(info.mip_gap), training_metrics["average_profit"]


def write_svg(results: list[dict[str, float]]) -> Path:
    """Write a dependency-free SVG line chart."""
    width, height = 860, 520
    left, right, top, bottom = 90, 35, 45, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    sds = np.array([row["Demand SD"] for row in results], dtype=float)
    series = {
        "High": np.array([row["Optimal High"] for row in results], dtype=float),
        "Medium": np.array([row["Optimal Medium"] for row in results], dtype=float),
        "Low": np.array([row["Optimal Low"] for row in results], dtype=float),
    }
    colors = {"High": "#2563eb", "Medium": "#f59e0b", "Low": "#16a34a"}
    y_max = float(np.ceil(max(values.max() for values in series.values()) / 100) * 100)

    def x_pos(value: float) -> float:
        return left + (value - sds.min()) / (sds.max() - sds.min()) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / y_max * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="430" y="27" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">Optimal Purchase Quantity by Demand Volatility</text>',
    ]
    for tick in np.linspace(0, y_max, 6):
        y = y_pos(float(tick))
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.0f}</text>')
    for sd in sds:
        x = x_pos(float(sd))
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 25}" text-anchor="middle" font-family="Arial" font-size="12">{sd:.0f}</text>')
    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="14">Demand SD</text>',
        f'<text x="22" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2:.1f})" font-family="Arial" font-size="14">Optimal Purchase Quantity</text>',
    ])
    for idx, (name, values) in enumerate(series.items()):
        points = " ".join(f"{x_pos(sd):.1f},{y_pos(value):.1f}" for sd, value in zip(sds, values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[name]}" stroke-width="3"/>')
        for sd, value in zip(sds, values):
            parts.append(f'<circle cx="{x_pos(sd):.1f}" cy="{y_pos(value):.1f}" r="4" fill="{colors[name]}"/>')
        legend_x = left + 255 * idx
        parts.append(f'<line x1="{legend_x}" y1="{height - 48}" x2="{legend_x + 28}" y2="{height - 48}" stroke="{colors[name]}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 36}" y="{height - 43}" font-family="Arial" font-size="13">{name}</text>')
    parts.append('</svg>')

    path = OUTPUT_DIR / "figures" / "demand_volatility_optimal_quantities.svg"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main() -> None:
    # Common random numbers across SDs; validation is independent of training.
    z_training = np.random.default_rng(TRAINING_SEED).standard_normal(TRAINING_SIZE)
    z_validation = np.random.default_rng(VALIDATION_SEED).standard_normal(VALIDATION_SIZE)

    results: list[dict[str, float]] = []
    for sd in DEMAND_SDS:
        training_demand = generate_demand(z_training, sd)
        validation_demand = generate_demand(z_validation, sd)
        q_star, status, gap, training_profit = optimize(training_demand)
        validation = excel_logic(validation_demand, q_star)
        results.append({
            "Demand SD": sd,
            "Optimal High": int(q_star[0]),
            "Optimal Medium": int(q_star[1]),
            "Optimal Low": int(q_star[2]),
            "Training Average Profit": training_profit,
            "Validation Average Profit": validation["average_profit"],
            "Validation Average Unmet Demand": validation["average_unmet_demand"],
            "Validation Average Leftover Inventory": validation["average_leftover_inventory"],
            "MIP Gap": gap,
            "Status": status,
        })

    csv_path = OUTPUT_DIR / "demand_volatility_sensitivity_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    chart_path = write_svg(results)

    print(f"Training seed: {TRAINING_SEED}; validation seed: {VALIDATION_SEED}")
    print(" | ".join(results[0].keys()))
    for row in results:
        print(" | ".join(
            f"{value:.2f}" if isinstance(value, float) else str(value)
            for value in row.values()
        ))
    print(f"Results: {csv_path}")
    print(f"Chart: {chart_path}")


if __name__ == "__main__":
    main()
