"""Medium acquisition-cost sensitivity analysis for ReCellular.

The Excel workbook is not read or modified. Results are simulated model
outputs based only on the stated parameters, not real operating data.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import highspy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results"

from demand_volatility_sensitivity import (
    SELLING_PRICE,
    TRAINING_SEED,
    TRAINING_SIZE,
    generate_demand,
)


DEMAND_SD = 250
MEDIUM_ACQUISITION_COSTS = [35, 30, 25, 20, 15, 10, 5]
REMANUFACTURING = np.array([30.0, 50.0, 85.0])
GRADES = ("H", "M", "L")


def evaluate_profit(demand: np.ndarray, q: np.ndarray, acquisition: np.ndarray) -> float:
    """Evaluate the original High -> Medium -> Low Excel profit logic."""
    high = np.minimum(demand, q[0])
    remaining = np.maximum(demand - high, 0)
    medium = np.minimum(remaining, q[1])
    remaining = np.maximum(remaining - medium, 0)
    low = np.minimum(remaining, q[2])
    sold = np.column_stack([high, medium, low])
    profit = (
        SELLING_PRICE * sold.sum(axis=1)
        - acquisition @ q
        - sold @ REMANUFACTURING
    )
    return float(profit.mean())


def optimize(training_demand: np.ndarray, medium_cost: float) -> dict[str, float | int | str]:
    """Solve the demand-aggregated MILP and verify it against Excel logic."""
    acquisition = np.array([50.0, float(medium_cost), 5.0])
    unique_demand, counts = np.unique(training_demand, return_counts=True)
    probabilities = counts / len(training_demand)

    model = highspy.Highs()
    model.setOptionValue("output_flag", False)
    model.setOptionValue("mip_rel_gap", 0.0)

    q = model.addVariables(
        3,
        lb=0,
        ub=int(training_demand.max()),
        obj=(-acquisition).tolist(),
        type=highspy.HighsVarType.kInteger,
        name=["Q_H", "Q_M", "Q_L"],
        out_array=True,
    )

    # Exact aggregation: scenarios with equal demand share one recourse block.
    for s, (demand, probability) in enumerate(zip(unique_demand, probabilities)):
        sales = []
        for g, grade in enumerate(GRADES):
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
    if status != "Optimal":
        raise RuntimeError(f"HiGHS did not prove optimality: {status}; gap={info.mip_gap}")

    q_star = np.rint(model.getSolution().col_value[:3]).astype(int)
    verified_profit = evaluate_profit(training_demand, q_star, acquisition)
    if abs(verified_profit - info.objective_function_value) > 1e-6:
        raise AssertionError("MILP objective does not match the original profit logic.")

    return {
        "Medium Acquisition Cost": int(medium_cost),
        "Optimal High": int(q_star[0]),
        "Optimal Medium": int(q_star[1]),
        "Optimal Low": int(q_star[2]),
        "Optimal Total Purchase": int(q_star.sum()),
        "Training Average Profit": verified_profit,
        "MIP Gap": float(info.mip_gap),
        "Status": status,
    }


def write_svg(results: list[dict[str, float | int | str]]) -> Path:
    width, height = 860, 520
    left, right, top, bottom = 90, 35, 45, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    costs = np.array([row["Medium Acquisition Cost"] for row in results], dtype=float)
    series = {
        "High": np.array([row["Optimal High"] for row in results], dtype=float),
        "Medium": np.array([row["Optimal Medium"] for row in results], dtype=float),
        "Low": np.array([row["Optimal Low"] for row in results], dtype=float),
    }
    colors = {"High": "#2563eb", "Medium": "#f59e0b", "Low": "#16a34a"}
    y_max = float(np.ceil(max(v.max() for v in series.values()) / 100) * 100)

    # Display high-to-low cost from left to right, matching the experiment order.
    def x_pos(value: float) -> float:
        return left + (costs.max() - value) / (costs.max() - costs.min()) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / y_max * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="430" y="27" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">Optimal Purchase Quantity by Medium Acquisition Cost</text>',
    ]
    for tick in np.linspace(0, y_max, 6):
        y = y_pos(float(tick))
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.0f}</text>')
    for cost in costs:
        x = x_pos(float(cost))
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 25}" text-anchor="middle" font-family="Arial" font-size="12">{cost:.0f}</text>')
    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="14">Medium Acquisition Cost (high to low)</text>',
        f'<text x="22" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2:.1f})" font-family="Arial" font-size="14">Optimal Purchase Quantity</text>',
    ])
    for idx, (name, values) in enumerate(series.items()):
        points = " ".join(f"{x_pos(c):.1f},{y_pos(v):.1f}" for c, v in zip(costs, values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[name]}" stroke-width="3"/>')
        for cost, value in zip(costs, values):
            parts.append(f'<circle cx="{x_pos(cost):.1f}" cy="{y_pos(value):.1f}" r="4" fill="{colors[name]}"/>')
        legend_x = left + 255 * idx
        parts.append(f'<line x1="{legend_x}" y1="{height - 48}" x2="{legend_x + 28}" y2="{height - 48}" stroke="{colors[name]}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 36}" y="{height - 43}" font-family="Arial" font-size="13">{name}</text>')
    parts.append("</svg>")

    path = OUTPUT_DIR / "figures" / "medium_acquisition_cost_optimal_quantities.svg"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main() -> None:
    # Match the SD=250 baseline training sample exactly.
    z_training = np.random.default_rng(TRAINING_SEED).standard_normal(TRAINING_SIZE)
    training_demand = generate_demand(z_training, DEMAND_SD)

    results = [optimize(training_demand, cost) for cost in MEDIUM_ACQUISITION_COSTS]
    csv_path = OUTPUT_DIR / "medium_acquisition_cost_sensitivity_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    chart_path = write_svg(results)

    print(f"Training seed: {TRAINING_SEED}; size: {TRAINING_SIZE}; demand SD: {DEMAND_SD}")
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
