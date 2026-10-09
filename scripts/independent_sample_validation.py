"""Independent out-of-sample validation for fixed ReCellular strategies.

The sample uses NumPy default_rng(2026) and is checked element by element
against the SD=250 validation demand from the demand-volatility script. It is
used only for evaluation, never for purchase-quantity optimization.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from demand_volatility_sensitivity import (
    DEMAND_MEAN,
    SELLING_PRICE,
    VALIDATION_SEED,
    VALIDATION_SIZE,
    excel_round_and_truncate,
    generate_demand,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_WORKBOOK = PROJECT_ROOT / "data" / "ReCellular_Inventory_Planning.xlsx"
DEFAULT_RESULT = PROJECT_ROOT / "results" / "independent_sample_validation_seed2026_results.csv"
DEMAND_SD = 250

STRATEGIES = {
    "H1000": (1000, 0, 0),
    "Mix-2": (965, 0, 302),
    "Solver-100": (788, 0, 451),
    "Solver-5000": (771, 0, 487),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workbook",
        nargs="?",
        type=Path,
        default=OFFICIAL_WORKBOOK,
        help=f"Workbook path (default: {OFFICIAL_WORKBOOK})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULT,
        help=f"Validation result CSV (default: {DEFAULT_RESULT})",
    )
    return parser.parse_args()


def evaluate_strategy(
    demand: np.ndarray,
    quantities: tuple[int, int, int],
    acquisition_cost: np.ndarray,
    remanufacturing_cost: np.ndarray,
    selling_price: float,
) -> dict[str, float]:
    high_qty, medium_qty, low_qty = quantities
    high_sold = np.minimum(demand, high_qty)
    remaining_after_high = np.maximum(demand - high_sold, 0)
    medium_sold = np.minimum(remaining_after_high, medium_qty)
    remaining_after_medium = np.maximum(remaining_after_high - medium_sold, 0)
    low_sold = np.minimum(remaining_after_medium, low_qty)

    sold = high_sold + medium_sold + low_sold
    purchase_cost = np.dot(np.asarray(quantities, dtype=float), acquisition_cost)
    reman_cost = (
        high_sold * remanufacturing_cost[0]
        + medium_sold * remanufacturing_cost[1]
        + low_sold * remanufacturing_cost[2]
    )
    profit = selling_price * sold - purchase_cost - reman_cost
    return {
        "Average Profit": float(np.mean(profit)),
        "Average Unmet Demand": float(np.mean(np.maximum(demand - sold, 0))),
        "Average Leftover Inventory": float(np.mean(sum(quantities) - sold)),
    }


def main() -> None:
    args = parse_args()
    workbook_path = args.workbook.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    workbook = load_workbook(workbook_path, data_only=True, read_only=True)
    inputs = workbook["Inputs"]

    acquisition_cost = np.array(
        [inputs["B2"].value, inputs["C2"].value, inputs["D2"].value], dtype=float
    )
    remanufacturing_cost = np.array(
        [inputs["B3"].value, inputs["C3"].value, inputs["D3"].value], dtype=float
    )
    selling_prices = np.array(
        [inputs["B4"].value, inputs["C4"].value, inputs["D4"].value], dtype=float
    )
    mean_demand = float(inputs["B6"].value)
    demand_sd = float(inputs["B7"].value)
    workbook.close()

    np.testing.assert_array_equal(acquisition_cost, [50, 35, 5])
    np.testing.assert_array_equal(remanufacturing_cost, [30, 50, 85])
    np.testing.assert_array_equal(selling_prices, [SELLING_PRICE] * 3)
    if mean_demand != DEMAND_MEAN or demand_sd != DEMAND_SD:
        raise AssertionError(f"Unexpected demand parameters: mean={mean_demand}, sd={demand_sd}")

    raw_demand = np.random.default_rng(VALIDATION_SEED).normal(
        DEMAND_MEAN, DEMAND_SD, VALIDATION_SIZE
    )
    validation_demand = excel_round_and_truncate(raw_demand)

    validation_z = np.random.default_rng(VALIDATION_SEED).standard_normal(VALIDATION_SIZE)
    sensitivity_validation_demand = generate_demand(validation_z, DEMAND_SD)
    np.testing.assert_array_equal(validation_demand, sensitivity_validation_demand)

    metrics = {
        name: evaluate_strategy(
            validation_demand,
            quantities,
            acquisition_cost,
            remanufacturing_cost,
            float(selling_prices[0]),
        )
        for name, quantities in STRATEGIES.items()
    }
    differences = {
        "Solver-100": (
            "Solver-100 minus Mix-2",
            metrics["Solver-100"]["Average Profit"] - metrics["Mix-2"]["Average Profit"],
        ),
        "Solver-5000": (
            "Solver-5000 minus Solver-100",
            metrics["Solver-5000"]["Average Profit"] - metrics["Solver-100"]["Average Profit"],
        ),
    }

    rows = []
    for name, quantities in STRATEGIES.items():
        difference_label, difference = differences.get(name, ("", ""))
        rows.append({
            "Strategy": name,
            "High": quantities[0],
            "Medium": quantities[1],
            "Low": quantities[2],
            **metrics[name],
            "Profit Difference Comparison": difference_label,
            "Profit Difference": difference,
            "Validation Seed": VALIDATION_SEED,
            "Validation Scenarios": VALIDATION_SIZE,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Independent simulation sample; not real operating data.")
    print(f"Seed={VALIDATION_SEED}; scenarios={VALIDATION_SIZE}; demand=Normal({DEMAND_MEAN}, {DEMAND_SD}).")
    print("Elementwise match with demand-volatility SD=250 validation demand: PASS")
    for row in rows:
        print(row)
    print(f"Result saved: {output_path}")


if __name__ == "__main__":
    main()
