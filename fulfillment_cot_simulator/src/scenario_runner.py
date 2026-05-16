import pandas as pd
import copy
from src.cot_rules import assign_cot_eligibility
from src.prep_time import add_ready_time
from src.routing_greedy import build_greedy_routes
from src.cost_model import calculate_total_plan_cost
from src.evaluator import evaluate_plan

SCENARIO_GRID = [
    {"name": "base_current",              "cot_time": "09:00", "extra_trucks": 0, "extra_cold_trucks": 0, "labor_mult": 1.0,  "pick_mult": 1.0,  "demand_mult": 1.0, "allow_exception": False},
    {"name": "cot_08_00",                 "cot_time": "08:00", "extra_trucks": 0, "extra_cold_trucks": 0, "labor_mult": 1.0,  "pick_mult": 1.0,  "demand_mult": 1.0, "allow_exception": False},
    {"name": "cot_08_30",                 "cot_time": "08:30", "extra_trucks": 0, "extra_cold_trucks": 0, "labor_mult": 1.0,  "pick_mult": 1.0,  "demand_mult": 1.0, "allow_exception": False},
    {"name": "add_2_trucks",              "cot_time": "09:00", "extra_trucks": 2, "extra_cold_trucks": 0, "labor_mult": 1.0,  "pick_mult": 1.0,  "demand_mult": 1.0, "allow_exception": False},
    {"name": "add_5_trucks",              "cot_time": "09:00", "extra_trucks": 5, "extra_cold_trucks": 0, "labor_mult": 1.0,  "pick_mult": 1.0,  "demand_mult": 1.0, "allow_exception": False},
    {"name": "add_cold_truck",            "cot_time": "09:00", "extra_trucks": 0, "extra_cold_trucks": 1, "labor_mult": 1.0,  "pick_mult": 1.0,  "demand_mult": 1.0, "allow_exception": False},
    {"name": "add_labor_10pct",           "cot_time": "09:00", "extra_trucks": 0, "extra_cold_trucks": 0, "labor_mult": 1.1,  "pick_mult": 1.0,  "demand_mult": 1.0, "allow_exception": False},
    {"name": "slotting_improvement_15pct","cot_time": "09:00", "extra_trucks": 0, "extra_cold_trucks": 0, "labor_mult": 1.0,  "pick_mult": 0.85, "demand_mult": 1.0, "allow_exception": False},
    {"name": "exception_vehicle",         "cot_time": "09:00", "extra_trucks": 0, "extra_cold_trucks": 0, "labor_mult": 1.0,  "pick_mult": 1.0,  "demand_mult": 1.0, "allow_exception": True},
    {"name": "demand_plus_20pct",         "cot_time": "09:00", "extra_trucks": 2, "extra_cold_trucks": 0, "labor_mult": 1.0,  "pick_mult": 1.0,  "demand_mult": 1.2, "allow_exception": False},
    {"name": "combined_best",             "cot_time": "08:30", "extra_trucks": 2, "extra_cold_trucks": 0, "labor_mult": 1.1,  "pick_mult": 0.85, "demand_mult": 1.0, "allow_exception": True},
]


def apply_scenario(base_inputs: dict, scenario: dict) -> dict:
    """Apply scenario modifications to base inputs."""
    inputs = copy.deepcopy(base_inputs)
    params = inputs["params"]

    params["cot_time"] = scenario["cot_time"]

    # Pick multiplier (slotting improvement)
    params["_pick_mult"] = scenario.get("pick_mult", 1.0)
    params["_labor_mult"] = scenario.get("labor_mult", 1.0)

    # Extra trucks
    trucks = inputs["trucks"].copy()
    base_5t = trucks[trucks["truck_type"] == "5T"].iloc[0].to_dict()
    base_cold = trucks[trucks["truck_type"] == "cold_2T"].iloc[0].to_dict()

    extra_rows = []
    for i in range(scenario.get("extra_trucks", 0)):
        row = base_5t.copy()
        row["truck_id"] = f"EXTRA_T{i+1:02d}"
        extra_rows.append(row)

    for i in range(scenario.get("extra_cold_trucks", 0)):
        row = base_cold.copy()
        row["truck_id"] = f"EXTRA_COLD{i+1:02d}"
        extra_rows.append(row)

    if not scenario.get("allow_exception", False):
        trucks = trucks[trucks["can_use_after_16"] == 0]

    if extra_rows:
        trucks = pd.concat([trucks, pd.DataFrame(extra_rows)], ignore_index=True)

    inputs["trucks"] = trucks

    # Demand multiplier
    mult = scenario.get("demand_mult", 1.0)
    if mult != 1.0:
        orders = inputs["orders"].copy()
        orders["weight_kg"] = orders["weight_kg"] * mult
        orders["cbm"] = orders["cbm"] * mult
        orders["carton_qty"] = (orders["carton_qty"] * mult).round().astype(int)
        inputs["orders"] = orders

    return inputs


def run_single_scenario(base_inputs: dict, scenario: dict, order_date: str = "2024-01-15") -> dict:
    """Run one scenario. Return metrics dict."""
    inputs = apply_scenario(base_inputs, scenario)
    params = inputs["params"].copy()
    pick_mult = params.pop("_pick_mult", 1.0)
    params.pop("_labor_mult", 1.0)

    # Apply pick multiplier to pick_sec
    params["pick_sec_per_sku"] = params["pick_sec_per_sku"] * pick_mult

    orders = assign_cot_eligibility(inputs["orders"], params, order_date)
    orders = add_ready_time(orders, params)

    route_df, stop_df = build_greedy_routes(
        orders, inputs["customers"], inputs["trucks"],
        inputs["travel_matrix"], params, order_date
    )

    # propagate unassigned_reasons
    unassigned_reasons = route_df.attrs.get("unassigned_reasons", {})
    stop_df.attrs["unassigned_reasons"] = unassigned_reasons

    late_orders = int((~stop_df["on_time"]).sum()) if not stop_df.empty else 0
    cost = calculate_total_plan_cost(route_df, orders, params, late_orders)
    metrics = evaluate_plan(route_df, stop_df, orders, cost)

    metrics["scenario_name"] = scenario["name"]
    metrics["cot_time"] = scenario["cot_time"]
    metrics["route_df"] = route_df
    metrics["stop_df"] = stop_df
    metrics["orders_with_flags"] = orders
    metrics["unassigned_reasons"] = unassigned_reasons

    return metrics


def run_scenario_grid(base_inputs: dict, scenario_grid: list = None,
                      order_date: str = "2024-01-15") -> tuple:
    """Run all scenarios. Return (comparison_df, all_results)."""
    if scenario_grid is None:
        scenario_grid = SCENARIO_GRID

    all_results = []
    for scenario in scenario_grid:
        print(f"  Running scenario: {scenario['name']} ...", end=" ", flush=True)
        result = run_single_scenario(base_inputs, scenario, order_date)
        all_results.append(result)
        print(f"SLA={result['on_time_rate']:.1%}  Cost={result['total_cost']/1e6:.1f}M  Trucks={result['used_trucks']}")

    columns = [
        "scenario_name", "cot_time", "eligible_orders", "assigned_orders",
        "unassigned_orders", "on_time_rate", "used_trucks", "total_km",
        "total_cost", "cost_per_order", "avg_fill_rate",
        "max_route_finish_time", "feasible_100_sla", "late_orders",
    ]
    rows = []
    for r in all_results:
        row = {c: r.get(c) for c in columns}
        rows.append(row)

    scenario_df = pd.DataFrame(rows)
    feasible = scenario_df[scenario_df["feasible_100_sla"] == True]
    if not feasible.empty:
        scenario_df["recommendation_rank"] = None
        feasible_sorted = feasible.sort_values("total_cost")
        for rank, idx in enumerate(feasible_sorted.index, 1):
            scenario_df.loc[idx, "recommendation_rank"] = rank

    return scenario_df, all_results


def find_best_feasible_scenario(scenario_df: pd.DataFrame) -> pd.Series:
    feasible = scenario_df[scenario_df["feasible_100_sla"] == True]
    if feasible.empty:
        return None
    return feasible.sort_values("total_cost").iloc[0]
