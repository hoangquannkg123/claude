import pandas as pd
from pathlib import Path


def export_bod_outputs(scenario_df: pd.DataFrame, best_result: dict,
                       output_dir: str, all_results: list = None):
    """Export scenario_comparison.csv, best_truck_plan.csv, risk_orders.csv."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Scenario comparison
    export_cols = [
        "scenario_name", "cot_time", "eligible_orders", "assigned_orders",
        "unassigned_orders", "on_time_rate", "late_orders", "used_trucks",
        "total_km", "total_cost", "cost_per_order", "avg_fill_rate",
        "max_route_finish_time", "feasible_100_sla", "recommendation_rank"
    ]
    cols_present = [c for c in export_cols if c in scenario_df.columns]
    scenario_df[cols_present].to_csv(out / "scenario_comparison.csv", index=False)

    if best_result is None:
        print("No feasible scenario found. Only scenario_comparison.csv exported.")
        return

    # 2. Best truck plan
    route_df = best_result.get("route_df", pd.DataFrame())
    if not route_df.empty:
        truck_plan_cols = [
            "route_id", "truck_id", "truck_type", "zone", "num_customers",
            "num_orders", "depart_time", "finish_time", "total_kg", "total_cbm",
            "kg_utilization", "cbm_utilization", "route_km", "sla_status", "deadline"
        ]
        cols_p = [c for c in truck_plan_cols if c in route_df.columns]
        route_df[cols_p].to_csv(out / "best_truck_plan.csv", index=False)

    # 3. Risk orders
    orders = best_result.get("orders_with_flags", pd.DataFrame())
    stop_df = best_result.get("stop_df", pd.DataFrame())
    unassigned_reasons = best_result.get("unassigned_reasons", {})

    risk_rows = []
    if not orders.empty:
        if "same_day_eligible" in orders.columns:
            eligible = orders[orders["same_day_eligible"] == True]
        else:
            eligible = orders
        assigned_ids = set(stop_df["order_id"].unique()) if not stop_df.empty else set()

        for _, o in eligible.iterrows():
            oid = o["order_id"]
            if oid not in assigned_ids:
                reason = unassigned_reasons.get(oid, "no_feasible_route")
                action = {
                    "no_cold_truck_available": "add_cold_truck",
                    "no_feasible_route": "add_truck_or_split_route",
                    "over_capacity": "add_truck",
                }.get(reason, "review_manually")
                risk_rows.append({
                    "order_id": oid,
                    "customer_id": o.get("customer_id"),
                    "reason": reason,
                    "suggested_action": action,
                })

        if not stop_df.empty:
            late_stops = stop_df[stop_df["on_time"] == False]
            for _, s in late_stops.iterrows():
                risk_rows.append({
                    "order_id": s["order_id"],
                    "customer_id": s["customer_id"],
                    "reason": "late_delivery_after_deadline",
                    "suggested_action": "add_truck_or_exception_vehicle",
                })

    pd.DataFrame(risk_rows).to_csv(out / "risk_orders.csv", index=False)
    print(f"Exported: scenario_comparison.csv, best_truck_plan.csv, risk_orders.csv -> {out}")
