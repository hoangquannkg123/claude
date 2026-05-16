import pandas as pd
import numpy as np


def evaluate_sla(stop_df: pd.DataFrame, orders: pd.DataFrame) -> dict:
    """Return SLA metrics."""
    eligible = orders[orders["same_day_eligible"] == True] if "same_day_eligible" in orders.columns else orders
    eligible_count = len(eligible)

    unassigned_reasons = stop_df.attrs.get("unassigned_reasons", {})
    assigned_ids = set(stop_df["order_id"].unique()) if not stop_df.empty else set()
    unassigned_count = len([oid for oid in eligible["order_id"] if oid not in assigned_ids])

    if stop_df.empty:
        return {
            "eligible_orders": eligible_count,
            "assigned_orders": 0,
            "unassigned_orders": eligible_count,
            "on_time_orders": 0,
            "late_orders": 0,
            "on_time_rate": 0.0,
            "p95_lead_time_min": None,
        }

    on_time = stop_df["on_time"].sum()
    late = (~stop_df["on_time"]).sum()
    assigned = len(stop_df)

    # p95 lead time: arrival - order_time
    order_times = orders.set_index("order_id")["order_time"]
    stop_merged = stop_df.copy()
    stop_merged["order_time"] = stop_merged["order_id"].map(order_times)
    stop_merged["lead_time_min"] = (
        stop_merged["arrival_time"] - stop_merged["order_time"]
    ).dt.total_seconds() / 60.0
    p95_lead = np.percentile(stop_merged["lead_time_min"].dropna(), 95) if len(stop_merged) > 0 else None

    return {
        "eligible_orders": eligible_count,
        "assigned_orders": assigned,
        "unassigned_orders": unassigned_count,
        "on_time_orders": int(on_time),
        "late_orders": int(late),
        "on_time_rate": round(on_time / eligible_count, 4) if eligible_count > 0 else 0.0,
        "p95_lead_time_min": round(p95_lead, 1) if p95_lead else None,
    }


def evaluate_truck_utilization(route_df: pd.DataFrame) -> dict:
    """Return truck utilization metrics."""
    if route_df.empty:
        return {
            "used_trucks": 0,
            "avg_kg_utilization": 0.0,
            "avg_cbm_utilization": 0.0,
            "avg_fill_rate": 0.0,
            "total_km": 0.0,
            "avg_stops_per_truck": 0.0,
            "max_route_finish_time": None,
        }

    return {
        "used_trucks": len(route_df),
        "avg_kg_utilization": round(route_df["kg_utilization"].mean(), 3),
        "avg_cbm_utilization": round(route_df["cbm_utilization"].mean(), 3),
        "avg_fill_rate": round((route_df["kg_utilization"] + route_df["cbm_utilization"]).mean() / 2, 3),
        "total_km": round(route_df["route_km"].sum(), 1),
        "avg_stops_per_truck": round(route_df["num_customers"].mean(), 1),
        "max_route_finish_time": route_df["finish_time"].max(),
    }


def evaluate_plan(route_df: pd.DataFrame, stop_df: pd.DataFrame,
                  orders: pd.DataFrame, cost_breakdown: dict) -> dict:
    """Return full scenario metrics dict."""
    sla = evaluate_sla(stop_df, orders)
    truck = evaluate_truck_utilization(route_df)
    metrics = {**sla, **truck, **cost_breakdown}
    metrics["cost_per_order"] = (
        round(cost_breakdown["total_cost"] / sla["eligible_orders"], 0)
        if sla["eligible_orders"] > 0 else 0
    )
    metrics["feasible_100_sla"] = (
        sla["on_time_rate"] == 1.0
        and sla["unassigned_orders"] == 0
        and sla["late_orders"] == 0
    )
    return metrics
