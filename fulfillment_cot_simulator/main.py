#!/usr/bin/env python3
"""
Fulfillment COT Truck Simulation Engine — V1
Run: python main.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.data_loader import load_all
from src.travel_time import build_travel_matrix
from src.scenario_runner import run_scenario_grid, find_best_feasible_scenario
from src.exporter import export_bod_outputs


def generate_sample_data():
    """Generate and save sample CSV files."""
    import random
    np.random.seed(42)
    random.seed(42)

    # Customers
    zones = {
        "north_hcm": (10.82, 106.68, ["Binh Thanh", "Go Vap", "Thu Duc"]),
        "south_hcm": (10.70, 106.70, ["District 7", "Nha Be", "Binh Chanh"]),
        "east_hcm": (10.78, 106.75, ["District 2", "District 9", "Thu Duc City"]),
        "west_hcm": (10.75, 106.60, ["Binh Tan", "Tan Phu", "District 6"]),
        "central_hcm": (10.77, 106.70, ["District 1", "District 3", "District 5"]),
    }
    access_types = ["5T_allowed", "5T_allowed", "5T_allowed", "2T_only", "cold_required", "small_vehicle_only"]
    cust_rows = []
    cid = 1
    for zone, (base_lat, base_lng, districts) in zones.items():
        for i in range(20):
            lat = base_lat + np.random.uniform(-0.05, 0.05)
            lng = base_lng + np.random.uniform(-0.05, 0.05)
            cust_rows.append({
                "customer_id": f"C{cid:03d}",
                "customer_name": f"Store_{cid:03d}",
                "lat": round(lat, 6),
                "lng": round(lng, 6),
                "district": random.choice(districts),
                "zone": zone,
                "receive_start": random.choice(["08:00", "08:00", "09:00"]),
                "receive_end": random.choice(["17:00", "17:00", "18:00"]),
                "unload_min": random.randint(15, 30),
                "truck_access_type": random.choices(
                    access_types, weights=[50, 50, 50, 20, 10, 5], k=1)[0],
                "priority": random.choices(["normal", "vip"], weights=[80, 20], k=1)[0],
            })
            cid += 1
    pd.DataFrame(cust_rows).to_csv(ROOT / "data/sample_customers.csv", index=False)

    # Orders
    customer_ids = [r["customer_id"] for r in cust_rows]
    cust_df = pd.DataFrame(cust_rows).set_index("customer_id")
    order_rows = []
    for i in range(220):
        cid = random.choice(customer_ids)
        cust = cust_df.loc[cid]
        # 70% before 09:00
        if random.random() < 0.70:
            hour = random.randint(6, 8)
            minute = random.randint(0, 59)
        else:
            hour = random.randint(9, 11)
            minute = random.randint(0, 59)
        cold = int(random.random() < 0.15)
        fresh = int(random.random() < 0.20)
        if cust["truck_access_type"] == "cold_required":
            cold = 1
        cartons = random.randint(1, 15)
        skus = random.randint(2, 20)
        weight = round(random.uniform(5, 400), 1)
        cbm = round(random.uniform(0.1, 6.0), 2)
        order_rows.append({
            "order_id": f"O{i+1:04d}",
            "customer_id": cid,
            "order_time": f"{hour:02d}:{minute:02d}",
            "sku_count": skus,
            "carton_qty": cartons,
            "weight_kg": weight,
            "cbm": cbm,
            "fresh_flag": fresh,
            "cold_flag": cold,
            "priority": cust["priority"],
            "inventory_available": random.choices([True, False], weights=[95, 5], k=1)[0],
        })
    pd.DataFrame(order_rows).to_csv(ROOT / "data/sample_orders.csv", index=False)

    # Trucks
    truck_rows = []
    for i in range(1, 6):
        truck_rows.append({"truck_id": f"T{i:02d}", "truck_type": "5T", "max_kg": 5000, "max_cbm": 28,
                           "cold_capable": 0, "fixed_cost": 800000, "cost_per_km": 8000,
                           "driver_cost_per_hour": 45000, "can_use_after_16": 0, "max_stops": 30})
    for i in range(6, 13):
        truck_rows.append({"truck_id": f"T{i:02d}", "truck_type": "2T", "max_kg": 2000, "max_cbm": 12,
                           "cold_capable": 0, "fixed_cost": 500000, "cost_per_km": 6000,
                           "driver_cost_per_hour": 40000, "can_use_after_16": 0, "max_stops": 20})
    for i in range(13, 15):
        truck_rows.append({"truck_id": f"T{i:02d}", "truck_type": "cold_2T", "max_kg": 2000, "max_cbm": 10,
                           "cold_capable": 1, "fixed_cost": 900000, "cost_per_km": 9000,
                           "driver_cost_per_hour": 50000, "can_use_after_16": 0, "max_stops": 15})
    for i in range(15, 17):
        truck_rows.append({"truck_id": f"T{i:02d}", "truck_type": "exception_van", "max_kg": 700, "max_cbm": 4,
                           "cold_capable": 0, "fixed_cost": 400000, "cost_per_km": 5000,
                           "driver_cost_per_hour": 35000, "can_use_after_16": 1, "max_stops": 10})
    pd.DataFrame(truck_rows).to_csv(ROOT / "data/sample_trucks.csv", index=False)

    # Productivity
    prod_rows = [
        {"process_step": "pick", "worker_role": "picker", "sec_per_unit": 30, "worker_count": 10, "efficiency": 0.85, "cost_per_hour": 45000},
        {"process_step": "pack", "worker_role": "packer", "sec_per_unit": 20, "worker_count": 6, "efficiency": 0.85, "cost_per_hour": 45000},
        {"process_step": "load", "worker_role": "loader", "sec_per_unit": 8, "worker_count": 4, "efficiency": 0.90, "cost_per_hour": 40000},
        {"process_step": "qc", "worker_role": "qc", "sec_per_unit": 15, "worker_count": 2, "efficiency": 0.90, "cost_per_hour": 50000},
    ]
    pd.DataFrame(prod_rows).to_csv(ROOT / "data/sample_productivity.csv", index=False)

    print("Sample data generated.")


def make_charts(scenario_df: pd.DataFrame, output_dir: Path):
    """Generate 3 BOD charts."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Fulfillment COT Truck Simulation — Scenario Comparison", fontsize=14, fontweight="bold")

    names = scenario_df["scenario_name"].str.replace("_", "\n")
    costs_m = scenario_df["total_cost"] / 1e6
    sla_pct = scenario_df["on_time_rate"] * 100
    trucks = scenario_df["used_trucks"]
    cpu_k = scenario_df["cost_per_order"] / 1000
    feasible = scenario_df["feasible_100_sla"]

    colors = ["#2ecc71" if f else "#e74c3c" for f in feasible]

    # Chart 1: Cost vs SLA
    ax1 = axes[0]
    ax1.scatter(costs_m, sla_pct, c=colors, s=100, zorder=3)
    for i, (x, y, name) in enumerate(zip(costs_m, sla_pct, scenario_df["scenario_name"])):
        ax1.annotate(name.replace("_", "\n"), (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=6.5)
    ax1.axhline(100, color="navy", linestyle="--", linewidth=1, label="100% SLA target")
    ax1.set_xlabel("Total Cost (Million VND)")
    ax1.set_ylabel("On-Time Rate (%)")
    ax1.set_title("Cost vs SLA Rate")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Chart 2: Trucks used by scenario
    ax2 = axes[1]
    bars = ax2.bar(range(len(scenario_df)), trucks, color=colors)
    ax2.set_xticks(range(len(scenario_df)))
    ax2.set_xticklabels(scenario_df["scenario_name"], rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Number of Trucks Used")
    ax2.set_title("Trucks Used per Scenario")
    ax2.grid(True, alpha=0.3, axis="y")

    # Chart 3: Cost per order
    ax3 = axes[2]
    bars3 = ax3.bar(range(len(scenario_df)), cpu_k, color=colors)
    ax3.set_xticks(range(len(scenario_df)))
    ax3.set_xticklabels(scenario_df["scenario_name"], rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("Cost per Order (Thousand VND)")
    ax3.set_title("Cost per Order by Scenario")
    ax3.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2ecc71", label="100% SLA Feasible"),
                       Patch(facecolor="#e74c3c", label="SLA Not Met")]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(output_dir / "scenario_charts.png", dpi=120, bbox_inches="tight")
    print(f"Charts saved -> {output_dir / 'scenario_charts.png'}")


def print_executive_summary(best_result: dict, best_scenario_row: pd.Series):
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY — BEST FEASIBLE SCENARIO")
    print("=" * 60)
    if best_scenario_row is None:
        print("No 100% SLA feasible scenario found.")
        print("Consider adding more trucks, earlier COT, or more labor.")
        return
    metrics = [
        ("Best Scenario", best_scenario_row["scenario_name"]),
        ("COT Time", best_scenario_row["cot_time"]),
        ("Eligible Orders", best_scenario_row["eligible_orders"]),
        ("Assigned Orders", best_scenario_row["assigned_orders"]),
        ("Unassigned Orders", best_scenario_row["unassigned_orders"]),
        ("Late Orders", best_scenario_row["late_orders"]),
        ("On-Time Rate", f"{best_scenario_row['on_time_rate']:.1%}"),
        ("Used Trucks", best_scenario_row["used_trucks"]),
        ("Total Distance km", best_scenario_row["total_km"]),
        ("Total Cost (VND)", f"{best_scenario_row['total_cost']:,.0f}"),
        ("Cost per Order (VND)", f"{best_scenario_row['cost_per_order']:,.0f}"),
        ("Avg Fill Rate", f"{best_scenario_row['avg_fill_rate']:.1%}"),
        ("Max Route Finish", best_scenario_row["max_route_finish_time"]),
        ("Operational Deadline", "15:30"),
        ("Feasible 100% SLA", best_scenario_row["feasible_100_sla"]),
    ]
    for k, v in metrics:
        print(f"  {k:<25} {v}")
    print("=" * 60)


def main():
    order_date = "2024-01-15"
    config_path = ROOT / "config/default_params.yaml"
    data_dir = ROOT / "data"
    output_dir = ROOT / "outputs"

    print("=" * 60)
    print("Fulfillment COT Truck Simulation Engine — V1")
    print("=" * 60)

    # Generate sample data if needed
    if not (data_dir / "sample_orders.csv").exists():
        print("\nGenerating sample data...")
        generate_sample_data()
    else:
        print("\nSample data found, loading...")

    # Load inputs
    print("\nLoading inputs...")
    inputs = load_all(str(data_dir), str(config_path), order_date)

    # Build travel matrix
    print("Building travel matrix (this may take ~30 seconds for 100 customers)...")
    params = inputs["params"]
    travel_matrix = build_travel_matrix(
        inputs["customers"],
        fc_lat=params["fc_lat"],
        fc_lng=params["fc_lng"],
        fc_id=params["fc_id"],
        avg_speed_kmh=params["avg_speed_kmh"],
    )
    inputs["travel_matrix"] = travel_matrix
    print(f"Travel matrix built: {len(travel_matrix)} pairs.")

    # Run scenarios
    print("\nRunning 11 scenarios...")
    scenario_df, all_results = run_scenario_grid(inputs, order_date=order_date)

    # Find best
    best_row = find_best_feasible_scenario(scenario_df)
    best_result = None
    if best_row is not None:
        best_name = best_row["scenario_name"]
        best_result = next((r for r in all_results if r["scenario_name"] == best_name), None)

    # Print summary
    print_executive_summary(best_result, best_row)

    # Export
    print("\nExporting outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)
    export_bod_outputs(scenario_df, best_result, str(output_dir), all_results)

    # Charts
    print("Generating charts...")
    make_charts(scenario_df, output_dir)

    print("\nDone. Outputs in:", output_dir)
    print("  - scenario_comparison.csv")
    print("  - best_truck_plan.csv")
    print("  - risk_orders.csv")
    print("  - scenario_charts.png")


if __name__ == "__main__":
    main()
