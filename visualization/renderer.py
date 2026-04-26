from __future__ import annotations

from typing import Any


def _scale_x(x: float, world_min: float, world_max: float, width: int, margin: int) -> float:
    span = max(world_max - world_min, 1.0)
    return margin + ((x - world_min) / span) * (width - 2 * margin)


def build_road_svg(state: dict[str, Any], title: str = "Platoon") -> str:
    width = 980
    height = 290
    margin = 34

    vehicles = state.get("vehicles", {})
    xs = [float(v.get("x", 0.0)) for v in vehicles.values()] if vehicles else [0.0, 100.0]
    world_min = min(xs) - 36.0
    world_max = max(xs) + 36.0

    color_map = {0: "#9ca3af", 1: "#3b82f6", 2: "#16a34a"}
    phase_colors = {
        "steady": "#0ea5e9",
        "brake_event": "#ef4444",
        "hold_low": "#f59e0b",
        "recovery": "#22c55e",
        "steady_2": "#0ea5e9",
        "pulse_brake": "#ef4444",
        "rebound": "#22c55e",
        "traffic_wave": "#a855f7",
        "recover": "#06b6d4",
        "downhill_gain": "#fb7185",
        "low_friction": "#eab308",
        "cutin_emergency": "#dc2626",
        "merge_zone": "#f59e0b",
        "post_merge": "#22c55e",
    }

    lane_top = 95
    lane_height = 88
    phase = str(state.get("phase", "steady"))
    phase_color = phase_colors.get(phase, "#334155")

    dynamics = state.get("dynamics", {})
    road_grip = float(dynamics.get("road_grip", 1.0))
    road_grade = float(dynamics.get("road_grade", 0.0))

    chunks: list[str] = []
    chunks.append(
        f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}'>"
    )
    chunks.append("<defs>")
    chunks.append("<linearGradient id='sky' x1='0' x2='0' y1='0' y2='1'>")
    chunks.append("<stop offset='0%' stop-color='#e0f2fe'/><stop offset='100%' stop-color='#dbeafe'/></linearGradient>")
    chunks.append("<linearGradient id='road' x1='0' x2='0' y1='0' y2='1'>")
    chunks.append("<stop offset='0%' stop-color='#1f2937'/><stop offset='100%' stop-color='#111827'/></linearGradient>")
    chunks.append("</defs>")
    chunks.append("<rect x='0' y='0' width='100%' height='100%' fill='url(#sky)'/>")
    chunks.append(f"<rect x='0' y='70' width='100%' height='4' fill='{phase_color}' opacity='0.75'/>")
    # Calculate camera offset (center on lead vehicle or average)
    avg_x = sum(float(v.get("x", 0.0)) for v in state.get("vehicles", {}).values()) / max(len(state.get("vehicles", {})), 1)
    view_width_m = 120.0
    world_min = avg_x - view_width_m / 2
    world_max = avg_x + view_width_m / 2

    # Draw scrolling road markers
    dash_size = 20
    gap_size = 20
    period = dash_size + gap_size
    # Offset based on world coordinate to simulate movement
    offset = (avg_x * 10) % period 

    chunks.append(
        f"<rect x='{margin}' y='{lane_top}' width='{width - 2 * margin}' height='{lane_height}' rx='12' fill='url(#road)' opacity='0.97'/>"
    )
    
    # Draw merging lane if in merge scenario
    if state.get("scenario") == "scenario_02_merge":
        # Merging road curve - also relative to camera
        merge_start_px = _scale_x(150.0 - 60.0, world_min, world_max, width, margin)
        merge_end_px = _scale_x(150.0, world_min, world_max, width, margin)
        
        chunks.append(
            f"<path d='M {merge_start_px} {lane_top + 120} Q {(merge_start_px + merge_end_px)/2} {lane_top + 100}, {merge_end_px} {lane_top + lane_height/2}' "
            f"fill='none' stroke='#1f2937' stroke-width='{lane_height}' stroke-linecap='round' opacity='0.9'/>"
        )

    chunks.append(
        f"<line x1='{margin}' y1='{lane_top + lane_height / 2}' x2='{width - margin}' y2='{lane_top + lane_height / 2}' "
        f"stroke='#f59e0b' stroke-width='3' stroke-dasharray='{dash_size} {gap_size}' stroke-dashoffset='{offset}'/>"
    )
    chunks.append(f"<text x='{margin}' y='36' font-size='22' font-family='Verdana' fill='#0f172a'>{title}</text>")
    chunks.append(
        f"<text x='{margin}' y='58' font-size='13' font-family='Verdana' fill='#334155'>Scenario: {state.get('scenario', 'scenario_01_brake')} | Step {state.get('timestep', 0)} | Phase: {phase}</text>"
    )
    chunks.append(
        f"<rect x='{width - 260}' y='16' width='236' height='48' rx='9' fill='#0f172a' opacity='0.82'/>"
    )
    chunks.append(
        f"<text x='{width - 246}' y='36' font-size='12' font-family='Verdana' fill='#e2e8f0'>Road grip: {road_grip:.2f}</text>"
    )
    chunks.append(
        f"<text x='{width - 246}' y='54' font-size='12' font-family='Verdana' fill='#e2e8f0'>Road grade: {road_grade:+.3f}</text>"
    )

    if state.get("collision"):
        chunks.append("<rect x='0' y='0' width='100%' height='100%' fill='#dc2626' opacity='0.16'/>")
        chunks.append("<text x='760' y='35' font-size='20' font-family='Verdana' fill='#b91c1c'>COLLISION</text>")

    for car_id in sorted(vehicles.keys()):
        vehicle = vehicles[car_id]
        x_pos = _scale_x(float(vehicle.get("x", 0.0)), world_min, world_max, width, margin)
        
        # Skip if far off-screen
        if x_pos < -200 or x_pos > width + 200:
            continue
            
        car_px_len = 56
        car_px_h = 24
        
        # Scale y: y=0 is center of main lane, y=3.5 is center of merge lane (visually below)
        y_offset_px = float(vehicle.get("y", 0.0)) * 20.0
        y = lane_top + (lane_height - car_px_h) / 2 + y_offset_px
        fill = color_map.get(int(car_id), "#64748b")

        speed = float(vehicle.get("velocity", 0.0))
        speed_bar = min(44.0, speed * 1.6)

        chunks.append(
            f"<rect x='{x_pos - car_px_len + 4}' y='{y + 4}' width='{car_px_len}' height='{car_px_h}' rx='7' fill='#0b1220' opacity='0.26'/>"
        )
        chunks.append(
            f"<rect x='{x_pos - car_px_len}' y='{y}' width='{car_px_len}' height='{car_px_h}' rx='7' fill='{fill}'/>"
        )
        chunks.append(
            f"<rect x='{x_pos - car_px_len}' y='{y + car_px_h + 5}' width='44' height='5' rx='2' fill='#dbeafe' opacity='0.9'/>"
        )
        chunks.append(
            f"<rect x='{x_pos - car_px_len}' y='{y + car_px_h + 5}' width='{speed_bar:.2f}' height='5' rx='2' fill='#0284c7'/>"
        )
        chunks.append(
            f"<text x='{x_pos - car_px_len}' y='{y - 8}' font-size='12' font-family='Verdana' fill='#0f172a'>Car {car_id} | v={speed:.2f} m/s</text>"
        )

    if 0 in vehicles and 1 in vehicles:
        gap_01 = float(vehicles[0]["x"] - vehicles[1]["x"] - vehicles[1].get("length", 4.5))
        chunks.append(
            f"<text x='{margin}' y='243' font-size='12' font-family='Verdana' fill='#0f172a'>Gap(0->1): {gap_01:.2f} m</text>"
        )

    if 1 in vehicles and 2 in vehicles:
        gap_12 = float(vehicles[1]["x"] - vehicles[2]["x"] - vehicles[2].get("length", 4.5))
        chunks.append(
            f"<text x='{margin + 170}' y='243' font-size='12' font-family='Verdana' fill='#0f172a'>Gap(1->2): {gap_12:.2f} m</text>"
        )

    chunks.append(
        f"<rect x='{margin}' y='252' width='{width - 2 * margin}' height='10' rx='5' fill='#dbeafe' opacity='0.9'/>"
    )
    progress = float(state.get("timestep", 0))
    max_steps = float(state.get("max_steps", 80))
    progress_ratio = min(progress / max(max_steps, 1.0), 1.0)
    chunks.append(
        f"<rect x='{margin}' y='252' width='{(width - 2 * margin) * progress_ratio:.2f}' height='10' rx='5' fill='{phase_color}'/>"
    )

    chunks.append("</svg>")
    return "".join(chunks)
