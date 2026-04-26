from __future__ import annotations

import math
from typing import Any


def _scale_x(x: float, world_min: float, world_max: float, width: int, margin: int) -> float:
    span = max(world_max - world_min, 1.0)
    return margin + ((x - world_min) / span) * (width - 2 * margin)


def _mph(mps: float) -> float:
    return mps * 2.2369362921


def _draw_topdown_car(
    chunks: list[str],
    cx: float,
    cy: float,
    car_id: int,
    velocity: float,
    path_type: str,
    uid: int,
    is_lead: bool,
    vehicle_role: str = "passenger",
) -> None:
    """Plan view: length along road (+x → right), width lateral. Matches a bird's-eye reading of the lane."""
    car_len = 52.0
    car_wid = 22.0
    x0 = cx - car_len / 2
    y0 = cy - car_wid / 2
    rx = 6.0

    if vehicle_role == "ambulance":
        fill = "#7f1d1d"
        stroke = "#fb923c"
        accent = "#fed7aa"
    elif is_lead:
        fill = "#4b5563"
        stroke = "#9ca3af"
        accent = "#d1d5db"
    elif car_id == 1:
        fill = "#1e3a5f"
        stroke = "#38bdf8"
        accent = "#7dd3fc"
    else:
        fill = "#164e63"
        stroke = "#22d3ee"
        accent = "#67e8f9"

    body_grad = f"tdBody_{uid}_{car_id}"
    chunks.append(
        f"<linearGradient id='{body_grad}' x1='0%' y1='0%' x2='100%' y2='0%'>"
        f"<stop offset='0%' stop-color='#0f172a'/><stop offset='50%' stop-color='{fill}'/>"
        f"<stop offset='100%' stop-color='#1e293b'/></linearGradient>"
    )

    chunks.append(
        f"<rect x='{x0}' y='{y0}' width='{car_len}' height='{car_wid}' rx='{rx}' "
        f"fill='url(#{body_grad})' stroke='{stroke}' stroke-width='1.5'/>"
    )
    # Rear (left, −x): subtle brake lamps
    chunks.append(
        f"<line x1='{x0 + 3}' y1='{y0 + 4}' x2='{x0 + 3}' y2='{y0 + car_wid - 4}' "
        f"stroke='#b91c1c' stroke-width='3' stroke-linecap='round' opacity='0.85'/>"
    )
    # Front (+x): heading mark (no headlight cone — avoids stray “X” with lane glow)
    chunks.append(
        f"<polygon points='{x0 + car_len - 2},{cy} {x0 + car_len - 11},{cy - 5} {x0 + car_len - 11},{cy + 5}' "
        f"fill='{accent}' opacity='0.9'/>"
    )
    chunks.append(
        f"<line x1='{x0 + car_len * 0.62}' y1='{y0 + 2}' x2='{x0 + car_len * 0.62}' y2='{y0 + car_wid - 2}' "
        f"stroke='#334155' stroke-width='1' opacity='0.7'/>"
    )

    if vehicle_role == "ambulance":
        label = "AMB"
    elif is_lead:
        label = "LEAD"
    else:
        label = f"A{car_id}"
    if path_type == "merge":
        label += " · MERGE"
    chunks.append(
        f"<text x='{cx}' y='{y0 + car_wid + 14}' text-anchor='middle' font-size='11' "
        f"font-family='Segoe UI, system-ui, sans-serif' fill='#94a3b8'>{label} · {_mph(velocity):.0f} mph</text>"
    )


def _power_kw_readout(net_accel: float, max_a: float = 8.0) -> float:
    """Map |a| to a pseudo kW needle (display-only)."""
    return float(min(160.0, max(0.0, abs(net_accel) / max_a * 95.0 + abs(net_accel) * 8.0)))


def build_road_svg(state: dict[str, Any], title: str = "Platoon") -> str:
    width = 1000
    height = 420
    margin = 36

    vehicles = state.get("vehicles", {})
    xs = [float(v.get("x", 0.0)) for v in vehicles.values()] if vehicles else [0.0, 100.0]
    avg_x = sum(xs) / max(len(xs), 1)
    view_width_m = 125.0
    world_min = avg_x - view_width_m / 2
    world_max = avg_x + view_width_m / 2

    phase = str(state.get("phase", "steady"))
    phase_colors = {
        "steady": "#38bdf8",
        "brake_event": "#f87171",
        "hold_low": "#fbbf24",
        "recovery": "#4ade80",
        "steady_2": "#38bdf8",
        "merge_zone": "#22d3ee",
        "post_merge": "#4ade80",
        "pulse_brake": "#f87171",
        "traffic_wave": "#c084fc",
        "low_friction": "#facc15",
        "cutin_emergency": "#ef4444",
        "downhill_gain": "#fb7185",
        "recover": "#22d3ee",
        "rebound": "#4ade80",
        "ambulance_approach": "#f472b6",
        "ambulance_pass": "#fb7185",
        "post_pass": "#34d399",
    }
    phase_color = phase_colors.get(phase, "#64748b")

    dynamics = state.get("dynamics", {})
    road_grip = float(dynamics.get("road_grip", 1.0))
    road_grade = float(dynamics.get("road_grade", 0.0))

    uid = abs(hash((state.get("timestep", 0), title, phase))) % 9_999_999

    chunks: list[str] = []
    chunks.append(
        f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg' "
        f"viewBox='0 0 {width} {height}' style='background:#07080c'>"
    )
    chunks.append("<defs>")
    chunks.append(
        f"<filter id='laneGlow_{uid}' x='-20%' y='-20%' width='140%' height='140%'>"
        f"<feGaussianBlur stdDeviation='2.8' result='blur'/></filter>"
    )
    chunks.append(
        "<linearGradient id='bgVignette' x1='0' x2='0' y1='0' y2='1'>"
        "<stop offset='0%' stop-color='#0c1220'/><stop offset='100%' stop-color='#050608'/></linearGradient>"
    )
    chunks.append(
        "<linearGradient id='roadMatte' x1='0' x2='0' y1='0' y2='1'>"
        "<stop offset='0%' stop-color='#1a1f2e'/><stop offset='100%' stop-color='#0f1218'/></linearGradient>"
    )
    chunks.append("</defs>")

    chunks.append("<rect x='0' y='0' width='100%' height='100%' fill='url(#bgVignette)'/>")

    # --- Instrument cluster strip (Tesla-like) ---
    ego_v = float(vehicles.get(1, vehicles.get(0, {})).get("velocity", 0.0)) if vehicles else 0.0
    ego_na = float(vehicles.get(1, {}).get("net_acceleration", 0.0)) if vehicles else 0.0
    kw = _power_kw_readout(ego_na)

    chunks.append(f"<text x='{margin}' y='34' font-size='13' font-family='Segoe UI, system-ui, sans-serif' fill='#64748b'>{title}</text>")
    chunks.append(
        f"<text x='{margin}' y='54' font-size='11' font-family='Segoe UI, system-ui, sans-serif' fill='#475569'>"
        f"{state.get('scenario', '')} · step {state.get('timestep', 0)} · {phase}</text>"
    )

    # Center speed
    cx_spd = width * 0.5
    chunks.append(
        f"<text x='{cx_spd}' y='88' text-anchor='middle' font-size='56' font-weight='600' "
        f"font-family='Segoe UI, system-ui, sans-serif' fill='#f8fafc'>{_mph(ego_v):.0f}</text>"
    )
    chunks.append(
        f"<text x='{cx_spd}' y='112' text-anchor='middle' font-size='14' font-family='Segoe UI, system-ui, sans-serif' fill='#64748b'>mph</text>"
    )
    chunks.append(
        f"<text x='{cx_spd}' y='130' text-anchor='middle' font-size='11' font-family='Segoe UI, system-ui, sans-serif' fill='#475569'>"
        f"{ego_v:.1f} m/s · grip {road_grip:.2f} · grade {road_grade:+.2f}</text>"
    )

    # Right power arc (semi-circle gauge)
    gx, gy, gr = width - margin - 70, 118, 58
    chunks.append(
        f"<path d='M {gx - gr} {gy} A {gr} {gr} 0 0 1 {gx + gr} {gy}' fill='none' stroke='#1e293b' stroke-width='10' stroke-linecap='round'/>"
    )
    arc_len = min(1.0, kw / 160.0) * (math.pi * gr)
    chunks.append(
        f"<path d='M {gx - gr} {gy} A {gr} {gr} 0 0 1 {gx + gr} {gy}' fill='none' stroke='#fb923c' "
        f"stroke-width='10' stroke-linecap='round' stroke-dasharray='{arc_len:.1f} {math.pi * gr * 2:.1f}'/>"
    )
    chunks.append(
        f"<text x='{gx}' y='{gy + 8}' text-anchor='middle' font-size='11' font-family='Segoe UI, system-ui, sans-serif' fill='#94a3b8'>kW</text>"
    )
    chunks.append(
        f"<text x='{gx}' y='{gy + 26}' text-anchor='middle' font-size='13' font-weight='500' font-family='Segoe UI, system-ui, sans-serif' fill='#e2e8f0'>{kw:.0f}</text>"
    )

    # Phase bar
    chunks.append(f"<rect x='0' y='138' width='100%' height='3' fill='{phase_color}' opacity='0.85'/>")

    road_top = 158
    road_h = 168
    vanish_y = road_top + road_h * 0.08

    chunks.append(
        f"<rect x='{margin}' y='{road_top}' width='{width - 2 * margin}' height='{road_h}' rx='14' fill='url(#roadMatte)' stroke='#1e293b' stroke-width='1'/>"
    )

    scenario = state.get("scenario", "")
    merge_layout = state.get("merge_layout") or {}

    # Perspective faux: horizontal scale slightly narrower at top
    def road_x_left(f: float) -> float:
        return margin + (width - 2 * margin) * (0.08 + f * 0.84)

    def road_x_right(f: float) -> float:
        return margin + (width - 2 * margin) * (0.92 - f * 0.84)

    if scenario == "scenario_02_merge" and merge_layout:
        xm = float(merge_layout.get("x_merge", 150.0))
        merge_start_px = _scale_x(xm - 42.0, world_min, world_max, width, margin)
        merge_tip_px = _scale_x(xm, world_min, world_max, width, margin)
        # Lower (near) edge of road
        y_lo = road_top + road_h - 8
        y_hi = road_top + 28
        # Main lane left / right glowing edges (trapezoid)
        lx0, lx1 = road_x_left(0.02), road_x_left(0.98)
        rx0, rx1 = road_x_right(0.02), road_x_right(0.98)
        chunks.append(
            f"<path d='M {lx0} {y_lo} L {lx1} {y_hi} M {rx0} {y_lo} L {rx1} {y_hi}' "
            f"stroke='#0ea5e9' stroke-width='5' stroke-linecap='round' opacity='0.95' filter='url(#laneGlow_{uid})'/>"
        )
        # Merge ramp (right side joining main)
        jx = min(max(merge_tip_px, margin + 40), width - margin - 40)
        chunks.append(
            f"<path d='M {jx + 55} {y_lo + 12} Q {jx + 15} {(y_lo + y_hi) / 2}, {merge_start_px} {y_hi + 35}' "
            f"fill='none' stroke='#0ea5e9' stroke-width='4' stroke-linecap='round' opacity='0.75' filter='url(#laneGlow_{uid})'/>"
        )
        chunks.append(
            f"<path d='M {jx + 72} {y_lo + 6} Q {jx + 22} {(y_lo + y_hi) / 2 - 5}, {merge_start_px + 18} {y_hi + 38}' "
            f"fill='none' stroke='#38bdf8' stroke-width='2' stroke-linecap='round' opacity='0.55'/>"
        )
    elif scenario == "scenario_03_ambulance" and (state.get("road_layout") or {}).get("kind") == "three_lane":
        lx0, lx1 = road_x_left(0.02), road_x_left(0.98)
        rx0, rx1 = road_x_right(0.02), road_x_right(0.98)
        y_lo = road_top + road_h - 8
        y_hi = road_top + 28
        chunks.append(
            f"<path d='M {lx0} {y_lo} L {lx1} {y_hi} M {rx0} {y_lo} L {rx1} {y_hi}' "
            f"stroke='#0ea5e9' stroke-width='5' stroke-linecap='round' opacity='0.95' filter='url(#laneGlow_{uid})'/>"
        )
        for u in (1.0 / 3.0, 2.0 / 3.0):
            xb = lx0 + u * (rx0 - lx0)
            xt = lx1 + u * (rx1 - lx1)
            chunks.append(
                f"<path d='M {xb} {y_lo} L {xt} {y_hi}' stroke='#38bdf8' stroke-width='3' stroke-linecap='round' "
                f"opacity='0.85' filter='url(#laneGlow_{uid})'/>"
            )
        chunks.append(
            f"<text x='{margin + 6}' y='{y_hi + 14}' font-size='10' fill='#64748b' font-family='Segoe UI, system-ui, sans-serif'>"
            f"3-lane · yield lane for AMB</text>"
        )
    else:
        lx0, lx1 = road_x_left(0.02), road_x_left(0.98)
        rx0, rx1 = road_x_right(0.02), road_x_right(0.98)
        y_lo = road_top + road_h - 8
        y_hi = road_top + 28
        chunks.append(
            f"<path d='M {lx0} {y_lo} L {lx1} {y_hi} M {rx0} {y_lo} L {rx1} {y_hi}' "
            f"stroke='#0ea5e9' stroke-width='5' stroke-linecap='round' opacity='0.95' filter='url(#laneGlow_{uid})'/>"
        )
        # Center lane assist line
        mx0 = (lx0 + rx0) / 2
        mx1 = (lx1 + rx1) / 2
        dash = 14
        off = (avg_x * 12) % (dash * 2)
        chunks.append(
            f"<path d='M {mx0} {y_lo} L {mx1} {y_hi}' stroke='#334155' stroke-width='2' stroke-dasharray='{dash} {dash}' "
            f"stroke-dashoffset='{off}' opacity='0.5'/>"
        )

    # Vanishing highlight
    chunks.append(
        f"<ellipse cx='{width / 2}' cy='{vanish_y}' rx='{width * 0.18}' ry='10' fill='#1e3a5f' opacity='0.25'/>"
    )

    if state.get("collision"):
        chunks.append("<rect x='0' y='0' width='100%' height='100%' fill='#7f1d1d' opacity='0.22'/>")
        chunks.append(
            f"<text x='{width / 2}' y='200' text-anchor='middle' font-size='26' font-weight='600' "
            f"font-family='Segoe UI, system-ui, sans-serif' fill='#fecaca'>COLLISION</text>"
        )

    # Cars: draw in order so nearer (lower y_px) paint last
    car_entries: list[tuple[float, float, int, dict[str, Any]]] = []
    lane_center_y = road_top + road_h * 0.72
    for car_id in sorted(vehicles.keys(), key=int):
        v = vehicles[car_id]
        x_pos = _scale_x(float(v.get("x", 0.0)), world_min, world_max, width, margin)
        y_off = float(v.get("y", 0.0)) * 22.0
        cy = lane_center_y + y_off * 0.35 + (float(car_id) * 1.2)
        car_entries.append((cy, x_pos, int(car_id), v))
    car_entries.sort(key=lambda t: t[0])

    for cy, x_pos, car_id, v in car_entries:
        if x_pos < -120 or x_pos > width + 120:
            continue
        vel = float(v.get("velocity", 0.0))
        ptype = str(v.get("path_type", "straight"))
        _draw_topdown_car(
            chunks,
            x_pos,
            cy,
            car_id,
            vel,
            ptype,
            uid * 17 + car_id,
            is_lead=(car_id == 0),
            vehicle_role=str(v.get("vehicle_role", "passenger")),
        )

    # Footer gaps / progress
    fy = height - 36
    if 0 in vehicles and 1 in vehicles:
        gap_01 = float(vehicles[0]["x"] - vehicles[1]["x"] - vehicles[1].get("length", 4.5))
        chunks.append(
            f"<text x='{margin}' y='{fy}' font-size='11' font-family='Segoe UI, system-ui, sans-serif' fill='#64748b'>Gap lead→1: {gap_01:.1f} m</text>"
        )
    if 1 in vehicles and 2 in vehicles:
        gap_12 = float(vehicles[1]["x"] - vehicles[2]["x"] - vehicles[2].get("length", 4.5))
        chunks.append(
            f"<text x='{margin + 200}' y='{fy}' font-size='11' font-family='Segoe UI, system-ui, sans-serif' fill='#64748b'>Gap 1→2: {gap_12:.1f} m</text>"
        )

    pb_w = width - 2 * margin
    progress = float(state.get("timestep", 0))
    max_steps = float(state.get("max_steps", 80))
    pr = min(progress / max(max_steps, 1.0), 1.0)
    chunks.append(
        f"<rect x='{margin}' y='{height - 18}' width='{pb_w}' height='6' rx='3' fill='#1e293b'/>"
    )
    chunks.append(
        f"<rect x='{margin}' y='{height - 18}' width='{pb_w * pr:.1f}' height='6' rx='3' fill='{phase_color}'/>"
    )

    chunks.append("</svg>")
    return "".join(chunks)
