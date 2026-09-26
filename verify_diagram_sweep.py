"""Render-sweep: every registered diagram type must produce real SVG
from a representative spec — a final coverage check across the engine."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import diagram_renderer

SAMPLES = {
    "triangle": {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                 "side_lengths": {"AB": 5, "BC": 6}},
    "circle": {"diagram_type": "circle", "points": [{"id": "O"}, {"id": "A"}, {"id": "B"}],
               "center_id": "O"},
    "angle": {"diagram_type": "angle", "vertex": "O",
              "rays": [{"id": "A", "direction_deg": 0}, {"id": "B", "direction_deg": 60}],
              "angle_marks": [{"between": ["A", "B"], "label": "60°"}]},
    "parallel_lines": {"diagram_type": "parallel_lines", "lines": ["l", "m"],
                       "transversal_label": "t",
                       "angle_marks": [{"line": "l", "position": "bottom_right", "label": "70°"}]},
    "coordinate_plot": {"diagram_type": "coordinate_plot",
                        "points": [{"id": "A", "x": -2, "y": 3}, {"id": "B", "x": 4, "y": -1}],
                        "segments": [{"from": "A", "to": "B"}]},
    "number_line": {"diagram_type": "number_line", "range_min": 0, "range_max": 3,
                    "marked_point": {"value": 1.5, "label": "1.5"}},
    "square_root_spiral": {"diagram_type": "square_root_spiral", "steps": 4},
    "quadrilateral": {"diagram_type": "quadrilateral", "shape": "parallelogram",
                      "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}],
                      "diagonals": ["AC"]},
    "trigonometry": {"diagram_type": "trigonometry", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                     "right_angle_at": "A", "dimensions": {"adjacent": 4, "opposite": 3},
                     "angle_at": "B"},
    "statistics": {"diagram_type": "statistics", "chart_type": "bar",
                   "categories": ["২০২০", "২০২১"], "values": [45, 60]},
    "surface_area_volume": {"diagram_type": "surface_area_volume", "solid": "cylinder",
                            "dimensions": {"radius": 4, "height": 12}},
    "construction": {"diagram_type": "construction",
                     "initial_points": [{"id": "B", "x": 0, "y": 0}, {"id": "C", "x": 7, "y": 0}],
                     "construction_steps": [
                         {"type": "line_segment", "from": "B", "to": "C"},
                         {"type": "arc", "center": "B", "radius": 5, "intersection_id": "A"},
                         {"type": "arc", "center": "C", "radius": 6, "intersection_id": "A"},
                         {"type": "join", "from": "A", "to": "B"},
                         {"type": "join", "from": "A", "to": "C"}]},
    "polynomial_long_division": {"diagram_type": "polynomial_long_division",
                                 "divisor": "x + 1", "quotient": "x + 2",
                                 "dividend": "x² + 3x + 2",
                                 "steps": [{"subtract": "x² + x", "remainder": "2x + 2"}]},
    "transformation": {"diagram_type": "transformation",
                       "shape": [{"id": "A", "x": 1, "y": 1}, {"id": "B", "x": 4, "y": 1}],
                       "transformation": {"type": "reflection", "axis": "x-axis"}},
    "function_graph": {"diagram_type": "function_graph",
                       "function": {"kind": "exponential", "base": 2}},
    "probability": {"diagram_type": "probability", "mode": "tree",
                    "stages": [{"branches": [{"label": "H", "probability": "1/2"},
                                             {"label": "T", "probability": "1/2"}]}]},
    "venn_diagram": {"diagram_type": "venn_diagram",
                     "sets": [{"id": "A", "label": "A", "elements": ["1", "2"]},
                              {"id": "B", "label": "B", "elements": ["2", "3"]}]},
    "elevation_depression": {"diagram_type": "elevation_depression", "mode": "elevation",
                             "points": {"eye": "A", "target": "B", "foot": "C"},
                             "angle_deg": 30, "known_side": "horizontal", "known_value": 20},
    "elevation_two_point": {"diagram_type": "elevation_two_point", "layout": "same_side",
                            "points": {"near": "C", "far": "D", "foot": "B", "target": "A"},
                            "near_angle_deg": 60, "far_angle_deg": 30, "distance_between": 20},
    "bearing": {"diagram_type": "bearing", "start": {"id": "O"},
                "legs": [{"to": {"id": "A"}, "bearing_deg": 60, "distance": 5}]},
    "unit_circle": {"diagram_type": "unit_circle", "angle_deg": 60},
    "circle_line_intersection": {"diagram_type": "circle_line_intersection", "radius": 5,
                                 "line": {"through": [{"x": -10, "y": 0}, {"x": 10, "y": 2}]}},
    "solid_net": {"diagram_type": "solid_net", "solid": "cuboid",
                  "dimensions": {"length": 6, "width": 4, "height": 3}},
    "cross_section": {"diagram_type": "cross_section", "solid": "cone",
                      "cut": "parallel_to_base", "dimensions": {"radius": 6, "height": 12},
                      "cut_height_from_apex": 3},
    "circle_sector": {"diagram_type": "circle_sector", "radius": 7, "angle_deg": 60,
                      "mode": "sector"},
    "composite_shaded_region": {"diagram_type": "composite_shaded_region",
                                "composite_type": "circle_in_square", "side": 10},
    "successive_magnification": {"diagram_type": "successive_magnification", "value": "3.76"},
    "rectilinear_composite": {"diagram_type": "rectilinear_composite",
                              "pieces": [{"x": 0, "y": 0, "width": 8, "height": 2},
                                         {"x": 0, "y": 2, "width": 3, "height": 4}],
                              "unit": "cm"},
}

import diagram_plugin_registry as reg
all_types = sorted(reg.list_plugins())
# built-in (non-plugin) types live in diagram_renderer's own dispatch:
BUILTINS = ["triangle", "circle", "angle", "parallel_lines", "coordinate_plot",
            "number_line", "square_root_spiral"]
all_types += [t for t in BUILTINS if t not in all_types]

missing_samples = []
ok = fail = 0
for t in all_types:
    spec = SAMPLES.get(t)
    if spec is None:
        missing_samples.append(t)
        continue
    try:
        svg = diagram_renderer.render_diagram(spec)
        if svg and "<svg" in svg:
            ok += 1
            print(f"[render OK] {t}")
        else:
            fail += 1
            print(f"[EMPTY]     {t} -> renderer returned no SVG")
    except Exception as e:
        fail += 1
        print(f"[CRASH]     {t} -> {e}")

if missing_samples:
    print(f"\n(no sample spec for: {', '.join(missing_samples)})")
print(f"\n{ok} rendered, {fail} failed, of {len(all_types)} registered types")
sys.exit(1 if fail or missing_samples else 0)
