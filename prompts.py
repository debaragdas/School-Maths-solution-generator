"""
prompts.py — the single prompt used per exercise. Gemini's job here is
strictly: read the exercise PDF, find every question, solve every
question, return structured JSON. It never designs a page, never
writes HTML, never writes SVG.

Uses plain string substitution (str.replace on sentinel tokens), not
str.format() — the JSON schema below can contain any number of literal
braces without ever risking a KeyError.

Also contains build_correction_prompt() — the Human Review workflow's
"Wrong AI Output Correction" capability (see correction_engine.py). It
follows the exact same sentinel-substitution / JSON-escaping / MathJax
rules as the main solve prompt below, scoped to exactly one question.
"""
import json

DIAGRAM_SPEC_SCHEMA = """
"diagram_spec": null OR ONE OF:

  {
    "diagram_type": "triangle" | "circle" | "angle" | "parallel_lines" | "coordinate_plot"
                     | "quadrilateral" | "trigonometry" | "statistics" | "surface_area_volume"
                     | "construction",
    "points": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}, ...],
                                              // used for "triangle" and "circle". For "angle" and
                                              // "parallel_lines" see their own fields further below
                                              // instead of "points".
                                              //
                                              // ONLY when diagram_type is "coordinate_plot": each
                                              // point MUST also include its exact "x" and "y"
                                              // numeric coordinates, e.g. {"id": "A", "label": "A",
                                              // "x": -3, "y": 5} — the renderer plots points at
                                              // their literal (x, y) position to scale; it does
                                              // NOT infer position from list order, so an "x"/"y"
                                              // pair is required for every point in a
                                              // coordinate_plot spec.
    "canvas_size": [400, 300],                // OPTIONAL, override default canvas size.

    // ONLY when diagram_type is "circle", these two OPTIONAL fields
    // unlock two extremely common real circle-question patterns that
    // plain "points" alone cannot represent:
    "center_id": "O",                        // OPTIONAL. If one of the declared "points" is
                                              // explicitly the circle's CENTER (almost every real
                                              // circle question says "O is the centre..." or
                                              // similar), name its id here so it renders at the
                                              // true center with a real radius line to every other
                                              // point — NOT as just another point sitting ON the
                                              // circle's edge like an inscribed-angle point would.
                                              // Omit this field entirely for questions that are
                                              // genuinely only about points ON the circle (e.g.
                                              // "ABCD is a cyclic quadrilateral") with no named
                                              // center — the diagram falls back to exactly the
                                              // same all-points-on-the-circumference rendering it
                                              // always has.
    "tangent_from": {                        // OPTIONAL. Only for "a tangent is drawn from an
      "external_point": "P",                 // external point P to a circle centred at O" style
      "tangent_points": ["A", "B"]           // questions. Requires "center_id" to also be set.
    },                                        // "external_point" and both "tangent_points" MUST
                                              // also be declared in "points" above (their exact
                                              // x/y is computed deterministically from real
                                              // tangent-line geometry — never estimate or place
                                              // them yourself). Use ONE tangent point ("PA") if the
                                              // question only asks about a single tangent, or both
                                              // ("PA" and "PB") for the classic "two tangents from
                                              // an external point" construction.

                                              // coordinate_plot spec, taken directly from what the
                                              // question states or from the book's own figure if
                                              // one is shown in the exercise PDF you were given.
                                              // Optionally add "show_projection": true to a point
                                              // to draw dashed helper perpendiculars from it down
                                              // to the x-axis and across to the y-axis (use this
                                              // whenever the question is about READING a point's
                                              // coordinates off the axes, not for every point).

    // ONLY when diagram_type is "coordinate_plot", these two OPTIONAL
    // arrays add the rest of the coordinate-geometry toolkit — every
    // id used below MUST already be declared (with x/y) in "points":

    "segments": [                            // a straight segment between two declared points —
      {"from": "A", "to": "B",               // use for "find the distance AB", "M is the midpoint
       "dashed": false,                      // of AB", "join AB", etc. Do NOT use this for a full
       "show_midpoint": true,                // line that should extend across the whole grid —
       "midpoint_label": "M",                // that is "lines" below instead.
       "show_length": true}                  // show_length prints the computed distance AB next
    ],                                        // to the segment; show_midpoint marks and labels the
                                              // midpoint with equal-length tick marks on each half.
    "lines": [                               // a full straight line through two declared points,
      {"through": ["A", "B"], "label": "l"}  // extended to the edge of the plot in both directions
    ],                                        // — use for "the line through A and B", "the graph of
    "legend": [                              // OPTIONAL list of labels for different lines/series
      {"label": "Line AB: y = 2x + 1"}       // shown in the top-right corner.
    ],
                                              // the linear equation...", "show A, B, C are
                                              // collinear", etc.
    "equal_marks": [["AB", "AC"]],           // side-pairs marked equal (tick marks) — also
                                              // works for an altitude segment like "BE" once
                                              // E is declared as that altitude's foot below
    "altitudes": [                           // OPTIONAL — a perpendicular dropped from a
      {"from": "B", "to_side": "AC", "foot": "E"}   // vertex onto the OPPOSITE side, landing
    ],                                        // at "foot". The renderer computes the foot's
                                              // EXACT position by real perpendicular
                                              // projection onto that side and draws its own
                                              // right-angle mark there automatically — you
                                              // do NOT place "foot" yourself, but "foot" MUST
                                              // still be listed in "points" like any other
                                              // point used in the figure. Use this whenever a
                                              // question involves an altitude/perpendicular
                                              // from a vertex to a NON-adjacent side (e.g. "BE
                                              // and CF are altitudes of triangle ABC") — do
                                              // NOT just add E/F as a bare extra point, since
                                              // without an altitudes[] entry the renderer has
                                              // no way to know E doesn't just sit on the base.
    "right_angle_at": "P" or ["P","Q"] or null,  // right-angle box AT one of the figure's own
                                              // points that is NOT already an altitude foot
                                              // (those get their mark automatically from
                                              // altitudes[] above) — e.g. the triangle is
                                              // itself right-angled at P. Accepts one id or a
                                              // list if more than one such mark is needed.
    "cevians": [                             // OPTIONAL — a NON-perpendicular line from a
      {"from": "B", "to_side": "AC", "foot": "D"}  // vertex to a point on the OPPOSITE side —
    ],                                        // use this instead of "altitudes" for an angle
                                              // BISECTOR or MEDIAN foot (anything that is
                                              // NOT a right-angle perpendicular — using
                                              // "altitudes" here would draw a wrong
                                              // right-angle mark). Same rules as altitudes:
                                              // "foot" MUST also be listed in "points", and
                                              // you do not place its position yourself.
    "cevian_intersection_label": "O",        // OPTIONAL — set this whenever the question
                                              // names the point where TWO cevians (e.g. two
                                              // angle bisectors, or a cevian and an altitude)
                                              // cross inside the triangle (e.g. "O is the
                                              // point where the bisectors meet"). REQUIRES
                                              // EXACTLY 2 entries in "cevians" above — the
                                              // renderer computes O as their real geometric
                                              // intersection, it is never a separate
                                              // declared point. Omitting this when the proof
                                              // names such a point means that point never
                                              // appears in the diagram at all.
    "join_vertex_to_intersection": "A",       // OPTIONAL — only meaningful together with
                                              // cevian_intersection_label — draws one more
                                              // segment from this vertex to that
                                              // intersection point (e.g. "join A to O").

    // WORKED EXAMPLE (this exact pattern was found rendered WRONG in a
    // real generated PDF — the figure showed two unrelated points D, E
    // sitting on the base instead of an interior intersection point —
    // so follow this shape precisely for this family of question):
    // Question: "In isosceles triangle ABC with AB = AC, the bisectors
    // of ∠B and ∠C meet at O. Join A to O. Show OB = OC and AO bisects
    // ∠A." The bisector of ∠B is a cevian FROM B landing somewhere on
    // the OPPOSITE side AC (not on BC); the bisector of ∠C is a cevian
    // FROM C landing on the OPPOSITE side AB. Correct spec:
    //   "points": [{"id":"A"},{"id":"B"},{"id":"C"}],
    //   "cevians": [{"from":"B","to_side":"AC","foot":"_bf1"},
    //               {"from":"C","to_side":"AB","foot":"_bf2"}],
    //   "cevian_intersection_label": "O",
    //   "join_vertex_to_intersection": "A"
    // Do NOT model this as two arbitrary extra points on side BC (that
    // draws the wrong construction entirely — there is no interior
    // intersection point in that shape, which is exactly the mismatch
    // that was found). "foot" ids for cevians that are only there to
    // make O computable (never referred to by name in the proof) can
    // use a throwaway id like "_bf1" — they still MUST be listed in
    // "points" like any other point, but never need to appear in the
    // written proof text.
    "extra_labels": ["not to scale"],        // any small caption text under the diagram —
                                              // NEVER put a figure number/reference here
                                              // (e.g. "চিত্ৰ 7.33") — this is a GENERATED
                                              // diagram, not the book's own figure, and
                                              // labelling it as if it were would mislead the
                                              // student. Use this only for a genuine note
                                              // like scale/units, never a figure citation.

    "side_lengths": {                        // OPTIONAL. Include ONLY when the question's
      "AB": 5, "BC": 6, "CA": 7              // OWN text states an actual numeric length for
    },                                        // one or more sides — NEVER estimate or invent
                                              // a "typical" value. Keys are any two of this
                                              // triangle's own point ids, either order
                                              // ("AB" and "BA" mean the same side). Give as
                                              // many as the question states — 1, 2, or all 3
                                              // — the renderer computes a to-scale diagram
                                              // whenever enough of these (or angles_deg
                                              // below) are given to uniquely determine the
                                              // triangle; otherwise it draws the usual
                                              // schematic (not-to-scale) figure exactly as
                                              // before, so omitting this is always safe.
    "angles_deg": {                          // OPTIONAL. Same rule: include ONLY an angle
      "A": 60, "B": 70                       // value the question's own text actually
    }                                         // states at a named vertex, never a guess.
  }

  OR, for an "angle" figure (a single angle, a linear pair, vertically
  opposite angles, or an angle bisector — anything centered on rays
  from ONE vertex):

  {
    "diagram_type": "angle",
    "vertex": "O",
    "rays": [                                // EVERY ray from the vertex that appears anywhere in
      {"id": "A", "direction_deg": 0},       // this question, each with its OWN direction in
      {"id": "B", "direction_deg": 60}       // degrees (0 = pointing right, 90 = straight up,
    ],                                        // increasing counter-clockwise — exactly like a
                                              // protractor placed with 0° on the right). Two rays
                                              // exactly 180° apart automatically look like ONE
                                              // straight line through O — this is how you draw a
                                              // "linear pair" or "angles on a straight line":
                                              // give the two outer rays directions 180° apart and
                                              // any ray(s) between them their own direction.
    "angle_marks": [                         // one entry per angle that needs an arc + label —
      {"between": ["A", "B"], "label": "60°"} // NOT every possible pair, only the ones the
    ],                                        // question actually names/asks about. "label" can be
                                              // a known value ("60°") or an unknown ("x°").
    "bisector": {                            // OPTIONAL — only when the question involves an
      "of": ["A", "B"], "to": "D"            // angle bisector: draws a dashed ray splitting the
    }                                         // angle between the two named rays exactly in half,
                                              // landing on a new labelled point "to".
  }

  OR, for a "parallel_lines" figure (two parallel lines cut by a
  transversal — corresponding/alternate/co-interior angle questions):

  {
    "diagram_type": "parallel_lines",
    "lines": ["l1", "l2"],                   // the two parallel lines' own names/labels, in the
                                              // SAME order the question uses (l1 drawn above l2)
    "transversal_label": "t",                // OPTIONAL label for the transversal line
    "angle_marks": [                         // one entry per angle the question actually names —
      {"line": "l1", "position": "bottom_right", "label": "70°"},
      {"line": "l2", "position": "top_right", "label": "70°"}
    ]                                         // "position" is which of the 4 angles at that
                                              // line's own crossing with the transversal:
                                              // top_left/top_right/bottom_left/bottom_right,
                                              // meaning above/below the parallel line and
                                              // left/right of the transversal — figure this out
                                              // from the question exactly as a student reading
                                              // the printed figure would.
  }

  OR, for placing a number on a number line (integers, decimals via
  successive magnification, or an irrational number via the Pythagorean
  spiral construction — e.g. "show √5 on the number line"):

  {
    "diagram_type": "number_line",
    "range_min": 0, "range_max": 4,          // integer bounds the line should span
    "marked_point": {"value": 2.236, "label": "√5"},  // the point being placed
    "construction": {                         // OPTIONAL — only for a Pythagorean/geometric
      "base_point": 2, "perpendicular_length": 1      // construction (right triangle + arc), not
    }                                                   // needed for a plain decimal-magnification question
  }

  OR, for a "square root spiral" construction (chaining right triangles
  from an origin so each hypotenuse is √2, √3, √4, ...):

  {
    "diagram_type": "square_root_spiral",
    "steps": 6   // how many points (P1..Pn) to draw, i.e. up to √(steps+1)
  }

  OR, for a "quadrilateral" figure (parallelogram, rectangle, rhombus,
  square, trapezium, or a general quadrilateral — area/property/proof
  questions about 4-sided figures):

  {
    "diagram_type": "quadrilateral",
    "shape": "parallelogram" | "rectangle" | "rhombus" | "square" | "trapezium" | "general",
    "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}],  // EXACTLY 4, in
                                              // order going around the shape (so the
                                              // edges are AB, BC, CD, DA) — matching the
                                              // book's own labeling order exactly.
    "diagonals": ["AC"],                     // OPTIONAL — which diagonal(s) to draw (dashed)
    "diagonal_intersection_label": "O",      // OPTIONAL — set this whenever the question
                                              // refers to the point where the diagonals
                                              // cross (e.g. "O is the intersection of the
                                              // diagonals", proofs using ∠OBA, ∠OAB, etc.).
                                              // REQUIRES both diagonals ("AC"-type AND
                                              // "BD"-type, i.e. both pairs of opposite
                                              // corners) to be listed in "diagonals" above
                                              // — the renderer computes O as their actual
                                              // geometric intersection, it is never a
                                              // separate declared point. If the question
                                              // names this point (even just "O"), you MUST
                                              // set this field — omitting it means the
                                              // point the proof discusses never appears in
                                              // the diagram at all.
    "equal_marks": [["AB", "CD"]],           // same meaning as in "triangle" above
    "right_angle_at": "A" or ["A", "B"] or null,  // right-angle box at named vertices
    "extra_labels": ["not to scale"]         // any small caption text under the diagram —
                                              // NEVER put a figure number/reference here
                                              // (e.g. "চিত্ৰ 7.33") — this is a GENERATED
                                              // diagram, not the book's own figure, and
                                              // labelling it as if it were would mislead the
                                              // student. Use this only for a genuine note
                                              // like scale/units, never a figure citation.
  }

  OR, for a "trigonometry" figure (a right triangle purpose-built for a
  "find sin/cos/tan of angle..." style question — use this instead of
  "triangle" whenever the question is specifically about a
  trigonometric ratio, not a general geometry proof):

  {
    "diagram_type": "trigonometry",
    "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}], // EXACTLY 3
    "right_angle_at": "A",                   // which vertex is the 90° corner, e.g. "A"
    "dimensions": {                          // the TWO sides that define the right
      "adjacent": 4, "opposite": 3           // angle — the renderer computes the
    },                                        // hypotenuse itself. Use the EXACT
                                              // numbers from the question.
    "angle_at": "B",                         // OPTIONAL — which of the OTHER two
                                              // vertices has the angle the question
                                              // asks about (must not be the same as
                                              // right_angle_at)
    "angle_value_deg": 30,                   // OPTIONAL numeric value, if known
    "angle_label": "θ",                      // OPTIONAL symbolic label instead of/alongside a value
    "side_labels": {"AB": "5 cm", "BC": "13 cm", "AC": "x"},  // OPTIONAL — text to
                                              // print along each named side (a given
                                              // length, or an unknown like "x")
    "show_ratio": "sin" | "cos" | "tan" | null  // OPTIONAL — prints the matching ratio
                                              // definition as a small caption
  }

  OR, for a "statistics" figure (a bar graph or histogram for a
  frequency-distribution question — "draw a bar graph for the
  following data", "represent this frequency table graphically"):

  {
    "diagram_type": "statistics",
    "chart_type": "bar" | "histogram" | "pie" | "frequency_polygon" | "ogive",
    "categories": ["2020", "2021", "2022"],  // For bar/pie: the discrete category labels.
                                              // For histogram: the class interval labels ("10-20").
                                              // For frequency_polygon: the class mark (midpoint).
                                              // For ogive: the class boundary (upper for "less
                                              // than", lower for "more than").
    "values": [45, 60, 38],                  // the actual numeric height for each bar —
                                              // MUST be the same length as "categories",
                                              // taken directly from the question's data
    "ogive_type": "less_than" | "more_than", // ONLY for chart_type: "ogive"
    "x_label": "Year",                       // OPTIONAL axis captions
    "y_label": "Number of students"
  }

  OR, for a "surface_area_volume" figure (a labelled 3D solid — cuboid,
  cube, cylinder, cone, or sphere — for a surface-area/volume question):

  {
    "diagram_type": "surface_area_volume",
    "solid": "cuboid" | "cube" | "cylinder" | "cone" | "sphere",
    "dimensions": {                          // ONLY the keys that solid actually needs:
      "length": 10, "width": 5, "height": 7  //   cuboid: length, width, height
      // "side": 8                            //   cube: side
      // "radius": 4, "height": 12            //   cylinder / cone: radius, height
      // "radius": 6                          //   sphere: radius
    },                                        // every value here is drawn as an actual
                                              // labelled dimension on the figure — use
                                              // the EXACT numbers the question gives, not
                                              // placeholders.
    "labels": {"height": "h = 12 cm"}        // OPTIONAL — override the auto-generated
                                              // label text for any dimension key above
  }

  OR, for a "construction" diagram (a CONSTRUCTION (অংকন) question — see
  the CONSTRUCTION QUESTIONS rule further below — showing the actual
  compass-and-ruler build sequence, not just the finished figure):

  {
    "diagram_type": "construction",
    "initial_points": [{"id": "B", "x": 0, "y": 0}, {"id": "C", "x": 7, "y": 0}],
                                              // the point(s) already fixed before any construction
                                              // step runs — numeric x/y in the SAME unit as the
                                              // question (e.g. cm), to scale relative to each other.
                                              // Usually just the two ends of the base segment.
    "construction_steps": [                  // ONE entry per physical action, in the EXACT order
                                              // the "steps" text describes performing it:
      {"type": "line_segment", "from": "B", "to": "C", "label": "7 cm"},
                                              //   draw the base segment between two already-known
                                              //   points (usually the first step)
      {"type": "arc", "center": "B", "radius": 5, "intersection_id": "A"},
                                              //   swing a compass arc from an already-known center;
                                              //   "intersection_id" names the NEW point this arc
                                              //   helps locate (not yet known before this step)
      {"type": "arc", "center": "C", "radius": 6, "intersection_id": "A"},
                                              //   a second arc sharing the same intersection_id as
                                              //   the previous one is how a third vertex (e.g. the
                                              //   apex of a triangle built from three known side
                                              //   lengths) is actually found — the renderer solves
                                              //   the two arcs' intersection deterministically; do
                                              //   NOT try to give "A" numeric coordinates yourself
                                              //   for a point defined this way
      {"type": "join", "from": "A", "to": "B"},
      {"type": "join", "from": "A", "to": "C"}
                                              //   "join" draws a straight line to/from a point that
                                              //   was found via a construction step (an arc's
                                              //   intersection_id) rather than given directly —
                                              //   use "line_segment" instead when both ends were
                                              //   already known from the start (e.g. the base)
    ]
  }

  OR, for a "polynomial_long_division" diagram (dividing one polynomial
  by another using the standard long-division layout — divisor outside
  the bracket, quotient on top, each step's subtraction under its own
  rule line). Use this whenever a question asks to divide/verify the
  division algorithm for polynomials — do NOT try to hand-draw this
  layout with LaTeX \require{enclose}/\begin{array} inside a solution
  step's text; it renders inconsistently. Every text field here is
  PLAIN text (not LaTeX) — write exponents as x², x³ (Unicode
  superscripts), π as π, √ as √, etc., exactly like every other
  diagram type's labels in this schema already do:

  {
    "diagram_type": "polynomial_long_division",
    "divisor": "x + π",                      // exactly as it appears outside the bracket
    "quotient": "x² + (3-π)x + (3-3π+π²)",    // the full quotient, shown above the bar
    "dividend": "x³ + 3x² + 3x + 1",          // the original polynomial being divided
    "steps": [                                // ONE entry per subtract-then-bring-down row,
                                              // in order, top to bottom
      {"subtract": "x³ + πx²",                // what's subtracted FROM the row above it
        "remainder": "(3-π)x² + 3x + 1"},    // what's left after that subtraction
      {"subtract": "(3-π)x² + (3-π)πx",
        "remainder": "[3-3π+π²]x + 1"},
      {"subtract": "[3-3π+π²]x + (3-3π+π²)π",
        "remainder": "[1-3π+3π²-π³]"}        // the LAST remainder is the division's final
                                              // remainder (0 for an exact division) — always
                                              // include it as the last step's "remainder",
                                              // never omit the final row
    ]
  }

  OR, for a "transformation" diagram (reflection, rotation, or
  translation of a polygon on the Cartesian plane). Give ONLY the
  ORIGINAL shape's coordinates and the transformation's parameters —
  never compute or supply the image's (reflected/rotated/translated)
  coordinates yourself; the renderer computes every image point
  deterministically so it is always exactly correct:

  {
    "diagram_type": "transformation",
    "shape": [{"id": "A", "x": 1, "y": 1}, {"id": "B", "x": 4, "y": 1}, {"id": "C", "x": 1, "y": 3}],
    "transformation": {
      "type": "reflection",                  // "reflection" | "rotation" | "translation"
      "axis": "x-axis"                        // ONLY for "reflection": "x-axis" | "y-axis" | "y=x"
                                              // | "y=-x" | "line" (give "through": [[x1,y1],[x2,y2]]
                                              // for an arbitrary line, two points on that line)
      // "center": [0, 0], "angle_deg": 90,   // ONLY for "rotation": center point, angle, and
      // "direction": "counterclockwise"      // "direction": "clockwise" | "counterclockwise"
      // "vector": [3, -2]                    // ONLY for "translation": [dx, dy]
    }
  }

  OR, for a "function_graph" diagram (exponential, logarithmic,
  absolute-value, or piecewise graphs — use "coordinate_plot" instead
  for plotting a handful of discrete points or a straight line).
  Give ONLY the function's kind and numeric parameters — never sample
  (x, y) points yourself; the renderer plots the exact mathematical
  curve:

  {
    "diagram_type": "function_graph",
    "function": {
      "kind": "exponential",                 // "exponential" | "logarithmic" | "absolute_value"
                                              // | "piecewise"
      "base": 2, "a": 1, "c": 0              // exponential: y = a*base^x + c (a, c optional, default 1, 0)
      // "base": 2, "a": 1, "c": 0           // logarithmic: y = a*log_base(x) + c (x > 0 only)
      // "a": 1, "h": 0, "k": 0              // absolute_value: y = a*|x - h| + k
      // "pieces": [{"expr": "linear", "a": 1, "b": 0, "domain": [-5, 0]},
      //            {"expr": "constant", "c": 2, "domain": [0, 5]}]
                                              // piecewise: each piece is "linear" (y = a*x + b) or
                                              // "constant" (y = c) over its own [lo, hi] "domain"
    },
    "x_range": [-5, 5],                      // OPTIONAL — sketch window; omit for a sensible default
    "y_range": [-5, 5],                      // OPTIONAL
    "label": "y = 2^x"                       // OPTIONAL caption
  }

  OR, for a "probability" diagram — mode "tree" for a sequential
  probability tree (e.g. two coin tosses, drawing balls with/without
  replacement) or mode "sample_space" for a grid (e.g. two dice).
  Give ONLY each branch's probability as a fraction string — never
  compute or supply a leaf's combined probability yourself; the
  renderer multiplies exactly with fractions.Fraction:

  {
    "diagram_type": "probability",
    "mode": "tree",
    "stages": [                              // one entry per sequential event, in order
      {"branches": [{"label": "H", "probability": "1/2"}, {"label": "T", "probability": "1/2"}]},
      {"branches": [{"label": "H", "probability": "1/2"}, {"label": "T", "probability": "1/2"}]}
    ]
  }

  OR:

  {
    "diagram_type": "probability",
    "mode": "sample_space",
    "rows": ["1", "2", "3", "4", "5", "6"],  // one axis's values (e.g. one die's faces)
    "cols": ["1", "2", "3", "4", "5", "6"],  // the other axis's values
    "combine": "sum"                         // "sum" | "product" — auto-computes each cell from
                                              // its row+col value. OR give "cells" (a 2D list, one
                                              // row per "rows" entry) directly instead of "combine"
                                              // for non-numeric outcomes like coin tosses ("HH", "HT"...)
  }

  OR, for a "venn_diagram" diagram (set theory — 2 or 3 sets). Give
  ONLY each set's own elements — never say which overlapping region an
  element belongs in yourself; the renderer places every element from
  set membership alone, so a shared element can never be drawn twice
  or in the wrong region:

  {
    "diagram_type": "venn_diagram",
    "sets": [
      {"id": "A", "label": "A", "elements": ["1", "2", "3", "4"]},
      {"id": "B", "label": "B", "elements": ["3", "4", "5", "6"]}
    ],
    "universal_set": {"label": "U"},         // OPTIONAL — draws the bounding rectangle
    "shaded_region": "intersection"          // OPTIONAL, 2-set only: "union" | "intersection"
                                              // | "A_only" | "B_only" | "complement_A"
                                              // | "complement_B" | "symmetric_difference"
  }

  OR, for an "elevation_depression" diagram (angle of elevation of the
  top of a tower/tree/cliff seen from the ground, or angle of
  depression from the top of a cliff/building looking down at an
  object). Give ONLY the angle and ONE known leg — never compute or
  supply the other leg yourself; the renderer solves it with tan() and
  draws the figure to the correct proportions (the unknown leg is
  drawn as "?" rather than its numeric value by default, so the
  diagram never gives away the exercise's own answer):

  {
    "diagram_type": "elevation_depression",
    "mode": "elevation",                     // "elevation" | "depression"
    "points": {"eye": "A", "target": "B", "foot": "C"},
                                              // "eye" = the observer (on the ground for elevation,
                                              // at the top for depression); "target" = the point
                                              // being sighted (top of the object for elevation,
                                              // the ground object for depression); "foot" = the
                                              // right-angle vertex directly below the elevated point
    "angle_deg": 30,                         // strictly between 0 and 90
    "known_side": "horizontal",              // "horizontal" | "vertical" — which leg's length is given
    "known_value": 20
    // "show_unknown_value": false,          // OPTIONAL, default false — set true only if the
                                              // question itself already states BOTH legs
    // "unknown_label": "h"                  // OPTIONAL — text shown on the withheld leg (default "?")
  }

  OR, for an "elevation_two_point" diagram — DISTINCT from
  "elevation_depression" above: use this one specifically when the
  question gives TWO different ground points (not one) with TWO
  different angles of elevation to the SAME top, e.g. "from two points
  on the same side of a tower, 20 m apart, the angles of elevation are
  60 degrees and 30 degrees" (points on the same side) or "from two
  points on opposite sides of a tower the angles of elevation are..."
  (foot between the two points). Give ONLY the two angles and the
  distance between the two points — never compute the height or either
  point's own distance from the foot yourself; the renderer solves both
  simultaneous tan() equations and draws the figure to the correct
  proportions (the height is withheld from the rendered label by
  default, same reasoning as elevation_depression above):

  {
    "diagram_type": "elevation_two_point",
    "layout": "same_side",                   // "same_side" | "opposite_sides"
    "points": {"near": "C", "far": "D", "foot": "B", "target": "A"},
                                              // "near" = the point closer to the foot (always sees
                                              // the LARGER angle for same_side); "far" = the other
                                              // point; "foot"/"target" = base/top of the tower
    "near_angle_deg": 60,                    // strictly between 0 and 90; for "same_side" this
                                              // MUST be strictly greater than far_angle_deg
    "far_angle_deg": 30,
    "distance_between": 20
    // "show_height": false                  // OPTIONAL, default false — set true only if the
                                              // question itself already states the height
  }

  OR, for a "bearing" diagram (navigation/survey: a ship, aircraft, or
  surveyor moving on stated compass bearings). Give ONLY each leg's
  bearing (0-360, clockwise from North) and distance — never compute
  a point's actual position yourself; the renderer places every point
  trigonometrically from the chain of legs:

  {
    "diagram_type": "bearing",
    "start": {"id": "O", "label": "O"},
    "legs": [
      {"to": {"id": "A", "label": "A"}, "bearing_deg": 60, "distance": 5},
      {"to": {"id": "B", "label": "B"}, "bearing_deg": 145, "distance": 8}
                                              // OPTIONAL "from_id" if this leg doesn't start from
                                              // the previous leg's "to" point — defaults to it
    ]
  }

  OR, for a "unit_circle" diagram (Class 10 trigonometry: a point on
  the circle of radius 1 at a given angle). Give ONLY the angle — never
  compute (cos, sin) yourself; the renderer computes the point from
  math.cos/math.sin and labels it exactly (a fraction/radical form for
  the standard 30-degree-multiple angles, a decimal otherwise):

  {
    "diagram_type": "unit_circle",
    "angle_deg": 60                          // any real number, standard math convention
    // "show_special_angles": false,         // OPTIONAL — tick every 30-degree angle
    // "quadrant_shading": false             // OPTIONAL — shade quadrant I as a visual aid
  }

  OR, for a "circle_line_intersection" diagram (a line/chord/secant/
  tangent crossing a circle). Give ONLY the circle and the line's own
  definition — never compute or supply the intersection point(s)
  yourself; the renderer solves the exact quadratic and classifies
  secant/tangent/no-intersection:

  {
    "diagram_type": "circle_line_intersection",
    "center": {"x": 0, "y": 0, "label": "O"}, // OPTIONAL, defaults to origin
    "radius": 5,
    "line": {"through": [{"x": -10, "y": 0}, {"x": 10, "y": 0}]}
                                              // OR {"point": {"x":..,"y":..}, "angle_deg": ..}
    // "labels": {"intersection_prefix": "P"} // OPTIONAL, default "P"
  }

  OR, for a "solid_net" diagram (the 2D net/unfolding of a 3D solid —
  "draw the net of...", "which net folds into..."). Give ONLY the
  solid's own dimensions — never lay out or size the individual faces
  yourself; the renderer derives every panel's exact size (e.g. a
  cylinder's curved face is an exact 2*pi*r by h rectangle):

  {
    "diagram_type": "solid_net",
    "solid": "cuboid",                       // "cube" | "cuboid" | "cylinder" | "cone"
    "dimensions": {"length": 6, "width": 4, "height": 3}
                                              // cube: {"side"}; cylinder: {"radius","height"};
                                              // cone: {"radius","slant_height"}
  }

  OR, for a "cross_section" diagram (a plane cutting through a 3D
  solid — "what shape do you get if you cut..."). Give ONLY the solid,
  its dimensions, and which named cut — never compute the resulting 2D
  shape's size yourself; the renderer derives it exactly (e.g. a cone
  cut parallel to its base uses similar triangles):

  {
    "diagram_type": "cross_section",
    "solid": "cone",                         // "cuboid" | "cylinder" | "cone"
    "cut": "parallel_to_base",                // cuboid: "parallel_to_face"; cylinder:
                                              // "parallel_to_base" | "through_axis"; cone:
                                              // "parallel_to_base" | "through_apex_axis"
    "dimensions": {"radius": 6, "height": 12},
    "cut_height_from_apex": 4                // REQUIRED only for cone + "parallel_to_base"
  }

  OR, for a "circle_sector" diagram (Class 10 "Areas Related to
  Circles" — a sector/pie-slice of a circle, or the chord-bounded
  segment next to it: "find the area of the sector...", "find the
  area of the minor/major segment..."). Give ONLY the circle and the
  angle AOB — never compute arc length, sector area, or segment area
  yourself; the renderer derives all of them from the exact closed-
  form formulas and independently re-measures the drawn arc before
  returning any SVG:

  {
    "diagram_type": "circle_sector",
    "center": {"x": 0, "y": 0, "label": "O"}, // OPTIONAL, defaults to origin
    "radius": 7,
    "angle_deg": 60,                         // angle AOB of the MINOR sector/segment, 0 < angle < 360
    "mode": "sector",                        // "sector" | "segment" — which region the question asks about
    "region": "minor"                        // "minor" | "major" — OPTIONAL, default "minor"
    // "shaded": true,                       // OPTIONAL, default true — fill the asked-about region
    // "labels": {"A": "A", "B": "B"}        // OPTIONAL — custom labels for the two radius endpoints
  }

  OR, for a "composite_shaded_region" diagram (Class 10 "combination of
  plane figures" — a shaded region formed by TWO combined figures, not
  one: a circle inscribed in a square, a square inscribed in a circle,
  a right triangle inscribed in a circle with the hypotenuse as
  diameter, or two overlapping circles). Give ONLY the figures' own
  dimensions — never compute the shaded area yourself; the renderer
  derives it from the exact closed-form formula for the chosen
  composite_type:

  {
    "diagram_type": "composite_shaded_region",
    "composite_type": "circle_in_square",    // "circle_in_square" | "square_in_circle" |
                                              // "triangle_in_circle" | "two_overlapping_circles"
    "side": 10,                              // circle_in_square ONLY: the square's side
    // "radius": 5,                          // circle_in_square: OPTIONAL, default side/2 (inscribed);
                                              // square_in_circle/triangle_in_circle/
                                              // two_overlapping_circles: REQUIRED, the circle's radius
    // "angle_deg": 30,                      // triangle_in_circle ONLY: the acute angle of the
                                              // inscribed right triangle at one end of the diameter
    // "distance": 6,                        // two_overlapping_circles ONLY: distance between the
                                              // two equal circles' centers (must be < 2*radius)
    "shaded": "between"                      // circle_in_square/square_in_circle/triangle_in_circle:
                                              // "between" | "circle" | "square"/"triangle";
                                              // two_overlapping_circles: "intersection" | "union" | "petals"
  }

  OR, for a "successive_magnification" diagram (Class 9 Number System —
  locating an irrational/repeating decimal on the number line via
  repeated 10-way zoom-in: "সংখ্যাৰেখাত ৪.২৬২৬ ক সূচিত কৰক", "show how
  successive magnification locates 3.765 on the number line"). Give
  ONLY the exact decimal value AS A STRING — never as a JSON number
  (a JSON number does not round-trip every decimal digit exactly) —
  and never compute or describe the individual magnification rows
  yourself; the renderer derives every row's range, ticks, and
  highlighted digit from the string's own exact decimal digits:

  {
    "diagram_type": "successive_magnification",
    "value": "3.765"                         // STRING, e.g. "3.765" or "4.2626" — 1 to 6 decimal digits
  }

  OR, for a "rectilinear_composite" diagram (Class 6-8 mensuration — a
  COMPOUND rectilinear figure built from axis-aligned rectangles: an
  L-shaped plot, a T-shaped beam cross-section, a step podium, "find the
  area/perimeter of the given figure"). Give ONLY each rectangle PIECE's
  position and size exactly as the question's own dimensions state them,
  in the question's own unit, with the origin at the figure's bottom-left
  and y increasing UPWARDS — never compute area or perimeter yourself;
  the renderer derives the union outline, its exact total area (Σ w×h)
  and perimeter from the pieces alone. Pieces must NOT overlap — split
  an L/T shape into its natural non-overlapping tiles:

  {
    "diagram_type": "rectilinear_composite",
    "pieces": [                              // 1-6 rectangles tiling the figure
      {"x": 0, "y": 0, "width": 8, "height": 2},   // e.g. an L: full-width base...
      {"x": 0, "y": 2, "width": 3, "height": 4}    // ...plus left upright
    ],
    "unit": "cm",                            // OPTIONAL — printed on dimension labels
    "shaded_pieces": [0]                     // OPTIONAL — indices of pieces to shade
                                              // ("all" shades everything; omit for none)
  }

MANDATORY RULE: if the question names or implies ANY geometric figure OR
number-line/spiral construction — a triangle, circle, angle, pair of
lines, quadrilateral, right-triangle trigonometry question, bar
graph/histogram, 3D solid (surface area/volume), placing a number
(integer, decimal, or irrational) on a number line, a reflection/
rotation/translation, an exponential/logarithmic/absolute-value/
piecewise graph, a probability tree or sample space, a Venn diagram, an
angle of elevation/depression (from ONE point — use "elevation_depression")
or from TWO different ground points to the same top (use
"elevation_two_point"), a compass-bearing navigation/survey
question, a unit-circle trigonometry point, a line/chord/secant/tangent
meeting a circle, the net (unfolding) of a 3D solid, a cross-section
cut through a 3D solid, a sector or segment of a circle (minor or
major — "areas related to circles"), a circle/square/triangle inscribed
inside another figure or two overlapping circles ("combination of plane
figures", "shaded region" formed by two combined shapes), locating an
irrational/repeating decimal via successive magnification (repeated
zoomed-in number-line rows), a COMPOUND rectilinear mensuration figure
built from rectangles ("find the area/perimeter of this L/T-shaped
figure" — use "rectilinear_composite"), or a "square
root spiral"
construction (this
includes "prove that...", "show that...", "in the figure...",
"represent √n on the number line", "construct a square root spiral",
"draw a bar graph for...", "draw the net of...", "find the area of the
sector...", "find the area of the minor/major segment...", "find the
area of the shaded region...", "show ... on the number line using
successive magnification...", "what shape do you
get if you cut...", "find the volume/surface area of...") — you MUST return a non-null
diagram_spec for it, using whichever of the shapes above actually
matches the question. A
CONSTRUCTION (অংকন) question — see that rule below — MUST use
diagram_type "construction" specifically, built from the same
line-segment/arc/join actions as its own "steps" text, NOT "triangle"
or another finished-figure type: the diagram must show the build
sequence the student performs, not just the completed result.
Only use null for questions that are purely algebraic/numeric with no
figure, chart, solid, or construction at all (e.g. "simplify 3/4 +
1/2", "express 0.6666... as a fraction"). Do not skip a diagram_spec
for a geometry/construction/chart/solid question just because the
exact coordinates aren't printed in the book — describing the
relationships is all that's needed; the renderer computes the actual
drawing from that.

════════════════════════════════════════════════════════════════════
STRICT GEOMETRY ACCURACY — the renderer draws EXACTLY what you specify
here, with no visual judgement of its own. A wrong or incomplete spec
produces a wrong or disjointed diagram. Follow these exactly:
════════════════════════════════════════════════════════════════════
  - EVERY point/vertex letter you use anywhere in "given", "required",
    "steps", or "final_answer" for this question (A, B, C, D, M, N, P,
    Q, ...) that is part of the figure MUST also appear in "points" —
    do not reference a point in your proof that isn't in the diagram,
    and do not invent a point in the diagram that never appears in the
    proof. The diagram and the written proof must use the IDENTICAL
    set of point letters, matching the book's own labeling exactly
    (not generic A/B/C substitutes).
  - For "triangle": list the 3 main triangle vertices FIRST, in
    points[0..2], in the order they'd be traced around the triangle
    (e.g. apex, then left-base, then right-base). Any additional point
    — a midpoint, a foot of a perpendicular, a point on an extended
    side — goes AFTER those three, in points[3:]. The renderer assumes
    this ordering to decide what connects to what; listing them out of
    order produces a disjointed-looking figure.
  - "equal_marks" and "right_angle_at" may ONLY reference point ids
    that are actually present in "points" — a reference to a point
    that isn't listed is silently dropped by the renderer, so the tick
    marks or right-angle box you intended just won't appear. Double
    check every id you use here is spelled identically to its entry in
    "points".
  - Keep the figure minimal and exactly matching what's needed for
    THIS question — don't add extra points, marks, or labels beyond
    what the question and proof actually reference.
"""

# Sentinel tokens — deliberately NOT curly-brace based, so they can
# never be confused with JSON syntax and never need escaping.
_TOKEN_CLASS = "@@CLASS_NAME@@"
_TOKEN_CHAPTER = "@@CHAPTER@@"
_TOKEN_EXERCISE = "@@EXERCISE_LABEL@@"
_TOKEN_DIAGRAM_SCHEMA = "@@DIAGRAM_SPEC_SCHEMA@@"

SOLVE_PROMPT_TEMPLATE = f"""
You are solving one Mathematics exercise from a Class {_TOKEN_CLASS} SEBA/NCERT
textbook, Chapter {_TOKEN_CHAPTER}, Exercise {_TOKEN_EXERCISE}. The exercise pages
are attached as a PDF.

TASK:
1. Read the PDF and identify EVERY question in this exercise, including all
   lettered sub-parts (a), (b), (c) etc. Do not skip any.
2. Solve every question completely and correctly. Double-check every
   calculation before finalizing an answer.
3. Write all explanatory text in natural, clear Assamese suitable for a
   school student — never machine-translated-sounding phrasing.
4. Return ONLY a single JSON object, no commentary, no markdown code fences,
   matching EXACTLY this schema:

{{
  "exercise_label": "{_TOKEN_EXERCISE}",
  "questions": [
    {{
      "question_number": "1",
      "sub_part": null or "a",
      "question_text": "the original question, transcribed exactly as printed",
      "given": "প্ৰদত্ত ... (Assamese) — see PROOF STRUCTURE rule below",
      "required": "প্ৰয়োজনীয় ... (Assamese) — see PROOF STRUCTURE rule below",
      "steps": ["সমাধান step 1 ...", "step 2 ...", "..."],
      "final_answer": "চূড়ান্ত উত্তৰ ...",
      "construction_instruments": null or ["জ্যামিতি বাকচ", "কম্পাছ", "স্কেল", "চাঁদমাৰি"],
                                              // REQUIRED (non-null, non-empty) whenever this is a
                                              // geometric CONSTRUCTION question (see the
                                              // CONSTRUCTION (অংকন) QUESTIONS rule below) — null for
                                              // every other question.
      {_TOKEN_DIAGRAM_SCHEMA}
    }}
  ]
}}

════════════════════════════════════════════════════════════════════
JSON ESCAPING — CRITICAL, THIS BREAKS THE ENTIRE RESPONSE IF WRONG
════════════════════════════════════════════════════════════════════
Your output is parsed as JSON. Every backslash you write for LaTeX
(\\frac, \\Delta, \\sqrt, \\cong, \\triangle, \\angle, \\left, \\right,
\\neq, ...) MUST be written as TWO backslashes in the JSON text, e.g.
the JSON string value must contain \\\\frac{{1}}{{2}}, not \\frac{{1}}{{2}}.
A single backslash before a letter is invalid JSON and will cause the
ENTIRE exercise (every question in it) to fail — not just the one
equation. When in doubt, double it.

════════════════════════════════════════════════════════════════════
MATH RENDERING RULES — STRICT, NO EXCEPTIONS
════════════════════════════════════════════════════════════════════
Every equation, fraction, exponent, root, or symbol MUST be wrapped in
MathJax delimiters so it renders as typeset mathematics, never as raw
text:
  - INLINE math (inside a sentence): \\( ... \\)   e.g. \\(AB = AC\\)
  - DISPLAY math (its own centered line): $$ ... $$  e.g. $$OB^2 = OA^2 + AB^2$$
  - Do NOT use single "$...$" for inline math under any circumstances —
    it is not configured and will render as a literal dollar sign.
  - Fractions are ALWAYS \\frac{{numerator}}{{denominator}} inside \\( \\)
    or $$ $$ — never write "3/4" or "15/1600" as plain text.
  - Every \\( has a matching \\), every $$ has a matching $$. Unbalanced
    delimiters break the whole page's rendering, not just one question.
  - Variable names, angle symbols (∠), congruence (≅), and triangle (△)
    symbols used in running Assamese prose (not inside an equation) can
    stay as plain Unicode — only wrap them in \\( \\) when they're part
    of an actual mathematical expression/equation.

════════════════════════════════════════════════════════════════════
PROOF / GEOMETRY STRUCTURE — MANDATORY for any proof, "show that",
"prove that", or geometry question
════════════════════════════════════════════════════════════════════
For such questions you MUST fill in ALL of:
  - "given": starts with "উক্তি:" or "প্ৰদত্ত:" content — restate what's
    given/assumed, in Assamese.
  - "required": what must be proved/shown/determined, in Assamese.
  - "steps": a numbered chain of reasoning, each step citing the
    geometric rule used in parentheses — e.g. "(প্ৰদত্ত)", "(সাধাৰণ বাহু)",
    "(CPCT)", "(SSS সৰ্বসমতাৰ চৰ্ত অনুসৰি)" — exactly the style used in a
    standard SEBA/NCERT textbook proof.
  - "final_answer": a one-line conclusion.
Only for a PURE calculation question with no "prove/show" framing (e.g.
"simplify √50") is it acceptable to leave "given"/"required" as short
empty strings — "steps" and "final_answer" are ALWAYS required either way.

════════════════════════════════════════════════════════════════════
STEP ECONOMY — WRITE LIKE THE PRINTED TEXTBOOK, NOT A TUTOR MONOLOGUE
════════════════════════════════════════════════════════════════════
Students STUDY from these solutions; every sentence must earn its place.
Brevity here is a correctness requirement, not a style preference:
  - Each "steps" entry carries EXACTLY ONE new thing: one substitution,
    one computation, or one rule application. Never two of those per
    step, never a step that introduces nothing new.
  - NEVER restate the question, repeat what an earlier step already
    established, define standard symbols, or narrate filler ("এতিয়া আমি
    দেখোঁ যে...", "ইয়াত আমি ... ব্যৱহাৰ কৰিম"). Open each step directly
    with the mathematics or the rule citation.
  - Compress obvious arithmetic INTO the step that uses its result
    (write the simplified value; do not spend a whole step on 2 + 3).
  - The mandatory structure above (প্ৰদত্ত/প্ৰয়োজনীয়/সমাধান/চূড়ান্ত উত্তৰ)
    and every rule citation stay REQUIRED — economy removes words,
    never rigor.
  - Calibration: a routine 2-3 mark exercise solves in about 2-4 steps,
    a standard proof in about 3-6; only a genuinely long derivation
    (simultaneous equations, multi-part construction) goes further.
    If two consecutive steps say nearly the same thing, delete one.

════════════════════════════════════════════════════════════════════
CONSTRUCTION (অংকন) QUESTIONS — MANDATORY for any question asking the
student to DRAW a figure with instruments (compass/ruler/protractor),
e.g. "অংকন কৰা", "নিৰ্মাণ কৰা", "construct a triangle with...", "draw a
line segment/angle/bisector using ruler and compass" — as distinct from
a question that merely REFERENCES an already-drawn figure to prove
something about it (that stays under the PROOF/GEOMETRY rule above, not
this one).
════════════════════════════════════════════════════════════════════
A construction question is NOT satisfied by showing only the final
figure — a student reading the solution must be able to physically
reproduce the drawing step by step, the same way the printed textbook
itself teaches construction chapters. For such questions you MUST:
  - Set "construction_instruments" to the actual, complete list of
    instruments needed for THIS specific construction (e.g. a triangle
    construction from three sides needs only "কম্পাছ" and "স্কেল"; one
    involving a given angle also needs "চাঁদমাৰি" (protractor); never
    list an instrument the construction doesn't actually use, and never
    leave it null for a genuine construction question).
  - Write "steps" as an ORDERED, literal drawing procedure — each step
    a physical instrument action with an exact measurement taken
    straight from the question's own numbers, e.g. "স্কেলৰ সহায়ত 5
    চে.মি. দৈৰ্ঘ্যৰ ৰেখাখণ্ড BC অংকন কৰা।", "B ক কেন্দ্ৰ কৰি 4 চে.মি.
    ব্যাসাৰ্ধৰে এটা চাপ অংকন কৰা।", "C ৰ পৰা একেই ব্যাসাৰ্ধৰে আন এটা চাপ
    অংকন কৰি প্ৰথম চাপটোক A বিন্দুত কাটিবলৈ দিয়া।" — never collapse
    the physical procedure into an algebraic/proof-style step; a
    construction's "steps" describe what the HAND does with the
    instrument, in the order it must be done, not why it works.
  - If the construction has a justification (why the resulting figure
    satisfies the question, e.g. "why does this bisect the angle") the
    book usually gives afterward, add that as one or two final entries
    in "steps", clearly after all the drawing actions, not mixed in
    with them — mirroring how a construction chapter is laid out.
  - "final_answer" for a construction question is a one-line statement
    of what was constructed (e.g. "ওপৰৰ পদ্ধতি অনুসৰি ত্ৰিভুজ ABC
    অংকন কৰা হ'ল।"), not a numeric result.
  - Set "diagram_spec" to diagram_type "construction" (see its schema
    above), with "construction_steps" built from the SAME physical
    actions as "steps" above, in the SAME order — every "draw segment
    BC of length X" becomes a "line_segment" entry, every "swing an
    arc from B/C with radius Y" becomes an "arc" entry (sharing one
    "intersection_id" with whichever other arc locates the same new
    point), and every "join A to B/C" becomes a "join" entry. Never
    use "triangle" or another finished-figure diagram_type for a
    construction question — the whole point is showing the build
    sequence, not the completed figure.
Every other question type (proofs, calculations, data questions) MUST
leave "construction_instruments" as null — do not set it just because a
diagram happens to be involved; it is reserved specifically for
questions that ask the student to physically draw something with
instruments.
"""


def build_solve_prompt(class_name: int, chapter: int, exercise_label: str, 
                       chapter_title: str = None, is_construction_chapter: bool = False) -> str:
    """Fills the sentinel tokens in SOLVE_PROMPT_TEMPLATE via plain
    str.replace — never str.format()/re-interpolation of the final text —
    so the literal JSON braces throughout the schema can never collide
    with a substitution (see the module docstring).
    
    Args:
        class_name: Class level (9, 10, etc.)
        chapter: Chapter number
        exercise_label: Exercise label (e.g., "7.1")
        chapter_title: Optional chapter title for construction detection
        is_construction_chapter: Whether this is a construction chapter
    """
    prompt = SOLVE_PROMPT_TEMPLATE
    prompt = prompt.replace(_TOKEN_CLASS, str(class_name))
    prompt = prompt.replace(_TOKEN_CHAPTER, str(chapter))
    prompt = prompt.replace(_TOKEN_EXERCISE, exercise_label)
    prompt = prompt.replace(_TOKEN_DIAGRAM_SCHEMA, DIAGRAM_SPEC_SCHEMA)
    
    # Add chapter-specific construction guidance
    if is_construction_chapter:
        construction_guidance = f"""

════════════════════════════════════════════════════════════════════
CHAPTER-SPECIFIC RULE: This is a CONSTRUCTION chapter ("{chapter_title or 'Constructions'}")
════════════════════════════════════════════════════════════════════
EVERY question in this chapter is a construction question. You MUST:
  - Set "construction_instruments" for EVERY question (never null)
  - Write "steps" as physical drawing procedures (ruler/compass actions)
  - Use diagram_type "construction" for EVERY diagram_spec
  - NEVER use proof-style reasoning for construction steps
  - NEVER leave "construction_instruments" as null for any question
"""
        prompt = prompt.replace("TASK:", construction_guidance + "\nTASK:")
    else:
        non_construction_guidance = f"""

════════════════════════════════════════════════════════════════════
CHAPTER-SPECIFIC RULE: This is NOT a construction chapter
════════════════════════════════════════════════════════════════════
This chapter requires mathematical solutions, NOT construction instructions.
  - Use "construction_instruments" ONLY when a question explicitly asks
    to draw/construct something with instruments (rare in this chapter)
  - For regular geometry questions, use proof-style reasoning in "steps"
  - Use appropriate diagram_type (triangle, circle, etc.) NOT "construction"
  - DO NOT add construction-style steps to non-construction questions
"""
        prompt = prompt.replace("TASK:", non_construction_guidance + "\nTASK:")
    
    return prompt


# ════════════════════════════════════════════════════════════════════
# HUMAN REVIEW WORKFLOW — CORRECTION PROMPT (capability 1: "Wrong AI
# Output Correction"). See correction_engine.py for the caller.
# ════════════════════════════════════════════════════════════════════
# Only the fields a reviewer/solve pass can actually edit are ever sent
# back to Gemini for a correction — internal bookkeeping fields
# (answer_kind, diagram_decision, book_diagram_base64, etc.) are set
# deterministically downstream (solver._normalize_all_text_fields,
# diagram_decision.decide_diagram) and must never be something the
# model itself invents or overwrites.
QUESTION_EDITABLE_FIELDS = (
    "question_number", "sub_part", "question_text", "given", "required",
    "steps", "final_answer", "construction_instruments", "diagram_spec",
)

_TOKEN_QUESTION_JSON = "@@QUESTION_JSON@@"
_TOKEN_CORRECTION = "@@CORRECTION_INSTRUCTION@@"

CORRECTION_PROMPT_TEMPLATE = f"""
You are correcting exactly ONE already-solved question from a Class {_TOKEN_CLASS}
SEBA/NCERT Mathematics textbook, Chapter {_TOKEN_CHAPTER}, Exercise {_TOKEN_EXERCISE}.

A human reviewer has checked this question's AI-generated solution and found a
specific problem with it. This is a TARGETED CORRECTION, not a fresh solve:
apply ONLY what the reviewer asks below, and leave every other part of the
question exactly as it already is. Do not "improve" wording, steps, or the
diagram beyond what the instruction actually asks for.

THE QUESTION AS IT CURRENTLY STANDS (JSON):
{_TOKEN_QUESTION_JSON}

REVIEWER'S CORRECTION INSTRUCTION:
\"\"\"
{_TOKEN_CORRECTION}
\"\"\"

TASK:
1. Read the reviewer's instruction carefully. It may ask you to fix a single
   step's calculation, a wrong final answer, a wrong diagram/graph
   description, a formula, or the layout/wording of one field — apply exactly
   that fix.
2. If the instruction says to keep something unchanged (e.g. "keep everything
   else unchanged", "recalculate from Step 2 only"), reproduce those
   untouched parts BYTE-FOR-BYTE identical to how they appear above — do not
   paraphrase or "clean up" text the reviewer didn't ask you to touch.
3. If the instruction only concerns the diagram, leave "question_text",
   "given", "required", "steps", and "final_answer" untouched and modify only
   "diagram_spec". If it only concerns the solution text, leave
   "diagram_spec" untouched (repeat it exactly as given above, or as null if
   it was null).
4. Follow the SAME JSON-escaping and MathJax-formatting rules as a full
   solve pass (below), and the same STEP ECONOMY discipline: any steps
   you rewrite must each carry exactly one new fact, with no filler or
   restated information.
5. Return ONLY a single corrected JSON object for THIS ONE question, no
   commentary, no markdown code fences, matching EXACTLY this schema:

{{
  "question_number": "same as above, unchanged",
  "sub_part": "same as above, unchanged",
  "question_text": "...",
  "given": "...",
  "required": "...",
  "steps": ["...", "..."],
  "final_answer": "...",
  "construction_instruments": null or [...],
  {_TOKEN_DIAGRAM_SCHEMA}
}}

════════════════════════════════════════════════════════════════════
JSON ESCAPING — CRITICAL, THIS BREAKS THE ENTIRE RESPONSE IF WRONG
════════════════════════════════════════════════════════════════════
Every backslash you write for LaTeX (\\frac, \\Delta, \\sqrt, \\cong,
\\triangle, \\angle, \\left, \\right, \\neq, ...) MUST be written as TWO
backslashes in the JSON text, e.g. the JSON string value must contain
\\\\frac{{1}}{{2}}, not \\frac{{1}}{{2}}. A single backslash before a letter is
invalid JSON. When in doubt, double it.

════════════════════════════════════════════════════════════════════
MATH RENDERING RULES — STRICT, NO EXCEPTIONS
════════════════════════════════════════════════════════════════════
  - INLINE math: \\( ... \\)   e.g. \\(AB = AC\\)
  - DISPLAY math (its own centered line): $$ ... $$  e.g. $$OB^2 = OA^2 + AB^2$$
  - Never use single "$...$" for inline math.
  - Fractions are ALWAYS \\frac{{numerator}}{{denominator}} inside \\( \\)
    or $$ $$ — never plain text like "3/4".
  - Every \\( has a matching \\), every $$ has a matching $$.
"""


def build_correction_prompt(question: dict, correction_instruction: str,
                             class_name: int, chapter: int, exercise_label: str) -> str:
    """Builds a narrow, single-question correction prompt: only this
    question's current editable fields plus the reviewer's free-text
    instruction are ever sent — never the rest of the exercise — so a
    targeted fix ("Step 3 is wrong, recalc from Step 2, keep everything
    else unchanged") can never perturb any other question. This is what
    makes "regenerate ONLY the affected component" possible without a
    second full exercise-solve call.
    """
    question_json = {field: question.get(field) for field in QUESTION_EDITABLE_FIELDS}
    prompt = CORRECTION_PROMPT_TEMPLATE
    prompt = prompt.replace(_TOKEN_CLASS, str(class_name))
    prompt = prompt.replace(_TOKEN_CHAPTER, str(chapter))
    prompt = prompt.replace(_TOKEN_EXERCISE, exercise_label)
    prompt = prompt.replace(_TOKEN_DIAGRAM_SCHEMA, DIAGRAM_SPEC_SCHEMA)
    prompt = prompt.replace(_TOKEN_QUESTION_JSON, json.dumps(question_json, ensure_ascii=False, indent=2))
    prompt = prompt.replace(_TOKEN_CORRECTION, (correction_instruction or "").strip())
    return prompt
