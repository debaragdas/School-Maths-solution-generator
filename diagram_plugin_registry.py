"""
diagram_plugin_registry.py — THE PLUGIN-BASED DIAGRAM ENGINE (Phase 4).

SCOPING DECISION, STATED UP FRONT: the existing 12 diagram types
(triangle, circle, angle, parallel_lines, coordinate_plot,
quadrilateral, trigonometry, statistics, surface_area_volume,
construction, number_line, square_root_spiral) are NOT migrated into
this registry. They stay exactly where they are — diagram_renderer.py's
_RENDERERS dict, diagram_decision.py's per-type schema checks,
diagram_renderer.py's per-type validate_*_spec functions — fully
unchanged, fully covered by their existing tests. Force-migrating
twelve already-correct, already-tested implementations into a new
interface purely for architectural uniformity would be exactly the
"redesign unrelated modules" this project's own principles rule out,
and it would risk regressing working code for zero user-facing
benefit — nobody's diagram gets more correct because triangle.py
became a plugin instead of a dict entry.

WHAT THIS MODULE ACTUALLY DELIVERS, FOR REAL: a genuine extension
point for diagram type #13 onward (probability, ogive-as-its-own-type,
algebra graphs, or whatever comes next). Before this module existed,
adding a new diagram type meant hand-editing THREE separate files
(diagram_renderer.py's _RENDERERS dict, diagram_decision.py's
_infer_type_mismatch's inline per-type schema-footprint checks, and
validate_diagram_spec's dispatch chain) and hoping nothing was missed
— exactly the fragility the original architecture audit flagged as
Weakness #4. From now on, adding a type means writing ONE
DiagramPlugin and calling register_plugin() once. diagram_renderer.py
and diagram_decision.py both consult this registry FIRST; anything
found here is handled entirely by its plugin, and anything not found
here falls through to the original hardcoded dispatch — so nothing
about the 12 existing types changes, ever, as a result of this module
existing.
"""
from dataclasses import dataclass
from typing import Callable, Optional

from utils import logger


@dataclass
class DiagramPlugin:
    """The common interface every future diagram type implements.

    type_name:    the diagram_spec["diagram_type"] string this plugin
                   handles (e.g. "probability").
    renderer:     (normalized_spec: dict) -> svg_string. Must never
                   raise (the registry's own render dispatch already
                   wraps every plugin call in a try/except as a safety
                   net, but a well-behaved plugin should still fail
                   soft internally where it reasonably can).
    schema_check: OPTIONAL (spec: dict) -> (ok: bool, issues: list[str]).
                   Deterministic structural validation, same contract
                   as diagram_renderer.validate_diagram_spec's existing
                   per-type functions. If omitted, the type gets the
                   same permissive default the 12 built-in types
                   without a dedicated check already get elsewhere —
                   "not proven wrong" rather than "proven right".
    solver:       OPTIONAL geometry-constraint solver for this type,
                   same (coords_or_None, reason) contract as
                   geometry_solver.solve_triangle — for types that
                   benefit from real-measurement-driven layout the way
                   Phase 1 added for triangle. Purely optional; a
                   plugin without one just always uses its own
                   schematic layout, same as every pre-Phase-1 type
                   still does today.
    description:  short human-readable note, for `list_plugins()`
                   output / operator visibility only.
    """
    type_name: str
    renderer: Callable[[dict], str]
    schema_check: Optional[Callable[[dict], tuple]] = None
    solver: Optional[Callable[..., tuple]] = None
    description: str = ""


_REGISTRY: dict = {}


def register_plugin(plugin: DiagramPlugin) -> None:
    """Registers a new diagram type. Raises ValueError for a malformed
    plugin (missing type_name/renderer) or an attempt to shadow one of
    the 12 built-in types (those are NOT extension points — a plugin
    claiming "triangle" would silently steal every existing triangle
    question away from the tested, production renderer, which is
    exactly the kind of silent, hard-to-diagnose regression this
    registry exists to prevent, not introduce)."""
    if not plugin.type_name or not isinstance(plugin.type_name, str):
        raise ValueError("DiagramPlugin.type_name must be a non-empty string")
    if not callable(plugin.renderer):
        raise ValueError(f"DiagramPlugin '{plugin.type_name}' must provide a callable renderer")
    if plugin.type_name in _BUILTIN_TYPE_NAMES:
        raise ValueError(f"'{plugin.type_name}' is one of the built-in diagram types and cannot "
                          f"be overridden via the plugin registry — see this module's docstring "
                          f"for why the built-in types are deliberately out of scope for plugins")
    if plugin.type_name in _REGISTRY:
        logger.warning(f"⚠️ diagram_plugin_registry: re-registering plugin type "
                        f"'{plugin.type_name}' — replacing the previous registration.")
    _REGISTRY[plugin.type_name] = plugin


def get_plugin(type_name: str) -> Optional[DiagramPlugin]:
    return _REGISTRY.get(type_name)


def is_plugin_type(type_name: str) -> bool:
    return type_name in _REGISTRY


def list_plugins() -> list:
    return sorted(_REGISTRY.keys())


def unregister_plugin(type_name: str) -> None:
    """Test/ops convenience — not used by the pipeline itself."""
    _REGISTRY.pop(type_name, None)


# The 12 built-in types, named here (not imported from diagram_renderer,
# to avoid a circular import: diagram_renderer imports THIS module to
# consult the registry, so this module must not import diagram_renderer
# back). Kept as a plain literal, deliberately — this list changes only
# if a brand new type is promoted from "plugin" to "built-in" after
# proving itself in production, which is a rare, deliberate decision,
# not something that should silently drift out of sync via an import.
_BUILTIN_TYPE_NAMES = frozenset({
    "triangle", "circle", "angle", "parallel_lines", "coordinate_plot",
    "quadrilateral", "trigonometry", "statistics", "surface_area_volume",
    "construction", "number_line", "square_root_spiral",
})


def render_via_plugin(spec: dict) -> Optional[str]:
    """Returns the rendered SVG for a plugin-registered type, or None
    if this type isn't a plugin (caller should fall through to the
    built-in dispatch) or if rendering failed (same fail-soft contract
    as diagram_renderer.render_diagram's own built-in path — a bad
    plugin must never take down the rest of the exercise)."""
    plugin = _REGISTRY.get(spec.get("diagram_type"))
    if plugin is None:
        return None
    try:
        return plugin.renderer(spec)
    except Exception as e:
        logger.warning(f"⚠️ diagram_plugin_registry: plugin '{plugin.type_name}' raised while "
                        f"rendering ({e}) — omitting this diagram.")
        return ""


def validate_via_plugin(spec: dict):
    """Returns (ok, issues) for a plugin-registered type using its own
    schema_check if it provided one, or (True, []) — the same
    permissive default every un-checked built-in type already gets —
    if it didn't. Returns None if this type isn't a plugin at all
    (caller should fall through to the built-in dispatch)."""
    plugin = _REGISTRY.get(spec.get("diagram_type"))
    if plugin is None:
        return None
    if plugin.schema_check is None:
        return True, []
    try:
        return plugin.schema_check(spec)
    except Exception as e:
        logger.warning(f"⚠️ diagram_plugin_registry: plugin '{plugin.type_name}' raised during "
                        f"schema_check ({e}) — treating as a validation failure rather than "
                        f"trusting an unstable check.")
        return False, [f"schema_check raised an exception: {e}"]
