"""Phase-7: the explanation layer and visualization exports."""

import json
import re
import xml.etree.ElementTree as ET

import pytest

import qrlib as qr
from qrlib import TerminalClass, semiquant, viz
from qrlib.analysis import explain, queries

from test_qsim_golden import bathtub, spring
from test_regions import two_region_bathtub


# --- explanation -----------------------------------------------------------


def test_explain_structure_and_events():
    m, initial = bathtub()
    result = qr.qsim(m, initial, config=qr.SimConfig.classic())
    (below_full,) = [
        b
        for b in result.behaviors()
        if b.terminal is TerminalClass.QUIESCENT
        and b.node_ids[-1] in queries.quiescent_states(result.graph)
        and result.graph.nodes[b.node_ids[-1]].model.spaces[0].describe(
            b.states[-1]["amount"].mag
        )
        == "amount*0"
    ]
    steps = explain.explain(result.graph, below_full)
    assert len(steps) == len(below_full.states)
    assert steps[0].summary.startswith("initially")
    assert steps[0].time_tag == "point"

    rising = {e.variable: e for e in steps[1].events}
    assert rising["amount"].kind == "starts"
    assert "rises into (0, FULL)" in rising["amount"].detail

    final = steps[-1]
    assert final.terminal == "quiescent"
    assert "amount*0" in final.minted
    events = {e.variable: e for e in final.events}
    assert events["amount"].kind == "steadies"
    assert "newly identified" in events["amount"].detail
    assert events["netflow"].kind == "reaches"
    assert "equilibrium" in final.summary


def test_narrate_reads_as_prose():
    m, initial = bathtub()
    result = qr.qsim(m, initial)
    overflow = next(
        b for b in result.behaviors() if b.terminal is TerminalClass.DOMAIN_EXIT
    )
    text = explain.narrate(result.graph, overflow)
    assert text.startswith("Behavior of 'bathtub': 3 states, ending in domain exit.")
    assert "amount reaches FULL, still rising" in text
    assert "leaves the model's domain" in text


def test_explain_cycle_and_region_crossing():
    m, initial = spring()
    result = qr.qsim(m, initial, config=qr.SimConfig(discover_landmarks=False))
    (cycle,) = result.behaviors()
    steps = explain.explain(result.graph, cycle)
    assert steps[-1].terminal == "cycle"
    assert "repeats step 0" in steps[-1].summary

    m, initial, _ = two_region_bathtub()
    result = qr.qsim(m, initial)
    crossing = next(b for b in result.behaviors() if "overflowing" in b.regions)
    steps = explain.explain(result.graph, crossing)
    entry = steps[-1]
    assert entry.region == "overflowing"
    assert entry.summary.startswith("crossing into overflowing")
    assert "levels off at FULL" in entry.summary


# --- viz data --------------------------------------------------------------


def test_timeline_bands_with_refinement():
    m, initial, _ = two_region_bathtub(values=True)
    result = qr.qsim(m, initial)
    crossing = next(b for b in result.behaviors() if "overflowing" in b.regions)
    refinement = semiquant.refine(result.graph, crossing)

    data = viz.timeline_bands(result.graph, crossing, refinement=refinement)
    json.dumps(data)  # plain data
    assert data["terminal"] == "quiescent"
    assert data["regions"][-1] == "overflowing"
    amount = next(v for v in data["variables"] if v["name"] == "amount")
    assert len(amount["segments"]) == len(crossing.states)
    boundary = amount["segments"][2]
    assert boundary["label"] == "FULL"
    assert boundary["bounds"] == [1.0, 1.0]
    assert boundary["time"][0] == 1.0  # the classic Q2 lower bound


def test_tree_layout_positions():
    m, initial = bathtub()
    result = qr.qsim(m, initial)
    layout = viz.tree_layout(result.graph)
    json.dumps(layout)
    by_id = {n["id"]: n for n in layout["nodes"]}
    assert set(by_id) == set(result.graph.nodes)
    assert by_id[result.graph.root]["y"] == 0
    leaves = [n for n in layout["nodes"] if n["terminal"]]
    assert len({n["x"] for n in leaves}) == len(leaves)  # distinct slots
    for a, b in layout["edges"]:
        assert by_id[b]["y"] == by_id[a]["y"] + 1


# --- svg -------------------------------------------------------------------


def test_timeline_svg_is_valid_and_labeled():
    m, initial = bathtub()
    result = qr.qsim(m, initial)
    svg = viz.timeline_svg(result.graph, result.behaviors()[0])
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "netflow" in svg and "bathtub" in svg


def test_tree_svg_marks_cycles():
    m, initial = spring()
    result = qr.qsim(m, initial, config=qr.SimConfig(discover_landmarks=False))
    svg = viz.tree_svg(result.graph)
    ET.fromstring(svg)
    assert "stroke-dasharray" in svg  # the cycle-closure edge
    assert "[cycle]" in svg


HOSTILE_VARIABLE_NAMES = (
    "<script>alert(1)</script>",
    "</text>",
    'a"b',
    "a&b",
    "a<b>c",
    "]]><!--",
    "a\nb",
)


def _model_with_hostile_variable_name(name):
    model = qr.Model("hostile")
    model.variable(name, landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    model.variable("d", landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    model.constrain(qr.Deriv(name, "d"))
    initial = model.state(
        **{
            name: (("0", "+inf"), qr.Qdir.DEC),
            "d": (("-inf", "0"), qr.Qdir.INC),
        }
    )
    return model, initial


@pytest.mark.parametrize("name", HOSTILE_VARIABLE_NAMES)
def test_visual_exports_escape_hostile_variable_names(name):
    """SVG and DOT exports treat legal model names as data, not markup."""
    model, initial = _model_with_hostile_variable_name(name)
    result = qr.qsim(model, initial, config=qr.SimConfig(max_states=30))

    tree = qr.viz.tree_svg(result.graph)
    timeline = qr.viz.timeline_svg(result.graph, result.behaviors()[0])
    for svg in (tree, timeline):
        root = ET.fromstring(svg)
        text = "".join(root.itertext())
        assert name.replace("\n", "") in text.replace("\n", "")
        assert root.find(".//script") is None

    dot = result.graph.to_dot()
    for label in re.findall(r'label="((?:[^"\\]|\\.)*)"', dot):
        assert '"' not in re.sub(r"\\.", "", label)


@pytest.mark.parametrize("maker", (bathtub, spring))
def test_query_results_agree_with_graph_contents(maker):
    """Analysis queries return exactly the graph nodes they describe."""
    model, initial = maker()
    result = qr.qsim(model, initial, config=qr.SimConfig(max_states=100))
    graph = result.graph

    census = queries.terminal_census(graph)
    assert sum(census.values()) == sum(
        node.terminal is not None for node in graph.nodes.values()
    )

    variable = graph.var_order[0]
    predicate = lambda state: state.as_dict()[variable].dir is qr.Qdir.STD
    hits = set(queries.find_states(graph, predicate))
    assert hits
    assert len(hits) < len(graph.nodes)
    assert hits == {
        node_id
        for node_id, node in graph.nodes.items()
        if predicate(node.state)
    }


def test_quiescent_query_returns_only_steady_states():
    """Every reported quiescent state has only steady tracked directions."""
    model, initial = bathtub()
    result = qr.qsim(model, initial, config=qr.SimConfig(max_states=100))
    node_ids = queries.quiescent_states(result.graph)

    assert node_ids
    for node_id in node_ids:
        directions = result.graph.nodes[node_id].state.as_dict().values()
        assert all(qval.dir in (qr.Qdir.STD, qr.Qdir.IGN) for qval in directions)
