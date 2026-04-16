"""
Visualize the 2-hop neighborhood around a seed node in a RelBench database.

Produces an interactive HTML file (Plotly).  Nodes are color-coded by the
table they belong to; hover text shows the table name and row ID.

Usage (CLI):
    python visualize_neighborhood.py \
        --dataset rel-f1 \
        --seed-table drivers \
        --seed-id 0 \
        [--cutoff-time "2005-01-01"] \
        [--output neighborhood.html]

Usage (Python):
    from visualize_neighborhood import visualize_2hop
    visualize_2hop("rel-f1", ("drivers", 0), cutoff_time=pd.Timestamp("2005-01-01"))
"""

import argparse
import json
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from compute_2hop_neighbors import build_undirected_adj


# ---------------------------------------------------------------------------
# Colour palette – one colour per table
# ---------------------------------------------------------------------------

# Kelly's colours of maximum contrast — perceptually distinct even with many tables
_PALETTE = [
    "#F3C300", "#875692", "#F38400", "#A1CAF1", "#BE0032",
    "#C2B280", "#008856", "#E68FAC", "#0067A5", "#F99379",
    "#604E97", "#B3446C", "#DCD300", "#882D17", "#8DB600",
    "#E25822", "#2B3D26", "#654522", "#F6A600", "#848482",
]


def _table_color_map(tables):
    return {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(sorted(tables))}


def _blend_hex(hex1, hex2):
    """Return the perceptual midpoint of two hex colours."""
    def to_rgb(h):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r1, g1, b1 = to_rgb(hex1)
    r2, g2, b2 = to_rgb(hex2)
    return f"#{(r1+r2)//2:02x}{(g1+g2)//2:02x}{(b1+b2)//2:02x}"


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------

def build_2hop_subgraph(seed_node, adj, node_time, cutoff_time=None):
    """
    Return a networkx Graph containing the seed node and all nodes reachable
    within 2 hops, respecting the optional temporal cutoff.

    Parameters
    ----------
    seed_node : tuple[str, int]   e.g. ("drivers", 3)
    adj       : dict from build_undirected_adj
    node_time : dict from build_undirected_adj
    cutoff_time : pd.Timestamp | None
        If given, neighbours whose timestamp >= cutoff_time are excluded.

    Returns
    -------
    G : nx.Graph
        Each node carries attributes: table (str), row_id (int), hop (0/1/2).
    """

    def valid(nbr):
        if cutoff_time is None:
            return True
        t = node_time.get(nbr)
        return t is None or t < cutoff_time

    G = nx.Graph()

    seed_table, seed_id = seed_node
    G.add_node(seed_node, table=seed_table, row_id=seed_id, hop=0)

    # --- Pass 1: discover nodes via BFS, assign hop distances ---
    one_hop = {nbr for nbr in adj[seed_node] if valid(nbr)}
    for nbr in one_hop:
        t, rid = nbr
        G.add_node(nbr, table=t, row_id=rid, hop=1)

    visited = {seed_node} | one_hop
    for nbr in one_hop:
        for nbr2 in adj[nbr]:
            if nbr2 not in visited and valid(nbr2):
                t2, rid2 = nbr2
                G.add_node(nbr2, table=t2, row_id=rid2, hop=2)
                visited.add(nbr2)

    # --- Pass 2: add every edge whose both endpoints are in the subgraph ---
    # Separating discovery from edge-recording means a races node first reached
    # via standings still gets its edges from results and qualifying too —
    # none are silently dropped by the visited-set check.
    subgraph_nodes = set(G.nodes())
    for node in subgraph_nodes:
        for nbr in adj[node]:
            if nbr in subgraph_nodes and valid(nbr):
                G.add_edge(node, nbr)

    return G


# ---------------------------------------------------------------------------
# Layout  –  force-directed (spring) layout
# ---------------------------------------------------------------------------

def _spring_positions(G, seed_node, seed=42):
    """
    Compute node positions using the Fruchterman-Reingold force-directed
    algorithm.  The seed node is pinned at the origin so it stays central;
    all other nodes are placed by the physics simulation, which naturally
    pulls tightly connected nodes together and pushes unconnected ones apart.
    """
    # Pin the seed at (0, 0); let everything else float
    fixed_pos = {seed_node: np.array([0.0, 0.0])}

    # k controls the preferred edge length — larger values spread nodes out more.
    # Scale with 1/sqrt(n) (the networkx default) but boosted for readability.
    n = G.number_of_nodes()
    k = 2.5 / (n ** 0.5)

    pos = nx.spring_layout(
        G,
        k=k,
        pos=fixed_pos,
        fixed=[seed_node],
        iterations=100,
        seed=seed,
    )
    return pos


# ---------------------------------------------------------------------------
# Plotly rendering
# ---------------------------------------------------------------------------

def _node_str(n):
    return f"{n[0]}:{n[1]}"


def _build_figure_and_metadata(G, seed_node, pos, color_map, title):
    """
    Build the Plotly figure and the metadata dict needed for JS highlight injection.

    Edge traces: one per edge (not per table-pair) so each can be individually
    greyed out or highlighted by the click handler.
    Node traces: one per table, with customdata=[node_id_str, ...] per point so
    the click handler knows which node was clicked.

    Returns
    -------
    fig      : go.Figure
    metadata : dict with keys edgeTraces, nodeTraces, seedNode
    """
    all_tables = sorted({d["table"] for _, d in G.nodes(data=True)})
    edges      = list(G.edges())

    # ---- one trace per edge ----
    edge_traces   = []
    edge_metadata = []  # parallel list; index i → trace index i in the figure

    for i, (u, v) in enumerate(edges):
        t_u   = G.nodes[u]["table"]
        t_v   = G.nodes[v]["table"]
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        color = _blend_hex(color_map[t_u], color_map[t_v])

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=1.5, color=color),
            hoverinfo="none",
            showlegend=False,
        ))
        edge_metadata.append({
            "traceIdx": i,
            "nodeA": _node_str(u),
            "nodeB": _node_str(v),
            "color": color,
        })

    num_edge_traces = len(edge_traces)

    # ---- one node trace per table ----
    node_traces   = []
    node_metadata = []  # parallel list; index i → trace index num_edge_traces+i

    for t_idx, table in enumerate(all_tables):
        trace_idx = num_edge_traces + t_idx
        nodes = [n for n, d in G.nodes(data=True) if d["table"] == table]

        xs      = [pos[n][0] for n in nodes]
        ys      = [pos[n][1] for n in nodes]
        sizes   = [22 if G.nodes[n]["hop"] == 0 else 14 if G.nodes[n]["hop"] == 1 else 9
                   for n in nodes]
        symbols = ["star" if G.nodes[n]["hop"] == 0 else "circle" for n in nodes]
        hover   = [f"<b>{table}</b><br>ID: {G.nodes[n]['row_id']}<br>Hop: {G.nodes[n]['hop']}"
                   for n in nodes]

        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            customdata=[_node_str(n) for n in nodes],
            marker=dict(size=sizes, color=color_map[table], symbol=symbols,
                        line=dict(width=1, color="#333333")),
            hovertext=hover,
            hoverinfo="text",
            name=f"{table} ({len(nodes)})",
            legendgroup=table,
        ))
        node_metadata.append({
            "traceIdx": trace_idx,
            "nodes": [_node_str(n) for n in nodes],
        })

    # ---- annotation: label the seed node ----
    seed_x, seed_y = pos[seed_node]
    annotations = [dict(
        x=seed_x, y=seed_y,
        text=f"<b>{seed_node[0]}:{seed_node[1]}</b>",
        showarrow=True, arrowhead=2, ax=30, ay=-30,
        font=dict(size=13, color="#111111"),
        xref="x", yref="y",
    )]

    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=True,
            legend=dict(title="Table", itemsizing="constant"),
            hovermode="closest",
            xaxis=dict(visible=False, scaleanchor="y"),
            yaxis=dict(visible=False),
            plot_bgcolor="#F9F9F9",
            paper_bgcolor="#FFFFFF",
            annotations=annotations,
            margin=dict(l=20, r=20, t=60, b=20),
            width=1000,
            height=900,
        ),
    )

    metadata = {
        "edgeTraces": edge_metadata,
        "nodeTraces": node_metadata,
        "seedNode":   _node_str(seed_node),
    }
    return fig, metadata


# ---------------------------------------------------------------------------
# HTML output with click-to-highlight injection
# ---------------------------------------------------------------------------

def _write_html_with_highlight(fig, G, seed_node, metadata, output):
    """
    Write the figure to an HTML file, injecting a JS click handler that:
      - Single-click a node  → grey out all nodes/edges not on the shortest
                               path from that node back to the seed; highlight
                               the path in its original colours.
      - Double-click anywhere → reset to full-colour view.

    Shortest paths are pre-computed in Python and embedded as JSON so the
    browser needs no graph library — just a plain Plotly.restyle call.
    """
    # Pre-compute the union of ALL simple paths (cutoff=2 covers the full 2-hop
    # subgraph) from every node back to the seed.  Storing the union as two sets
    # (path nodes, path edges) avoids sending redundant path lists to JS and lets
    # the highlight function work in O(1) per node/edge.
    path_nodes_json: dict[str, list] = {}
    path_edges_json: dict[str, list] = {}

    for node in G.nodes():
        p_nodes: set[str] = set()
        p_edges: set[str] = set()

        for path in nx.all_simple_paths(G, node, seed_node, cutoff=2):
            for n in path:
                p_nodes.add(_node_str(n))
            for i in range(len(path) - 1):
                a = _node_str(path[i])
                b = _node_str(path[i + 1])
                p_edges.add(a + "|" + b if a < b else b + "|" + a)

        path_nodes_json[_node_str(node)] = list(p_nodes)
        path_edges_json[_node_str(node)] = list(p_edges)

    payload = json.dumps({
        "pathNodes":  path_nodes_json,
        "pathEdges":  path_edges_json,
        "edgeTraces": metadata["edgeTraces"],
        "nodeTraces": metadata["nodeTraces"],
        "seedNode":   metadata["seedNode"],
    })

    # Double-click fires plotly_click twice then plotly_doubleclick.
    # We debounce single-click by 250 ms so a double-click doesn't
    # accidentally trigger two highlight operations.
    js = f"""
<script>
(function () {{
  var payload     = {payload};
  var pathNodes   = payload.pathNodes;   /* nodeStr -> [nodeStr, ...] union over all paths */
  var pathEdges   = payload.pathEdges;   /* nodeStr -> ["a|b", ...] union over all paths  */
  var edgeTraces  = payload.edgeTraces;
  var nodeTraces  = payload.nodeTraces;
  var seedNode    = payload.seedNode;
  var highlighted = false;
  var clickTimer  = null;

  /* ---- helpers ---- */
  function edgeKey(a, b) {{ return a < b ? a + '|' + b : b + '|' + a; }}

  function reset(div) {{
    Plotly.restyle(div,
      {{'line.color': edgeTraces.map(function(e) {{ return e.color; }})}},
      edgeTraces.map(function(e) {{ return e.traceIdx; }})
    );
    Plotly.restyle(div,
      {{'marker.opacity': nodeTraces.map(function(t) {{
          return Array(t.nodes.length).fill(1.0);
      }})}},
      nodeTraces.map(function(t) {{ return t.traceIdx; }})
    );
    highlighted = false;
  }}

  function highlight(div, clickedStr) {{
    /* pathNodes/pathEdges are the union of ALL simple paths to seed */
    var pNodes = pathNodes[clickedStr];
    var pEdges = pathEdges[clickedStr];
    if (!pNodes || pNodes.length === 0) return;

    var pathNodeSet = new Set(pNodes);
    var pathEdgeSet = new Set(pEdges);

    /* restyle edges: highlight every edge on any path, grey out others */
    Plotly.restyle(div,
      {{'line.color': edgeTraces.map(function(e) {{
          return pathEdgeSet.has(edgeKey(e.nodeA, e.nodeB)) ? e.color : '#DDDDDD';
      }})}},
      edgeTraces.map(function(e) {{ return e.traceIdx; }})
    );

    /* restyle nodes: full opacity for nodes on any path, dim others */
    Plotly.restyle(div,
      {{'marker.opacity': nodeTraces.map(function(t) {{
          return t.nodes.map(function(n) {{ return pathNodeSet.has(n) ? 1.0 : 0.15; }});
      }})}},
      nodeTraces.map(function(t) {{ return t.traceIdx; }})
    );

    highlighted = true;
  }}

  /* ---- attach handlers once Plotly has rendered ---- */
  function attach() {{
    var div = document.querySelector('.js-plotly-plot');
    if (!div) {{ setTimeout(attach, 100); return; }}

    div.on('plotly_click', function (data) {{
      var pt = data.points[0];
      if (pt.customdata === undefined || pt.customdata === null) return;

      /* debounce: ignore first click of a double-click pair */
      if (clickTimer) {{ clearTimeout(clickTimer); clickTimer = null; return; }}
      var clickedStr = pt.customdata;
      clickTimer = setTimeout(function () {{
        clickTimer = null;
        if (highlighted && clickedStr === seedNode) {{
          reset(div);
        }} else {{
          highlight(div, clickedStr);
        }}
      }}, 250);
    }});

    div.on('plotly_doubleclick', function () {{
      if (clickTimer) {{ clearTimeout(clickTimer); clickTimer = null; }}
      reset(div);
    }});
  }}

  attach();
}})();
</script>
"""

    html = fig.to_html(full_html=True, include_plotlyjs=True)
    html = html.replace("</body>", js + "\n</body>")
    with open(output, "w") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Cutoff-time lookup from task table
# ---------------------------------------------------------------------------

def resolve_cutoff_time(dataset_name, task_name, split, seed_table, seed_id):
    """
    Find the cutoff time(s) for a given seed node in a task split table and
    return the latest one.

    Returns pd.Timestamp, or None if the seed is not found in that split.
    """
    task = get_task(dataset_name, task_name, download=True)

    if not hasattr(task, "entity_col"):
        raise NotImplementedError(
            f"Auto cutoff lookup is only supported for EntityTask, "
            f"not {type(task)}."
        )
    if task.entity_table != seed_table:
        raise ValueError(
            f"Task {task_name!r} uses entity table {task.entity_table!r}, "
            f"but seed table is {seed_table!r}."
        )

    mask_input = split == "test"
    table = task.get_table(split, mask_input_cols=mask_input)
    df = table.df

    matches = df[df[task.entity_col] == seed_id]
    if matches.empty:
        return None

    timestamps = pd.to_datetime(matches[task.time_col]).sort_values()
    print(f"  Found {len(timestamps)} timestamp(s) for seed {seed_table}:{seed_id} "
          f"in {split} split:")
    for ts in timestamps:
        print(f"    {ts.date()}")

    chosen = timestamps.iloc[-1]
    print(f"  Using latest: {chosen.date()}")
    return chosen


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_2hop(
    dataset_name,
    seed_node,
    cutoff_time=None,
    task_name=None,
    split="train",
    output="neighborhood.html",
    verbose=True,
):
    """
    Build and save an interactive 2-hop neighbourhood visualisation.

    Parameters
    ----------
    dataset_name : str
    seed_node    : tuple  (table_name, row_id)
    cutoff_time  : pd.Timestamp | None
        If None and task_name is given, the cutoff is auto-looked up from the
        task split table (latest timestamp for this seed).
        If None and task_name is also None, the full static graph is used.
    task_name    : str | None   e.g. "driver-top3"
    split        : str          "train" | "val" | "test"  (used for auto-lookup)
    output       : str          path for the HTML output file
    """
    if verbose:
        print(f"Loading dataset {dataset_name!r} …")
    dataset = get_dataset(dataset_name, download=True)
    db = dataset.get_db()

    # Auto-resolve cutoff from task table when not provided explicitly
    if cutoff_time is None and task_name is not None:
        if verbose:
            print(f"Looking up cutoff time from task {task_name!r} / {split} split …")
        cutoff_time = resolve_cutoff_time(
            dataset_name, task_name, split, seed_node[0], seed_node[1]
        )
        if cutoff_time is None and verbose:
            print(f"  Seed not found in {split} split — using full static graph.")

    if verbose:
        print("Building adjacency list …")
    adj, node_time = build_undirected_adj(db)

    if verbose:
        ct_str = str(cutoff_time.date()) if cutoff_time is not None else "none (static graph)"
        print(f"Extracting 2-hop subgraph around {seed_node}  cutoff={ct_str} …")

    G = build_2hop_subgraph(seed_node, adj, node_time, cutoff_time)

    n1 = sum(1 for _, d in G.nodes(data=True) if d["hop"] == 1)
    n2 = sum(1 for _, d in G.nodes(data=True) if d["hop"] == 2)
    if verbose:
        print(f"  Subgraph: {G.number_of_nodes()} nodes "
              f"({n1} at hop-1, {n2} at hop-2), {G.number_of_edges()} edges")

    all_tables = sorted({d["table"] for _, d in G.nodes(data=True)})
    color_map = _table_color_map(all_tables)

    pos = _spring_positions(G, seed_node)

    total_neighbors = n1 + n2
    cutoff_label = f"  cutoff: {cutoff_time.date()}" if cutoff_time is not None else ""
    title = (
        f"2-hop neighbourhood — seed: {seed_node[0]}:{seed_node[1]}{cutoff_label}"
        f"<br><sup>Total 2-hop neighbors: {total_neighbors}"
        f" &nbsp;|&nbsp; {n1} hop-1 · {n2} hop-2 · {G.number_of_edges()} edges</sup>"
    )

    fig, metadata = _build_figure_and_metadata(G, seed_node, pos, color_map, title)
    _write_html_with_highlight(fig, G, seed_node, metadata, output)

    if verbose:
        print(f"Saved → {output}")

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise the 2-hop neighbourhood around a RelBench seed node."
    )
    parser.add_argument("--dataset",    required=True, help="e.g. rel-f1")
    parser.add_argument("--seed-table", required=True, help="e.g. drivers")
    parser.add_argument("--seed-id",    required=True, type=int,
                        help="Row index of the seed node")
    parser.add_argument("--task",  default=None,
                        help="Task name used to auto-derive cutoff time, "
                             "e.g. driver-top3")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test"],
                        help="Split to look up the cutoff time from (default: train)")
    parser.add_argument("--cutoff-time", default=None,
                        help="Explicit ISO cutoff timestamp, e.g. 2005-01-01. "
                             "Overrides auto-lookup. If omitted and --task is "
                             "not given, the full static graph is used.")
    parser.add_argument("--output", default="neighborhood.html",
                        help="Output HTML file (default: neighborhood.html)")
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.cutoff_time) if args.cutoff_time else None

    visualize_2hop(
        dataset_name=args.dataset,
        seed_node=(args.seed_table, args.seed_id),
        cutoff_time=cutoff,
        task_name=args.task,
        split=args.split,
        output=args.output,
    )


if __name__ == "__main__":
    main()
