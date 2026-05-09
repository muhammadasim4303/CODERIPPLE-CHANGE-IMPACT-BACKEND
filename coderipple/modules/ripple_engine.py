"""
ripple_engine.py
────────────────
Stage 4: Ripple Propagation Engine

Given a set of changed function node IDs and a DependencyGraph,
performs BFS to compute:
  • directly impacted nodes   (graph distance == 1)
  • indirectly impacted nodes (graph distance  > 1)
  • ripple depth              (max BFS depth reached)
  • ripple size               (total reachable nodes including origin)

The traversal follows REVERSE edges — i.e. "who calls this function?" —
because we want to find what breaks when a function changes.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field

import networkx as nx

from .dependency_graph import DependencyGraph

logger = logging.getLogger(__name__)


@dataclass
class RippleResult:
    changed_nodes:   list[str]
    direct_impact:   list[str]          # distance == 1
    indirect_impact: list[str]          # distance >= 2
    impacted_files:  list[str]
    ripple_depth:    int
    ripple_size:     int                # total impacted (excl. changed nodes)
    impact_map:      dict[str, int]     # node_id → distance


class RippleEngine:
    """Propagates change impact through the dependency graph."""

    def __init__(self, dep_graph: DependencyGraph) -> None:
        self.dg        = dep_graph
        # Build reverse graph once (callers of each node)
        self._rev_graph: nx.DiGraph = dep_graph.graph.reverse(copy=True)

    # ── Public API ────────────────────────────────────────────────────────

    def propagate(self, changed_node_ids: list[str], max_depth: int = 10) -> RippleResult:
        """
        BFS from every changed node through the REVERSE graph.
        Returns a RippleResult summarising the blast radius.
        """
        # Resolve aliases — try to find the closest matching node
        resolved = [self._resolve(nid) for nid in changed_node_ids]
        resolved = [n for n in resolved if n is not None]

        if not resolved:
            logger.warning("None of the changed nodes found in graph: %s", changed_node_ids)
            return RippleResult(
                changed_nodes   = changed_node_ids,
                direct_impact   = [],
                indirect_impact = [],
                impacted_files  = [],
                ripple_depth    = 0,
                ripple_size     = 0,
                impact_map      = {},
            )

        impact_map: dict[str, int] = {}   # node → min distance from any changed node

        queue: deque[tuple[str, int]] = deque()
        for n in resolved:
            queue.append((n, 0))
            impact_map[n] = 0

        max_reached = 0

        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for caller in self._rev_graph.successors(node):
                if caller not in impact_map or impact_map[caller] > depth + 1:
                    impact_map[caller] = depth + 1
                    max_reached = max(max_reached, depth + 1)
                    queue.append((caller, depth + 1))

        # Separate changed, direct, indirect
        changed_set  = set(resolved)
        direct_set   = {n for n, d in impact_map.items() if d == 1 and n not in changed_set}
        indirect_set = {n for n, d in impact_map.items() if d >= 2  and n not in changed_set}

        impacted_files = sorted({
            self.dg.graph.nodes[n].get("file_path", "")
            for n in (direct_set | indirect_set)
            if n in self.dg.graph
        } - {""})

        return RippleResult(
            changed_nodes   = list(changed_set),
            direct_impact   = sorted(direct_set),
            indirect_impact = sorted(indirect_set),
            impacted_files  = impacted_files,
            ripple_depth    = max_reached,
            ripple_size     = len(direct_set) + len(indirect_set) + len(changed_set),
            impact_map      = {k: v for k, v in impact_map.items() if k not in changed_set},
        )

    # ── Node resolution ───────────────────────────────────────────────────

    def _resolve(self, node_id: str) -> str | None:
        """
        Return the node_id if it exists in the graph.
        If not, try partial matching by function name.
        """
        if node_id in self.dg.graph:
            return node_id

        # Try to find by function name suffix
        name = node_id.split("::")[-1]
        for n in self.dg.graph.nodes:
            if n.endswith(f"::{name}"):
                logger.debug("Resolved %s → %s", node_id, n)
                return n

        logger.debug("Node not found in graph: %s", node_id)
        return None

    # ── Utility ───────────────────────────────────────────────────────────

    def subgraph_for_display(self, ripple: RippleResult, max_nodes: int = 50) -> dict:
        """
        Return a simplified adjacency structure suitable for JSON serialisation
        (e.g., to feed a front-end graph visualiser).
        """
        relevant = set(ripple.changed_nodes) | set(ripple.direct_impact) | set(ripple.indirect_impact)
        relevant = list(relevant)[:max_nodes]

        nodes = []
        for n in relevant:
            dist      = ripple.impact_map.get(n, 0)
            category  = "changed" if n in ripple.changed_nodes else (
                        "direct"  if dist == 1 else "indirect")
            node_data  = self.dg.graph.nodes.get(n, {})
            node_kind  = node_data.get("kind", "unknown")
            file_path  = node_data.get("file_path", "")
            raw_label  = n.split("::")[-1]
            # For file nodes (__file__ sentinel), show the actual filename instead
            if raw_label == "__file__":
                display_label = os.path.basename(file_path) if file_path else n.split("::")[0].split("/")[-1].split("\\")[-1]
            else:
                display_label = raw_label
            nodes.append({
                "id":       n,
                "label":    display_label,
                "kind":     node_kind,
                "file":     file_path,
                "distance": dist,
                "category": category,
            })

        edges = []
        for u, v, data in self.dg.graph.edges(data=True):
            if u in relevant and v in relevant:
                edges.append({"source": u, "target": v, "type": data.get("type", "calls")})

        return {"nodes": nodes, "edges": edges}
