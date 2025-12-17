from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set
import math
import heapq

from cgshop2026_pyutils.geometry import Point, FlippableTriangulation

Edge = Tuple[int, int]
TriKey = Tuple[Edge, ...]


def _normalize_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def _triangulation_key_from_edges(edges: Iterable[Edge]) -> TriKey:
    return tuple(sorted(_normalize_edge(u, v) for (u, v) in edges))


@dataclass(order=True)
class _QueueItem:
    priority: float
    key: TriKey = field(compare=False)


@dataclass
class FlipStep:
    edge: Edge
    new_edge: Edge
    resulting_edges: TriKey


def _edge_difference_heuristic(current: TriKey, goal_edges: Set[Edge]) -> int:
    cur = set(current)
    return len(cur - goal_edges)


def _neighbors(
    points: Sequence[Point],
    state: TriKey,
) -> Iterable[Tuple[Edge, Edge, TriKey]]:
    edges = list(state)
    tri = FlippableTriangulation.from_points_edges(points, edges)
    edge_set = set(state)

    for e in tri.possible_flips():
        e_norm = _normalize_edge(*e)
        new_e = _normalize_edge(*tri.get_flip_partner(e))

        new_edges = set(edge_set)
        new_edges.remove(e_norm)
        new_edges.add(new_e)
        new_key = _triangulation_key_from_edges(new_edges)
        yield e_norm, new_e, new_key


def find_flip_path_edges(
    points_xy: Sequence[Sequence[float]],
    start_edges: Iterable[Edge],
    goal_edges: Iterable[Edge],
    *,
    max_expansions: int = 100000,
) -> Optional[List[FlipStep]]:
    points = [Point(x, y) for (x, y) in points_xy]

    start_key = _triangulation_key_from_edges(start_edges)
    goal_key = _triangulation_key_from_edges(goal_edges)
    goal_edges_set = set(goal_key)

    if start_key == goal_key:
        return []

    open_heap: List[_QueueItem] = []
    heapq.heappush(
        open_heap,
        _QueueItem(
            priority=_edge_difference_heuristic(start_key, goal_edges_set),
            key=start_key,
        ),
    )

    g_scores: Dict[TriKey, int] = {start_key: 0}
    parents: Dict[TriKey, Tuple[TriKey, Edge, Edge]] = {}

    expansions = 0

    while open_heap:
        current_item = heapq.heappop(open_heap)
        current_key = current_item.key

        if current_key == goal_key:
            return _reconstruct_path_edges(current_key, parents)

        expansions += 1
        if expansions > max_expansions:
            return None

        current_g = g_scores[current_key]

        for old_e, new_e, neigh_key in _neighbors(points, current_key):
            tentative_g = current_g + 1
            if tentative_g < g_scores.get(neigh_key, math.inf):
                g_scores[neigh_key] = tentative_g
                parents[neigh_key] = (current_key, old_e, new_e)
                h = _edge_difference_heuristic(neigh_key, goal_edges_set)
                heapq.heappush(open_heap, _QueueItem(tentative_g + h, neigh_key))

    return None


def _reconstruct_path_edges(
    goal_key: TriKey,
    parents: Dict[TriKey, Tuple[TriKey, Edge, Edge]],
) -> List[FlipStep]:
    path: List[FlipStep] = []
    current = goal_key
    while current in parents:
        parent, edge, new_edge = parents[current]
        path.append(FlipStep(edge=edge, new_edge=new_edge, resulting_edges=current))
        current = parent
    path.reverse()
    return path

