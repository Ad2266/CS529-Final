from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

from cgshop2026_pyutils.geometry import Point, FlippableTriangulation

Edge = Tuple[int, int]
TriKey = Tuple[Edge, ...]


def normalize_edge(e: Edge) -> Edge:
    u, v = e
    return (u, v) if u < v else (v, u)


def trikey_from_edges(edges: Iterable[Edge]) -> TriKey:
    return tuple(sorted(normalize_edge(e) for e in edges))


def edge_set_from_trikey(key: TriKey) -> Set[Edge]:
    return set(key)


def cheap_edge_distance(a: TriKey, b: TriKey) -> int:
    ea = edge_set_from_trikey(a)
    eb = edge_set_from_trikey(b)
    return len(ea ^ eb) // 2


@dataclass
class CenterSearchResultEdges:
    center: TriKey
    objective: int
    history: List[int]


def _objective_sum(
    center_key: TriKey,
    tri_edge_sets: Sequence[Set[Edge]],
) -> int:
    center_edges = edge_set_from_trikey(center_key)
    total = 0
    for edges_i in tri_edge_sets:
        total += len(center_edges ^ edges_i) // 2
    return total


def _choose_initial_center(
    tri_keys: Sequence[TriKey],
    tri_edge_sets: Sequence[Set[Edge]],
) -> Tuple[int, TriKey, int]:
    best_idx = 0
    best_obj = float("inf")
    for j, key in enumerate(tri_keys):
        obj = _objective_sum(key, tri_edge_sets)
        if obj < best_obj:
            best_obj = obj
            best_idx = j
    return best_idx, tri_keys[best_idx], int(best_obj)


def find_central_triangulation_edges(
    points_xy: Sequence[Sequence[float]],
    triangulations_edges: Sequence[Iterable[Edge]],
    *,
    max_iterations: int = 50,
) -> CenterSearchResultEdges:
    if not triangulations_edges:
        raise ValueError("Need at least one triangulation")

    points: List[Point] = [Point(x, y) for (x, y) in points_xy]

    tri_keys: List[TriKey] = [
        trikey_from_edges(edges) for edges in triangulations_edges
    ]
    tri_edge_sets: List[Set[Edge]] = [edge_set_from_trikey(k) for k in tri_keys]

    _, current_center, current_obj = _choose_initial_center(
        tri_keys, tri_edge_sets
    )
    history: List[int] = [current_obj]

    for _ in range(max_iterations):
        improved = False
        best_neighbor: TriKey | None = None
        best_obj = current_obj

        current_edges = edge_set_from_trikey(current_center)
        tri = FlippableTriangulation.from_points_edges(points, list(current_edges))

        for e in tri.possible_flips():
            old_e: Edge = normalize_edge(e)
            new_e: Edge = normalize_edge(tri.get_flip_partner(e))

            if old_e not in current_edges:
                continue

            neighbor_edges = set(current_edges)
            neighbor_edges.remove(old_e)
            neighbor_edges.add(new_e)
            neighbor_key = trikey_from_edges(neighbor_edges)

            obj = _objective_sum(neighbor_key, tri_edge_sets)

            if obj < best_obj:
                best_obj = obj
                best_neighbor = neighbor_key
                improved = True

        if not improved or best_neighbor is None:
            break

        current_center = best_neighbor
        current_obj = best_obj
        history.append(current_obj)

    return CenterSearchResultEdges(
        center=current_center,
        objective=current_obj,
        history=history,
    )

