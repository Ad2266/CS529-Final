from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple, Optional, Callable
import random

from cgshop2026_pyutils.geometry import Point, FlippableTriangulation

Edge = Tuple[int, int]
TriKey = Tuple[Edge, ...]
DistFn = Callable[[TriKey, Set[Edge]], int]


def normalize_edge(e: Edge) -> Edge:
    u, v = e
    return (u, v) if u < v else (v, u)


def trikey_from_edges(edges: Iterable[Edge]) -> TriKey:
    return tuple(sorted(normalize_edge(e) for e in edges))


def edge_set_from_trikey(key: TriKey) -> Set[Edge]:
    return set(key)


@dataclass
class CenterSearchResultEdges:
    center: TriKey
    objective: int
    history: List[int]


def _orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> int:
    val = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if val > 0:
        return 1
    if val < 0:
        return -1
    return 0


def _segments_properly_cross(e1: Edge, e2: Edge, pts: Sequence[Tuple[float, float]]) -> bool:
    a, b = e1
    c, d = e2

    if a in (c, d) or b in (c, d):
        return False

    ax, ay = pts[a]
    bx, by = pts[b]
    cx, cy = pts[c]
    dx, dy = pts[d]

    o1 = _orientation(ax, ay, bx, by, cx, cy)
    o2 = _orientation(ax, ay, bx, by, dx, dy)
    o3 = _orientation(cx, cy, dx, dy, ax, ay)
    o4 = _orientation(cx, cy, dx, dy, bx, by)

    return (o1 * o2 < 0) and (o3 * o4 < 0)


def make_crossing_dist_fn(points_xy: Sequence[Sequence[float]]) -> DistFn:
    pts = [(float(x), float(y)) for (x, y) in points_xy]

    def dist_fn(center_key: TriKey, edges_i: Set[Edge]) -> int:
        center_edges = edge_set_from_trikey(center_key)
        total = 0
        for e1 in center_edges:
            for e2 in edges_i:
                if _segments_properly_cross(e1, e2, pts):
                    total += 1
        return total

    return dist_fn


def _objective_sum(
    center_key: TriKey,
    tri_edge_sets: Sequence[Set[Edge]],
    dist_fn: DistFn,
) -> int:
    return sum(dist_fn(center_key, edges_i) for edges_i in tri_edge_sets)


def _choose_initial_center(
    tri_keys: Sequence[TriKey],
    tri_edge_sets: Sequence[Set[Edge]],
    dist_fn: DistFn,
) -> Tuple[int, TriKey, int]:
    best_idx = 0
    best_obj = float("inf")
    for j, key in enumerate(tri_keys):
        obj = _objective_sum(key, tri_edge_sets, dist_fn)
        if obj < best_obj:
            best_obj = obj
            best_idx = j
    return best_idx, tri_keys[best_idx], int(best_obj)


def _hill_climb_center(
    points: List[Point],
    tri_edge_sets: Sequence[Set[Edge]],
    init_center: TriKey,
    *,
    max_iterations: int,
    dist_fn: DistFn,
) -> CenterSearchResultEdges:
    current_center = init_center
    current_obj = _objective_sum(current_center, tri_edge_sets, dist_fn)
    history = [current_obj]

    for _ in range(max_iterations):
        current_edges = edge_set_from_trikey(current_center)
        tri = FlippableTriangulation.from_points_edges(points, list(current_edges))

        best_neighbor: Optional[TriKey] = None
        best_obj = current_obj

        for e in tri.possible_flips():
            e_raw = (int(e[0]), int(e[1]))
            old_e = normalize_edge(e_raw)
            new_raw = tri.get_flip_partner(e)
            new_e = normalize_edge((int(new_raw[0]), int(new_raw[1])))

            if old_e not in current_edges:
                continue

            neighbor_edges = set(current_edges)
            neighbor_edges.remove(old_e)
            neighbor_edges.add(new_e)
            neighbor_key = trikey_from_edges(neighbor_edges)

            obj = _objective_sum(neighbor_key, tri_edge_sets, dist_fn)
            if obj < best_obj:
                best_obj = obj
                best_neighbor = neighbor_key

        if best_neighbor is None:
            break

        current_center = best_neighbor
        current_obj = best_obj
        history.append(current_obj)

    return CenterSearchResultEdges(center=current_center, objective=current_obj, history=history)


def _random_walk_from_center(
    points: List[Point],
    start_center: TriKey,
    *,
    steps: int,
    rng: random.Random,
) -> TriKey:
    edges = set(start_center)
    tri = FlippableTriangulation.from_points_edges(points, list(edges))

    for _ in range(steps):
        poss = list(tri.possible_flips())
        if not poss:
            break
        e = rng.choice(poss)
        e_raw = (int(e[0]), int(e[1]))
        old_e = normalize_edge(e_raw)

        new_raw = tri.get_flip_partner(e)
        new_e = normalize_edge((int(new_raw[0]), int(new_raw[1])))

        tri.add_flip(e_raw)
        tri.commit()

        if old_e in edges:
            edges.remove(old_e)
        edges.add(new_e)

    return trikey_from_edges(edges)


def find_central_triangulation_edges_multistart(
    points_xy: Sequence[Sequence[float]],
    triangulations_edges: Sequence[Iterable[Edge]],
    *,
    num_starts: int = 20,
    random_walk_steps: int = 5,
    max_iterations: int = 50,
    seed: int = 0,
    include_medoid_start: bool = True,
) -> CenterSearchResultEdges:
    if not triangulations_edges:
        raise ValueError("Need at least one triangulation")

    rng = random.Random(seed)
    points: List[Point] = [Point(x, y) for (x, y) in points_xy]
    tri_keys: List[TriKey] = [trikey_from_edges(edges) for edges in triangulations_edges]
    tri_edge_sets: List[Set[Edge]] = [edge_set_from_trikey(k) for k in tri_keys]

    dist_fn = make_crossing_dist_fn(points_xy)

    starts: List[TriKey] = []
    if include_medoid_start:
        _, medoid_key, _ = _choose_initial_center(tri_keys, tri_edge_sets, dist_fn)
        starts.append(medoid_key)

    for _ in range(num_starts):
        starts.append(rng.choice(tri_keys))

    best: Optional[CenterSearchResultEdges] = None
    for s in starts:
        s0 = _random_walk_from_center(points, s, steps=random_walk_steps, rng=rng) if random_walk_steps > 0 else s
        res = _hill_climb_center(points, tri_edge_sets, s0, max_iterations=max_iterations, dist_fn=dist_fn)
        if best is None or res.objective < best.objective:
            best = res

    assert best is not None
    return best


def find_central_triangulation_edges_deterministic_starts(
    points_xy: Sequence[Sequence[float]],
    triangulations_edges: Sequence[Iterable[Edge]],
    *,
    max_iterations: int = 50,
    include_medoid_start: bool = True,
) -> CenterSearchResultEdges:
    if not triangulations_edges:
        raise ValueError("Need at least one triangulation")

    points = [Point(x, y) for (x, y) in points_xy]
    tri_keys = [trikey_from_edges(edges) for edges in triangulations_edges]
    tri_edge_sets = [edge_set_from_trikey(k) for k in tri_keys]

    dist_fn = make_crossing_dist_fn(points_xy)

    starts: List[TriKey] = []
    if include_medoid_start:
        _, medoid_key, _ = _choose_initial_center(tri_keys, tri_edge_sets, dist_fn)
        starts.append(medoid_key)

    seen = set(starts)
    for k in tri_keys:
        if k not in seen:
            starts.append(k)
            seen.add(k)

    best: Optional[CenterSearchResultEdges] = None
    for s in starts:
        res = _hill_climb_center(points, tri_edge_sets, s, max_iterations=max_iterations, dist_fn=dist_fn)
        if best is None or res.objective < best.objective:
            best = res

    assert best is not None
    return best

