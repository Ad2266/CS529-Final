from typing import Iterable, List, Tuple, Set, Dict, Optional
import random

from cgshop2026_pyutils.geometry import Point, FlippableTriangulation

Edge = Tuple[int, int]


def normalize_edge(e: Edge) -> Edge:
    u, v = e
    return (u, v) if u < v else (v, u)


def pack_sequential_flips_to_parallel_mis(
    points: List[Point],
    start_edges: Iterable[Edge],
    sequential_flips: Iterable[Edge],
    *,
    randomize_order: bool = True,
    seed: Optional[int] = None,
) -> List[List[Edge]]:
    rng = random.Random(seed)
    start_edges_norm = [normalize_edge(e) for e in start_edges]
    remaining: List[Edge] = [normalize_edge(e) for e in sequential_flips]

    tri = FlippableTriangulation.from_points_edges(points, start_edges_norm)

    rounds: List[List[Edge]] = []

    while remaining:
        possible_now: Set[Edge] = {normalize_edge(e) for e in tri.possible_flips()}

        candidate_indices: List[int] = [
            i for i, e in enumerate(remaining) if e in possible_now
        ]
        if not candidate_indices:
            raise RuntimeError(
                "No remaining flips are flippable in current triangulation. "
                "Sequential flip sequence may be inconsistent."
            )

        candidate_edges: Set[Edge] = {remaining[i] for i in candidate_indices}

        conflict_map: Dict[Edge, Set[Edge]] = {}
        for e in candidate_edges:
            conflicts_for_e = {
                normalize_edge(c)
                for c in tri._flip_map.conflicting_flips(e)
            }
            conflicts_for_e.add(e)
            conflict_map[e] = conflicts_for_e

        order = candidate_indices[:]
        if randomize_order:
            rng.shuffle(order)
        else:
            order.sort(key=lambda idx: len(conflict_map[remaining[idx]]))

        selected_indices: List[int] = []
        selected_edges: Set[Edge] = set()

        for idx in order:
            e = remaining[idx]
            has_conflict = any(
                (e in conflict_map[se]) or (se in conflict_map[e])
                for se in selected_edges
            )
            if has_conflict:
                continue
            selected_indices.append(idx)
            selected_edges.add(e)

        if not selected_indices:
            idx0 = candidate_indices[0]
            selected_indices = [idx0]
            selected_edges = {remaining[idx0]}

        current_round = [remaining[i] for i in selected_indices]

        for e in current_round:
            tri.add_flip(e)
        tri.commit()

        executed = set(selected_indices)
        remaining = [e for i, e in enumerate(remaining) if i not in executed]

        rounds.append(current_round)

    return rounds

