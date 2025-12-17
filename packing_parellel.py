from typing import Iterable, List, Tuple
from cgshop2026_pyutils.geometry import Point, FlippableTriangulation

Edge = Tuple[int, int]


def normalize_edge(e: Edge) -> Edge:
    """(u,v) and (v,u)"""
    u, v = e
    return (u, v) if u < v else (v, u)


def pack_sequential_flips_to_parallel(
    points: List[Point],
    start_edges: Iterable[Edge],
    sequential_flips: Iterable[Edge],
) -> List[List[Edge]]:
    start_edges_norm = [normalize_edge(e) for e in start_edges]
    remaining = [normalize_edge(e) for e in sequential_flips]

    tri = FlippableTriangulation.from_points_edges(points, start_edges_norm)
    rounds: List[List[Edge]] = []

    while remaining:
        possible_now = {normalize_edge(e) for e in tri.possible_flips()}

        blocked = set()
        current_round: List[Edge] = []

        j = 0
        while j < len(remaining):
            e = remaining[j]
            if e not in possible_now:
                break
            if e in blocked:
                break

            current_round.append(e)

            for c in tri._flip_map.conflicting_flips(e):
                blocked.add(normalize_edge(c))

            j += 1

        if not current_round:
            raise RuntimeError(
                f"Packing stalled at next edge {remaining[0]}; "
                "sequential path likely inconsistent with start triangulation."
            )


        for e in current_round:
            tri.add_flip(e)
        tri.commit()

        rounds.append(current_round)

        remaining = remaining[len(current_round):]

    return rounds
