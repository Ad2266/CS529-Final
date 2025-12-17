from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Set, Optional

from cgshop2026_pyutils.geometry import Point, FlippableTriangulation

Edge = Tuple[int, int]
TriKey = Tuple[Edge, ...]


def _normalize_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def _triangulation_key_from_edges(edges: Iterable[Edge]) -> TriKey:
    return tuple(sorted(_normalize_edge(u, v) for (u, v) in edges))


@dataclass
class FlipStep:
    edge: Edge
    new_edge: Edge
    resulting_edges: TriKey


def find_flip_path_edges_greedy_fast(
    points_xy: Sequence[Sequence[float]],
    start_edges: Iterable[Edge],
    goal_edges: Iterable[Edge],
    *,
    max_steps: int = 50_000,
) -> Optional[List[FlipStep]]:


    # 1) 点坐标 => Point
    points: List[Point] = [Point(x, y) for (x, y) in points_xy]

    # 2) 当前边集 & 目标边集
    cur_set: Set[Edge] = set(_triangulation_key_from_edges(start_edges))
    goal_key: TriKey = _triangulation_key_from_edges(goal_edges)
    goal_set: Set[Edge] = set(goal_key)

    # diff_edges = E_cur Δ G
    diff_edges: Set[Edge] = cur_set ^ goal_set
    diff_size: int = len(diff_edges)


    if diff_size == 0:
        return []

    # 3) 用 FlippableTriangulation 维护当前 triangulation
    tri = FlippableTriangulation.from_points_edges(points, list(cur_set))

    path: List[FlipStep] = []

    for step in range(max_steps):
        # 检查是否到了目标
        if diff_size == 0:
            return path

        best_new_diff: Optional[int] = None
        best_move: Optional[Tuple[Edge, Edge]] = None  # (old_e, new_e)


        for e in tri.possible_flips():
            u, v = int(e[0]), int(e[1])
            old_e: Edge = _normalize_edge(u, v)

            if old_e not in diff_edges:
                continue

            partner = tri.get_flip_partner(e)
            new_e: Edge = _normalize_edge(int(partner[0]), int(partner[1]))

            delta = 0
            if old_e in goal_set:

                delta += 1
            else:

                delta -= 1


            if new_e in goal_set:

                delta -= 1
            else:

                delta += 1

            new_diff_size = diff_size + delta
            if best_new_diff is None or new_diff_size < best_new_diff:
                best_new_diff = new_diff_size
                best_move = (old_e, new_e)

        if best_move is None:
            return None

        old_e, new_e = best_move

        # 真正执行 flip，更新 tri + cur_set + diff_edges + diff_size
        tri.add_flip((old_e[0], old_e[1]))
        tri.commit()

        # 更新 cur_set
        cur_set.remove(old_e)
        cur_set.add(new_e)


        #  old_e
        if old_e in goal_set:

            diff_edges.add(old_e)
        else:
G
            diff_edges.discard(old_e)

        # new_e
        if new_e in goal_set:

            diff_edges.discard(new_e)
        else:

            diff_edges.add(new_e)

        diff_size = len(diff_edges)

   
        path.append(
            FlipStep(
                edge=old_e,
                new_edge=new_e,
                resulting_edges=_triangulation_key_from_edges(cur_set),
            )
        )

      

    return None
