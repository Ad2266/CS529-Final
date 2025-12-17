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
    sym_diff = cur ^ goal_edges
    return len(sym_diff) // 2


def _neighbors(
    points: Sequence[Point],
    state: TriKey,
) -> Iterable[Tuple[Edge, Edge, TriKey]]:
    edges = list(state)
    tri = FlippableTriangulation.from_points_edges(points, edges)
    edge_set = set(state)

    for e in tri.possible_flips():
        e_norm = _normalize_edge(int(e[0]), int(e[1]))
        new_partner = tri.get_flip_partner(e)
        new_e = _normalize_edge(int(new_partner[0]), int(new_partner[1]))

        new_edges = set(edge_set)
        new_edges.remove(e_norm)
        new_edges.add(new_e)
        new_key = _triangulation_key_from_edges(new_edges)
        yield e_norm, new_e, new_key


def _greedy_tail_from_state(
    points: Sequence[Point],
    start_key: TriKey,
    goal_key: TriKey,
    goal_edges_set: Set[Edge],
    *,
    max_steps: int = 50_000,
) -> Optional[List[FlipStep]]:
    current_key = start_key
    tail: List[FlipStep] = []

    for _ in range(max_steps):
        if current_key == goal_key:
            return tail

        edges = list(current_key)
        tri = FlippableTriangulation.from_points_edges(points, edges)
        edge_set = set(current_key)

        base_h = _edge_difference_heuristic(current_key, goal_edges_set)
        best_improve = 0
        best_move = None

        for e in tri.possible_flips():
            old_e = _normalize_edge(int(e[0]), int(e[1]))
            new_partner = tri.get_flip_partner(e)
            new_e = _normalize_edge(int(new_partner[0]), int(new_partner[1]))

            new_edges = set(edge_set)
            new_edges.remove(old_e)
            new_edges.add(new_e)
            new_key = _triangulation_key_from_edges(new_edges)

            h_new = _edge_difference_heuristic(new_key, goal_edges_set)
            improve = base_h - h_new
            if improve > best_improve:
                best_improve = improve
                best_move = (old_e, new_e, new_key)

        if best_move is None:
            return None

        old_e, new_e, new_key = best_move
        tail.append(FlipStep(edge=old_e, new_edge=new_e, resulting_edges=new_key))
        current_key = new_key

    return None


def find_flip_path_edges(
    points_xy: Sequence[Sequence[float]],
    start_edges: Iterable[Edge],
    goal_edges: Iterable[Edge],
    *,
    max_expansions: int = 100_000,
    weight: float = 1.0,
    greedy_fallback: bool = True,
    greedy_max_steps: int = 50_000,
) -> Optional[List[FlipStep]]:
    if weight < 1.0:
        weight = 1.0

    points = [Point(x, y) for (x, y) in points_xy]

    start_key = _triangulation_key_from_edges(start_edges)
    goal_key = _triangulation_key_from_edges(goal_edges)
    goal_edges_set = set(goal_key)

    if start_key == goal_key:
        return []

    open_heap: List[_QueueItem] = []
    start_h = _edge_difference_heuristic(start_key, goal_edges_set)
    heapq.heappush(open_heap, _QueueItem(priority=start_h, key=start_key))

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
            if not greedy_fallback:
                return None

            prefix = _reconstruct_path_edges(current_key, parents)
            tail = _greedy_tail_from_state(
                points,
                start_key=current_key,
                goal_key=goal_key,
                goal_edges_set=goal_edges_set,
                max_steps=greedy_max_steps,
            )
            if tail is None:
                return None
            return prefix + tail

        current_g = g_scores[current_key]

        for old_e, new_e, neigh_key in _neighbors(points, current_key):
            tentative_g = current_g + 1
            if tentative_g < g_scores.get(neigh_key, math.inf):
                g_scores[neigh_key] = tentative_g
                parents[neigh_key] = (current_key, old_e, new_e)
                h = _edge_difference_heuristic(neigh_key, goal_edges_set)
                f = tentative_g + weight * h
                heapq.heappush(open_heap, _QueueItem(f, neigh_key))

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


def _simulate_states_from_seq(
    start_edges: Iterable[Edge],
    seq_path: List[FlipStep],
) -> List[TriKey]:
    current_set: Set[Edge] = set(_triangulation_key_from_edges(start_edges))
    states: List[TriKey] = [_triangulation_key_from_edges(current_set)]

    for step in seq_path:
        old_e = _normalize_edge(*step.edge)
        new_e = _normalize_edge(*step.new_edge)
        if old_e not in current_set:
            raise RuntimeError(
                f"FlipStep inconsistent: edge {old_e} not in current triangulation."
            )
        current_set.remove(old_e)
        current_set.add(new_e)
        states.append(_triangulation_key_from_edges(current_set))

    return states


def _remove_immediate_inverses(seq_path: List[FlipStep]) -> List[FlipStep]:
    def is_inverse(a: FlipStep, b: FlipStep) -> bool:
        return (
            _normalize_edge(*a.edge) == _normalize_edge(*b.new_edge)
            and _normalize_edge(*a.new_edge) == _normalize_edge(*b.edge)
        )

    result: List[FlipStep] = []
    i = 0
    n = len(seq_path)
    while i < n:
        if i + 1 < n and is_inverse(seq_path[i], seq_path[i + 1]):
            i += 2
        else:
            result.append(seq_path[i])
            i += 1
    return result


def _local_optimize_windows(
    points_xy: Sequence[Sequence[float]],
    start_edges: Iterable[Edge],
    seq_path: List[FlipStep],
    *,
    window_size: int = 8,
    max_local_rounds: int = 2,
    local_max_expansions: int = 20_000,
    local_weight: float = 1.0,
) -> List[FlipStep]:
    if not seq_path:
        return seq_path

    for _ in range(max_local_rounds):
        improved = False
        states = _simulate_states_from_seq(start_edges, seq_path)
        n = len(seq_path)
        assert len(states) == n + 1

        i = 0
        while i < n:
            j = min(i + window_size, n)
            start_key = states[i]
            goal_key = states[j]

            if start_key == goal_key:
                del seq_path[i:j]
                n = len(seq_path)
                states = _simulate_states_from_seq(start_edges, seq_path)
                improved = True
                continue

            new_subpath = find_flip_path_edges(
                points_xy,
                start_edges=list(start_key),
                goal_edges=list(goal_key),
                max_expansions=local_max_expansions,
                weight=local_weight,
                greedy_fallback=False,
            )

            if new_subpath is not None and len(new_subpath) < (j - i):
                seq_path = seq_path[:i] + new_subpath + seq_path[j:]
                n = len(seq_path)
                states = _simulate_states_from_seq(start_edges, seq_path)
                improved = True
                i = max(i - window_size // 2, 0)
                continue

            i += 1

        if not improved:
            break

    return seq_path


def shorten_flip_path(
    points_xy: Sequence[Sequence[float]],
    start_edges: Iterable[Edge],
    seq_path: List[FlipStep],
    *,
    window_size: int = 8,
    max_local_rounds: int = 2,
    local_max_expansions: int = 20_000,
    local_weight: float = 1.0,
    verify_final: bool = True,
) -> List[FlipStep]:
    if not seq_path:
        return seq_path

    orig_states = _simulate_states_from_seq(start_edges, seq_path)
    orig_final = orig_states[-1]

    seq_opt = _remove_immediate_inverses(seq_path)

    seq_opt = _local_optimize_windows(
        points_xy,
        start_edges,
        seq_opt,
        window_size=window_size,
        max_local_rounds=max_local_rounds,
        local_max_expansions=local_max_expansions,
        local_weight=local_weight,
    )

    if verify_final:
        new_states = _simulate_states_from_seq(start_edges, seq_opt)
        new_final = new_states[-1]
        if new_final != orig_final:
            raise RuntimeError(
                "shorten_flip_path changed the final triangulation; "
                "this should not happen. Please check implementation."
            )

    return seq_opt


def _local_optimize_windows_aggressive(
    points_xy: Sequence[Sequence[float]],
    start_edges: Iterable[Edge],
    seq_path: List[FlipStep],
    *,
    window_size: int = 24,
    max_local_rounds: int = 5,
    local_max_expansions: int = 80_000,
    local_weight: float = 1.0,
) -> List[FlipStep]:
    if not seq_path:
        return seq_path

    for _ in range(max_local_rounds):
        improved = False
        states = _simulate_states_from_seq(start_edges, seq_path)
        n = len(seq_path)
        assert len(states) == n + 1

        i = 0
        while i < n:
            j = min(i + window_size, n)
            start_key = states[i]
            goal_key = states[j]
            seg_len = j - i

            if start_key == goal_key:
                del seq_path[i:j]
                n = len(seq_path)
                states = _simulate_states_from_seq(start_edges, seq_path)
                improved = True
                continue

            new_subpath = find_flip_path_edges(
                points_xy,
                start_edges=list(start_key),
                goal_edges=list(goal_key),
                max_expansions=local_max_expansions,
                weight=local_weight,
                greedy_fallback=False,
            )

            if new_subpath is not None and len(new_subpath) < seg_len:
                seq_path = seq_path[:i] + new_subpath + seq_path[j:]
                n = len(seq_path)
                states = _simulate_states_from_seq(start_edges, seq_path)
                improved = True
                i = max(i - window_size // 2, 0)
                continue

            i += 1

        if not improved:
            break

    return seq_path


def _global_refine_with_astar(
    points_xy: Sequence[Sequence[float]],
    start_edges: Iterable[Edge],
    seq_path: List[FlipStep],
    *,
    global_max_expansions: int = 300_000,
    global_weight: float = 1.0,
) -> List[FlipStep]:
    if not seq_path:
        return seq_path

    states = _simulate_states_from_seq(start_edges, seq_path)
    final_state = states[-1]
    L_old = len(seq_path)

    new_path = find_flip_path_edges(
        points_xy,
        start_edges=start_edges,
        goal_edges=list(final_state),
        max_expansions=global_max_expansions,
        weight=global_weight,
        greedy_fallback=False,
    )

    if new_path is not None and len(new_path) < L_old:
        return new_path
    return seq_path


def shorten_flip_path_aggressive(
    points_xy: Sequence[Sequence[float]],
    start_edges: Iterable[Edge],
    seq_path: List[FlipStep],
    *,
    window_size: int = 24,
    max_local_rounds: int = 5,
    local_max_expansions: int = 80_000,
    local_weight: float = 1.0,
    do_global_refine: bool = True,
    global_max_expansions: int = 300_000,
    global_weight: float = 1.0,
    verify_final: bool = True,
) -> List[FlipStep]:
    if not seq_path:
        return seq_path

    orig_states = _simulate_states_from_seq(start_edges, seq_path)
    orig_final = orig_states[-1]

    seq_opt = _remove_immediate_inverses(seq_path)

    seq_opt = _local_optimize_windows_aggressive(
        points_xy,
        start_edges,
        seq_opt,
        window_size=window_size,
        max_local_rounds=max_local_rounds,
        local_max_expansions=local_max_expansions,
        local_weight=local_weight,
    )

    if do_global_refine:
        seq_opt = _global_refine_with_astar(
            points_xy,
            start_edges,
            seq_opt,
            global_max_expansions=global_max_expansions,
            global_weight=global_weight,
        )

    if verify_final:
        new_states = _simulate_states_from_seq(start_edges, seq_opt)
        new_final = new_states[-1]
        if new_final != orig_final:
            raise RuntimeError(
                "shorten_flip_path_aggressive changed the final triangulation; "
                "this should not happen. Please check implementation."
            )

    return seq_opt

