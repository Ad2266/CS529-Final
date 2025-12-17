from cgshop2026_pyutils.io import read_instance
from cgshop2026_pyutils.geometry import Point
from cgshop2026_pyutils.schemas import CGSHOP2026Solution
from cgshop2026_pyutils.verify import check_for_errors
from search_central import find_central_triangulation_edges
from packing_parellel import pack_sequential_flips_to_parallel
# from sequential_distance import find_flip_path_edges
from sequential_distance_v2 import find_flip_path_edges, shorten_flip_path
from sequential_distance_v3 import shorten_flip_path_aggressive
from cgshop2026_pyutils.geometry import (
    draw_edges,
    draw_flips
)
import matplotlib.pyplot as plt
# from search_central_mul import find_central_triangulation_edges_multistart, find_central_triangulation_edges_deterministic_starts
from search_central_v3 import find_central_triangulation_edges_deterministic_starts, find_central_triangulation_edges_multistart
from packing_parellel_v2 import pack_sequential_flips_to_parallel_mis

# 1. 读 instance
inst = read_instance("test.instance.json")
points_xy = list(zip(inst.points_x, inst.points_y))
triangulations_edges = inst.triangulations  # List[List[Edge]]
print("reading instance ... done")

# 2. 找中心 triangulation
#center_res = find_central_triangulation_edges(points_xy, triangulations_edges)
#center_res = find_central_triangulation_edges_multistart(
#points_xy,triangulations_edges,num_starts=20,random_walk_steps=1,max_iterations=50,seed=36,include_medoid_start= True)

center_res = find_central_triangulation_edges_deterministic_starts(
    points_xy,
    triangulations_edges,
    max_iterations =50,
)
#center_res = find_central_triangulation_edges_multistart(points_xy,triangulations_edges,num_starts=20,random_walk_steps=5,max_iterations=50,seed=42,)
print("history len:", len(center_res.history))
center_key = center_res.center
center_edges = list(center_key)
print("find center ... done")

# 3. 对每个 Ti，算从 center 到 Ti 的顺序 flip 路径，再压成 parallel rounds
all_flips_parallel = []
for T_edges in triangulations_edges:
    seq_path = find_flip_path_edges(points_xy, T_edges, center_edges, weight=1.1)
    print(len(seq_path))

    #seq_path = shorten_flip_path(points_xy,start_edges=T_edges,seq_path=seq_path,window_size=8,
    #max_local_rounds=2,local_max_expansions=20_000,local_weight=1.0,)

    #seq_path = shorten_flip_path_aggressive(points_xy,start_edges=T_edges,
    #seq_path=seq_path,window_size=16,max_local_rounds=5,local_max_expansions=80_000,
    #local_weight=1.0,do_global_refine=True,global_max_expansions=300_000,global_weight=1.0,)
    
    #print(len(seq_path))
    
    seq_edges = [step.edge for step in seq_path]  # 视你的实现而定

    rounds = pack_sequential_flips_to_parallel(points=[Point(x, y) for x, y in points_xy],
        start_edges=T_edges,sequential_flips=seq_edges,)
    #rounds = pack_sequential_flips_to_parallel_mis(points=[Point(x, y) for x, y in points_xy],
    #    start_edges=T_edges,sequential_flips=seq_edges,)
    all_flips_parallel.append(rounds)

print("step 3 ... done")

# 4. 构造 solution + 验证
solution = CGSHOP2026Solution(
    instance_uid=inst.instance_uid,
    flips=all_flips_parallel,
)

print("step 4 ... done")

errors = check_for_errors(inst, solution)
print("errors:", errors)

import json
from typing import Any, Iterable, List, Tuple, Optional

Edge = Tuple[int, int]

def _norm_edge(e: Any) -> Tuple[int, int]:

    u, v = int(e[0]), int(e[1])
    return (u, v) if u < v else (v, u)

def _round_to_str(round_edges: Iterable[Any]) -> str:

    edges = [_norm_edge(e) for e in round_edges]
    return "[" + ", ".join(f"[{u},{v}]" for (u, v) in edges) + "]"

def _format_solution_json(
    instance_uid: str,
    flips: List[List[List[Any]]],
    meta: Optional[dict] = None,
) -> str:
    meta = {} if meta is None else meta

    lines: List[str] = []
    lines.append("{")
    lines.append('  "content_type": "CGSHOP2026_Solution",')
    lines.append(f'  "instance_uid": {json.dumps(instance_uid, ensure_ascii=False)},')
    lines.append('  "flips": [')

    for i, rounds in enumerate(flips):
        lines.append("    [")
        for r, round_edges in enumerate(rounds):
            round_str = _round_to_str(round_edges)
            comma = "," if r < len(rounds) - 1 else ""
            lines.append(f"      {round_str}{comma}")
        comma2 = "," if i < len(flips) - 1 else ""
        lines.append(f"    ]{comma2}")

    lines.append("  ],")

 
    if meta == {}:
        meta_str = "{  }"
    else:
        meta_str = json.dumps(meta, ensure_ascii=False)

    lines.append(f'  "meta": {meta_str}')
    lines.append("}")
    return "\n".join(lines)

def write_solution_json_pretty_mixed(
    instance_uid: str,
    flips: List[List[List[Any]]],
    out_path: str = "solution.json",
    meta: Optional[dict] = None,
) -> str:
    s = _format_solution_json(instance_uid, flips, meta)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(s)
    return out_path


write_solution_json_pretty_mixed(inst.instance_uid, all_flips_parallel, "solution.json", meta={})
