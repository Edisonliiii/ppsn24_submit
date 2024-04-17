import time

import networkx as nx
import numpy as np

import setting
from dp import mc_dp
from share_input import get_shared_parser
from tm_heuristics import get_td_and_tm
from utility import (
    count_cut,
    get_boundaries,
    init_string,
    opo_mc_fitness,
    ranged_mutation,
    read_graph,
    reduce_td,
)


def zoomin_mc_fitness(g: dict, td: dict, root, tm, mc_b, tm_b):
    """fitness func for zoomin method
    tm_v: original tm without nbrs
    tm_b: tm + nbrs bitstring
    """
    tm_inner, tm_boundaries = tm
    # assert len(tm_v & tm_nbrs_v) == 0
    # assert set(np.where(tm_b == 1)[0]) == tm_v | tm_nbrs_v
    zoomed_one_v = set(np.where(tm_b & mc_b == 1)[0])
    known_ones = zoomed_one_v & tm_boundaries  # ones on boundary
    known_zeros = tm_boundaries - known_ones  # zeros on boundary
    _, out_zoomed_one_v = mc_dp(g, td, root, (known_ones, known_zeros))
    total_v = out_zoomed_one_v | zoomed_one_v
    return count_cut(g, total_v, set(g.keys())), total_v
    # return count_cut(g, total_v, set()), total_v


def opo_single_mask_mc(G: nx.Graph, td: dict, root, num_generations, tm_b, mc_b):
    """tm, mc both bitstring"""
    mc_b = mc_b[0] if mc_b.ndim > 1 else mc_b
    tm_v = set(np.where(tm_b == 1)[0])  # original tm
    tm_boundaries = get_boundaries(G, tm_v)
    tm_inner = tm_v - tm_boundaries
    # assert tm_v == (tm_inner | tm_boundaries)
    # print(f"mutation range: {len(tm_inner | tm_boundaries)}")
    zoomed_mutation_rate = 1 / len(tm_v) if len(tm_v) > 0 else 0
    dict_g = nx.to_dict_of_lists(G)
    reduce_td(td, tm_inner)
    best_score = opo_mc_fitness(G, mc_b)
    print(f"[opo single mask mc] round: {0}, ans: {best_score}, time: {0.000000}")
    start_time = time.time()
    for i in range(1, num_generations):
        # if time.time() - start_time >= 3600.0:
        #     break
        mutated_mc = ranged_mutation(mc_b.copy(), tm_b, zoomed_mutation_rate)
        mutated_score, v_set = zoomin_mc_fitness(
            dict_g,
            td,
            root,
            (tm_inner, tm_boundaries),
            mutated_mc,
            tm_b,
        )
        if best_score <= mutated_score:
            mc_b = mutated_mc
            best_score = mutated_score
        # debug & record
        print(
            f"[opo single mask mc] round: {i}, ans: {best_score}, time: {time.time() - start_time}"
        )
    return mc_b


# get input
parser = get_shared_parser()
parser.add_argument("--iteration", type=int)
parser.add_argument("--init-string", type=float)
parser.add_argument("--tw-ub", type=int)
parser.add_argument("--lb-threshold", type=int)
args = parser.parse_args()
# build graph and gurobi answer
g = read_graph(args.input, args.delimiter)
# build td and tm
td, root, tm_b, tm_v = get_td_and_tm(g, args.tw_ub, args.lb_threshold)
# run
init_mc_b = init_string(g.copy(), "mc", args.init_string, g.number_of_nodes())
opo_single_mask_mc(g, td, root, args.iteration, tm_b, init_mc_b)
