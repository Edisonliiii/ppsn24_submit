import pickle
import time

import networkx as nx
import numpy as np

import setting
from dp import mvc_dp
from share_input import get_shared_parser
from tm_heuristics import get_td_and_tm
from utility import (
    get_neighbors,
    init_string,
    opo_mvc_fitness,
    ranged_mutation,
    read_graph,
    reduce_td,
    validate_mvc,
)


def zoomin_mvc_fitness(g: nx.Graph, td: dict, root, tm_v, mvc_b, tm_b):
    zoomed_one_v = set(np.where(tm_b & mvc_b == 1)[0])
    zoomed_zero_v = tm_v - zoomed_one_v
    zoomed_zero_v_nbrs = get_neighbors(g, zoomed_zero_v) - tm_v
    zoomed_one_b = np.zeros_like(mvc_b)
    zoomed_one_v |= zoomed_zero_v_nbrs
    if zoomed_one_v:
        zoomed_one_b[np.array(list(zoomed_one_v))] = 1
    in_lens_score = opo_mvc_fitness(g.subgraph(tm_v | zoomed_zero_v_nbrs), zoomed_one_b)
    # remove tm + outside in-solution's nbrs
    reduce_td(td, tm_v | zoomed_zero_v_nbrs)
    _, out_zoomed_one_v = mvc_dp(g, td, root)
    out_lens_score = len(out_zoomed_one_v)
    total_score = in_lens_score - out_lens_score
    return total_score, zoomed_one_v | out_zoomed_one_v


def opo_single_mask_mvc(G: nx.Graph, td: dict, root, num_generations, tm_b, mvc_b):
    mvc_b = mvc_b[0] if mvc_b.ndim > 1 else mvc_b
    tm_v = set(np.where(tm_b == 1)[0])
    zoomed_mutation_rate = 1 / len(tm_v) if len(tm_v) > 0 else 0
    best_score = opo_mvc_fitness(G, mvc_b)
    best_mvc = set(np.where(mvc_b == 1)[0])
    print(f"[opo single mask mvc] round: {0}, ans: {len(best_mvc)}, time: {0.000000}")
    start_time = time.time()
    for i in range(1, num_generations):
        # if time.time() - start_time >= 3600.0:
        #     break
        mutated_mvc = ranged_mutation(mvc_b.copy(), tm_b, zoomed_mutation_rate)
        mutated_score, v_set = zoomin_mvc_fitness(
            G, pickle.loads(pickle.dumps(td)), root, tm_v, mutated_mvc, tm_b
        )
        if best_score <= mutated_score:
            mvc_b = mutated_mvc
            best_score = mutated_score
            best_mvc = v_set
        # debug & record
        print(
            f"[opo single mask mvc] round: {i}, ans: {len(best_mvc)}, time: {time.time() - start_time}"
        )
    return mvc_b


# get input
parser = get_shared_parser()
parser.add_argument("--iteration", type=int)
parser.add_argument("--init-string", type=float)
parser.add_argument("--tw-ub", type=int)
parser.add_argument("--lb-threshold", type=int)
args = parser.parse_args()
g = read_graph(args.input, args.delimiter)
# build td and tm
td, root, tm_b, tm_v = get_td_and_tm(g.copy(), args.tw_ub, args.lb_threshold)
# run
init_mis_b = init_string(g.copy(), "mvc", args.init_string, g.number_of_nodes())
opo_single_mask_mvc(g, td, root, args.iteration, tm_b, init_mis_b)
