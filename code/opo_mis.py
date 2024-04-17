import time

import networkx as nx
import numpy as np

import setting
from share_input import get_shared_parser
from utility import (
    init_string,
    one_plus_one_mutation,
    opo_mis_fitness,
    read_graph,
    validate_mis,
)


def opl_ea_mis(G: nx.Graph, num_generations, init_solution, lambda_v):
    mutation_rate = 1 / G.number_of_nodes()
    best_bitstring = init_solution
    best_score = opo_mis_fitness(G, best_bitstring)
    best_mis = set()
    start_time = time.time()
    for i in range(1, num_generations):
        # if time.time() - start_time >= 3600.0:
        #     break
        lambda_bitstring = best_bitstring
        for _ in range(lambda_v):
            mutated_bitstring = one_plus_one_mutation(
                best_bitstring.copy(), mutation_rate
            )
            mutated_score = opo_mis_fitness(G, mutated_bitstring)
            if best_score <= mutated_score:
                lambda_bitstring = mutated_bitstring
                best_score = mutated_score
        best_bitstring = lambda_bitstring
        best_mis = set(np.where(best_bitstring == 1)[0])
        if (i + 1) % 100 == 0:
            print(
                f"[opo ea mis]: round: {i}, ans: {len(best_mis)}, time: {time.time()-start_time}"
            )


def opo_ea_mis(G: nx.Graph, num_generations, init_solution):
    """(1+1) EA solving MIS"""
    mutation_rate = 1 / G.number_of_nodes()
    best_bitstring = init_solution
    best_score = opo_mis_fitness(G, best_bitstring)
    best_mis = set(np.where(best_bitstring == 1)[0])
    print(f"[opo ea mis]: round: {0}, ans: {len(best_mis)}, time: {0.00000}")
    start_time = time.time()
    for i in range(1, num_generations):
        # if time.time() - start_time >= 3600.0:
        #     break
        mutated_bitstring = one_plus_one_mutation(best_bitstring.copy(), mutation_rate)
        mutated_score = opo_mis_fitness(G, mutated_bitstring)
        if best_score <= mutated_score:
            best_bitstring = mutated_bitstring
            best_score = mutated_score
            best_mis = set(np.where(mutated_bitstring == 1)[0])
        if (i + 1) % 100 == 0:
            print(
                f"[opo ea mis]: round: {i}, ans: {len(best_mis)}, time: {time.time()-start_time}"
            )
    return best_bitstring


parser = get_shared_parser()
parser.add_argument("--iteration", type=int)
parser.add_argument("--init-string", type=float)
parser.add_argument("--lambda-v", type=int)
args = parser.parse_args()
# build graph and gurobi answer
g = read_graph(args.input, args.delimiter)
init_bitstring = init_string(g.copy(), "mis", args.init_string, g.number_of_nodes())
if args.lambda_v == 1:
    opo_ea_mis(g, args.iteration, init_bitstring)
else:
    opl_ea_mis(g, args.iteration, init_bitstring, args.lambda_v)
