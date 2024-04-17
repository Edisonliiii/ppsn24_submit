import time

import networkx as nx
import numpy as np

import setting
from share_input import get_shared_parser
from utility import init_string, one_plus_one_mutation, opo_mc_fitness, read_graph


def opl_ea_mc(G: nx.Graph, num_generations, init_solution, lambda_v):
    """(1+\lambda)EA solving MC"""
    mutation_rate = 1 / G.number_of_nodes()
    best_bitstring = init_solution
    best_score = opo_mc_fitness(G, best_bitstring)
    best_mc = set()
    start_time = time.time()
    for i in range(1, num_generations):
        # if time.time() - start_time >= 3600.0:
        #     break
        lambda_bitstring = best_bitstring
        for _ in range(lambda_v):
            mutated_bitstring = one_plus_one_mutation(
                best_bitstring.copy(), mutation_rate
            )
            mutated_score = opo_mc_fitness(G, mutated_bitstring)
            if best_score <= mutated_score:
                lambda_bitstring = mutated_bitstring
                best_score = mutated_score
        best_bitstring = lambda_bitstring
        best_mc = set(np.where(best_bitstring == 1)[0])
        if (i + 1) % 100 == 0:
            print(
                f"[opl ea mc]: round: {i}, ans: {best_score}, time: {time.time() - start_time}"
            )
    return best_bitstring


def opo_ea_mc(G: nx.Graph, num_generations, init_solution):
    """(1+1)EA solving MC"""
    mutation_rate = 1 / G.number_of_nodes()
    best_bitstring = init_solution
    best_score = opo_mc_fitness(G, best_bitstring)
    best_mc = set(np.where(best_bitstring == 1)[0])
    print(f"[opo ea mc]: round: {0}, ans: {best_score}, time: {0.00000000}")
    start_time = time.time()
    for i in range(1, num_generations):
        # if time.time() - start_time >= 3600.0:
        #     break
        mutated_bitstring = one_plus_one_mutation(best_bitstring.copy(), mutation_rate)
        mutated_score = opo_mc_fitness(G, mutated_bitstring)
        if best_score <= mutated_score:
            best_bitstring = mutated_bitstring
            best_score = mutated_score
            best_mc = set(np.where(mutated_bitstring == 1)[0])
        if (i + 1) % 100 == 0:
            print(
                f"[opo ea mc]: round: {i}, ans: {best_score}, time: {time.time() - start_time}"
            )
    return best_bitstring


# get input
parser = get_shared_parser()
parser.add_argument("--iteration", type=int)
parser.add_argument("--init-string", type=float)
parser.add_argument("--lambda-v", type=int)
args = parser.parse_args()
# run
g = read_graph(args.input, args.delimiter)
init_bitstring = init_string(g.copy(), "mc", args.init_string, g.number_of_nodes())
if args.lambda_v == 1:
    opo_ea_mc(g, args.iteration, init_bitstring)
else:
    opl_ea_mc(g, args.iteration, init_bitstring, args.lambda_v)
