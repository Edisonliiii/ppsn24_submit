import math
from time import sleep

import networkx as nx
import numpy as np

from gurobi_solver import gb_treewidth_modulator
from utility import (
    absorb_subset,
    absorb_subset_2,
    cc_count,
    delete_subset_bags,
    get_attributes,
    is_tree_decomp,
    make_tree_directed,
    one_plus_one_mutation,
    td_to_python,
)


def tm_fitness(G: nx.Graph, bags: dict, reduce_nodes):
    fitness = 0
    total_nodes = G.number_of_nodes()
    for _, b in bags.items():
        b_len = len(b - reduce_nodes)
        if b_len > tw_upper:
            fitness += b_len - tw_upper
        else:
            b -= reduce_nodes
    if fitness == 0:
        fitness = total_nodes - len(reduce_nodes)
        return (fitness, bags)
    else:
        return -fitness


def opo_given_tm(G: nx.Graph, td: nx.DiGraph, num_generations):
    mutation_rate = 1 / G.number_of_nodes()
    # best_bitstring = np.random.randint(0, 2, (1, G.number_of_nodes()))[0]
    best_bitstring = np.ones(G.number_of_nodes(), dtype=int)
    best_score = -math.inf
    best_tm = set()
    bags = get_attributes(td, "bag")
    for i in range(num_generations):
        mutated_solution = one_plus_one_mutation(best_bitstring.copy(), mutation_rate)
        reduce_nodes = frozenset(np.where(mutated_solution == 1)[0])
        mutated_return = tm_fitness(G, bags.copy(), reduce_nodes)
        if isinstance(mutated_return, tuple):
            mutated_score, new_bags = mutated_return
        else:
            mutated_score = mutated_return
        if best_score <= mutated_score:
            best_bitstring = mutated_solution
            best_score = mutated_score
            best_tm = reduce_nodes
        print(
            f"round: {i}, current score: {mutated_score}, best tm len: {len(best_tm)}"
        )
    return best_bitstring


def exact_tm(g: nx.Graph, td: nx.DiGraph, tw_ub):
    """tw_ub: upper bound for tw"""
    gb_tm = gb_treewidth_modulator(g.copy(), td.copy(), tw_ub)
    print(f"gb tm: {len(gb_tm)}")
    tm_b = np.zeros(g.number_of_nodes(), dtype=int)
    tm_v = frozenset(gb_tm.copy())
    tm_b[list(tm_v)] = 1
    print(f"tm length: {len(tm_v)}")
    # sleep(3)
    return tm_b, tm_v


def evo_tm(g, td, generation):
    tm_b = opo_given_tm(g.copy(), td.copy(), generation)
    tm_v = frozenset(np.where(tm_b == 1)[0])
    print(f"tm length: {len(tm_v)}")
    return tm_b, tm_v


def get_td_and_tm(g, tw_ub, lb_threshold):
    """
    tw_ub: tree width upper bound, used for calculating TM
    lb_threshold: if tw>lb_threshold, we do bag absorb to reduce the total number of large bags
                  this operation will make the entire DP faster.
                  For example, if lb_threshold=50, it means, we do bag absorb for all large bags whose width>=50
    NOTE: large bag absorb will happen before calculating the TM, because it makes sense to do so. If we use it after
          knowing the TM, we don't really need it anymore considering all bags are below tw_ub.
          (also can be after, need more consideration, e.g. in that way, we should have lb_threshold <= tw_ub)
    """
    tw, td = nx.approximation.treewidth_min_degree(g.copy())
    # tw, td = nx.approximation.treewidth_min_fill_in(g.copy())
    print(f"tw: {tw}")
    tw_ub = min(tw_ub, tw) + 1
    # simplify td
    root = max(td.nodes, key=len)
    td, root = make_tree_directed(td.copy(), root)
    root = delete_subset_bags(td, root)

    # convert label
    td = nx.convert_node_labels_to_integers(td, label_attribute="bag")
    # find root
    root = max(td.nodes, key=lambda n: len(td.nodes[n]["bag"]))
    # make directed
    td, root = make_tree_directed(td.copy(), root)

    # absorb to reduce #large bags
    absorb_subset(td, root, tw, tw_ub, lb_threshold)
    # absorb_subset_2(td, root, lb_threshold)
    # assert is_tree_decomp(g.copy(), td.to_undirected()) == True

    # get tm
    tm_b, tm_v = exact_tm(g.copy(), td.copy(), tw_ub)
    # tm_b, tm_v = evo_tm(g.copy(), td.copy(), 2500)
    # turn nx to dict
    td = td_to_python(td.copy())

    # cc_count(g, tm_v)
    return td, root, tm_b, tm_v
