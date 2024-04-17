import os
import queue
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def read_graph(name, delimiter):
    g = read_single_real_graph(name, delimiter)
    g = nx.convert_node_labels_to_integers(g)
    print(g)
    return g


def random_k_mis(g: nx.Graph, k):
    mis = []
    count = 0
    while g.number_of_nodes() > 0 and count < k:
        n = random.choice(list(g.nodes))
        nbr = get_neighbors(g, [n])
        g.remove_nodes_from(nbr | {n})

        mis.append(n)
        count += len(nbr) + 1
    return mis


def parse_filename(directory_path, key):
    experiment_data = []
    # Listing all files in the given directory
    try:
        filenames = os.listdir(directory_path)
    except FileNotFoundError:
        print(f"The directory '{directory_path}' was not found.")
        return []
    for filename in filenames:
        # Removing the file extension if present
        filename_without_extension = os.path.splitext(filename)[0]
        # Splitting the filename using underscore as delimiter
        parts = filename_without_extension.split("_")
        if len(parts) != 3:
            print(f"Filename '{filename}' does not follow the expected pattern.")
            continue
        # Unpacking the parts into respective variables
        etype, input_data_name, algorithm = parts
        if (etype, input_data_name) == key:
            # Adding the tuple to the list
            experiment_data.append(
                (pd.read_pickle(f"{directory_path}/{filename}"), algorithm)
            )
    return experiment_data


def draw_iteration_fitness(title, data_list, path, iteration_threshold):
    for data in data_list:
        d, l = data
        filtered_d = d[d.index <= iteration_threshold]
        plt.plot(
            filtered_d.index,
            filtered_d["Fitness"],
            label=l,
        )
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Value")
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()


def draw_timestamp_fitness(title, data_list, path, timestamp_threshold):
    for data in data_list:
        d, l = data
        # Slice the DataFrame to include only rows where Timestamp <= timestamp_threshold
        filtered_d = d[d["Timestamp"] <= timestamp_threshold]
        plt.plot(
            filtered_d["Timestamp"],
            filtered_d["Fitness"],
            label=l,
        )
    plt.xlabel("Timestamp")
    plt.ylabel("Fitness Value")
    plt.title(title + f", threshold: {timestamp_threshold}s")
    plt.legend()
    plt.savefig(path)
    plt.close()


def td_to_python(td: nx.DiGraph):
    """turn td to a dict"""
    tree = dict(dict())  # no other way
    for n in td.nodes:
        pid = list(td.predecessors(n))[0] if list(td.predecessors(n)) else None
        cid = list(td.successors(n))
        bag = td.nodes[n]["bag"]
        tree[n] = {}
        tree[n]["p"] = pid
        tree[n]["c"] = cid
        tree[n]["bag"] = bag
    return tree


def bfs_traversal(td: dict, root):
    # assert td[root]["p"] is None
    q = queue.Queue()
    q.put(root)
    seq = []
    while q.empty() == False:
        nid = q.get()
        # assert isinstance(nid, int)
        seq.append(nid)
        children = td[nid]["c"]
        for c in children:
            # assert len(children) > 0
            q.put(c)
    return list(reversed(seq))


# def get_bag(td: dict):
#     return [n["bag"] for n in td.values()]


def get_attributes(G, attr):
    return {node: G.nodes[node][attr] for node in G.nodes()}


# def largebag_subtree(td: nx.DiGraph):
#     large_bags = []
#     for n in td.nodes:
#         c_tw = len(n)
#         if c_tw >= tw_upper:
#             large_bags.append(n)
#     return td.subgraph(large_bags).copy()


# def cut_td(td: nx.DiGraph):
#     cut_edges = []
#     for n in td.nodes:
#         parent = list(td.predecessors(n))[0] if list(td.predecessors(n)) else None
#         assert isinstance(parent, (int, None))
#         if parent is not None:
#             if td.nodes[parent]["bag"] & td.nodes[n]["bag"]:
#                 cut_edges.append((n, parent))

#     td.remove_edges_from(cut_edges)
#     return td


def make_tree_directed(tree, root):
    d_tree = nx.DiGraph()
    d_tree.add_nodes_from(tree.nodes(data=True))
    for start, end in nx.bfs_edges(tree, source=root):
        d_tree.add_edge(start, end)
    return d_tree, root


# def nested_dict():
#     return defaultdict(nested_dict)


def read_single_real_graph(path, delim):
    G = nx.Graph()
    with open(path, "rb") as file:
        G = nx.read_edgelist(path=file, delimiter=delim, data=False)
    file.close()
    # remove self-loops
    self_loops = []
    for n in G.nodes:
        if (n, n) in G.edges:
            self_loops.append((n, n))
    G.remove_edges_from(self_loops)
    return G


def get_neighbors(G: nx.Graph, vertices):
    nbrs = set()
    for i in vertices:
        nbrs |= set(G.neighbors(i))
    return nbrs - set(vertices)


def is_tree_decomp(G, td):
    """Check if the given tree decomposition is valid."""
    for x in G.nodes():
        appear_once = False
        for id, val in td.nodes(data=True):
            if x in val["bag"]:
                appear_once = True
                break
        assert appear_once

    # Check if each connected pair of nodes are at least once together in a bag
    for x, y in G.edges():
        appear_together = False
        for id, val in td.nodes(data=True):
            if x in val["bag"] and y in val["bag"]:
                appear_together = True
                break
        assert appear_together

    # Check if the nodes associated with vertex v form a connected subset of T
    for v in G.nodes():
        subset = []
        for id, val in td.nodes(data=True):
            if v in val["bag"]:
                subset.append(id)
        sub_graph = td.subgraph(subset)
        assert nx.is_connected(sub_graph)

    # is tree?
    assert nx.is_tree(td)

    print(f"All tree decompo test passed!")
    return True


def delete_subset_bags(td: nx.DiGraph, root):
    """adjust the td into a simplified version, there is no inclusion bags after"""
    q = queue.Queue()
    q.put(root)
    new_root = root
    while not q.empty():
        current = q.get()

        # child includes parent
        successors = set(td.successors(current)) if current in td.nodes else set([])
        max_successor = current
        for s in successors:
            if s > max_successor:  # impossible to get two euqal bags
                max_successor = s
        if max_successor != current:  # has changed
            new_edges = []
            parent = list(td.predecessors(current))
            if parent:
                q.put(parent[0])
                new_edges.append((parent[0], max_successor))
            children = successors - {max_successor}
            new_edges += [(max_successor, child) for child in children]
            td.add_edges_from(new_edges)
            td.remove_node(current)

        # parent includes child
        new_edges = []
        for s in successors - {max_successor}:
            if max_successor >= s:
                new_edges += [
                    (max_successor, child) for child in list(td.successors(s))
                ]
                td.remove_node(s)
            else:
                q.put(s)
        td.add_edges_from(new_edges)

    assert td.number_of_edges() == td.number_of_nodes() - 1
    for n in td.nodes:
        if len(list(td.predecessors(n))) == 0:
            new_root = n
            break
    return new_root


def reduce_td(td: dict, tm):
    for id, val in td.items():
        val["bag"] -= tm


def validate_mis(G: nx.Graph, solution: set):
    """check if the solution is legit"""
    num_edges = 0
    for u, v in G.edges:
        if u in solution and v in solution:
            num_edges += 1
    return num_edges


def validate_mvc(G: nx.Graph, solution: set):
    uncovered_edges = 0
    for u, v in G.edges:
        if u not in solution and v not in solution:
            uncovered_edges += 1
    return uncovered_edges


def td_info(G, td: nx.DiGraph):
    print()
    print(f"#################### td info")
    print(f"#total bags in td: {td.number_of_nodes()}")
    avg_tw = 0
    max_tw = -1
    num_max_tw = 0
    num_large_bags = 0
    lb_nodes = set()
    sb_nodes = set()
    cc = 1
    for n in td.nodes:
        c_tw = len(td.nodes[n]["bag"])
        avg_tw += c_tw
        max_tw = max(max_tw, c_tw)
        if c_tw > tw_upper:
            num_large_bags += 1
            lb_nodes |= td.nodes[n]["bag"]
        else:
            sb_nodes |= td.nodes[n]["bag"]
        parent = list(td.predecessors(n))[0] if list(td.predecessors(n)) else None
        if parent is not None:
            if not td.nodes[parent]["bag"] & td.nodes[n]["bag"]:
                cc += 1
    avg_tw /= td.number_of_nodes()
    print(f"#large bags: {num_large_bags}")
    print(f"larg bag ONLY nodes: {len(lb_nodes-sb_nodes)/G.number_of_nodes()}")
    print(f"#cc: {cc}")
    print(f"average tw: {avg_tw}")
    print(f"max tw: {max_tw-1}")
    print()


# --------------------------------------------------------------------- (1+1)EA
def ranged_mutation(bitstring, mask, mutation_rate):
    """mask is bitstring, 1 -- mutable bit, 0 -- immutable bit"""
    # bitstring_copy = bitstring.copy()
    mask = mask.astype(bool)  # mask has to be bool
    prob_mask = np.random.rand(*mask.shape)
    # if prob is 1, it will never to be considered for any bit-flips (all 0 bits in the original mask)
    prob_mask[~mask] = 1
    mutation_mask = prob_mask < mutation_rate
    bitstring[mutation_mask] ^= 1

    # debug
    # assert mutation_mask.dtype == bool
    # assert np.array_equal(bitstring_copy[~mask], bitstring[~mask])
    return bitstring


def one_plus_one_mutation(population, mutation_rate):
    """regular mutation operator"""
    mutation_prob = np.random.rand(*population.shape)
    mutation_mask = mutation_prob < mutation_rate
    population[mutation_mask] ^= 1
    return population


def opo_mis_fitness(G, bitstring):
    # one individual
    num_of_nodes = np.sum(bitstring, axis=-1).item()
    num_of_edges = 0
    for u, v in G.edges:
        if bitstring[u] == 1 and bitstring[v] == 1:
            num_of_edges += 1
    if num_of_edges != 0:
        return -num_of_edges
    else:
        return num_of_nodes
    # return (-num_of_edges, num_of_nodes)


def opo_mvc_fitness(G, bitstring):
    """increasing from neg to pos"""
    total_nodes = G.number_of_nodes()
    uncovered_e = 0
    num_v = np.sum(bitstring, axis=-1).item()
    # assert isinstance(num_v, int)
    for u, v in G.edges:
        if bitstring[u] == 0 and bitstring[v] == 0:
            uncovered_e += 1
    # if uncovered_e > 0:
    #     return -uncovered_e  # - G.number_of_nodes()
    # else:
    #     return total_nodes - num_v
    if uncovered_e > 0:
        return -uncovered_e - G.number_of_nodes()
    else:
        return -num_v


def opo_mc_fitness(G, bitstring):
    """increasing"""
    X = set(np.where(bitstring == 1)[0])
    Y = G.nodes - X
    cut_size = 0
    for u, v in G.edges:
        if (u in X and v in Y) or (v in X and u in Y):
            cut_size += 1
    return cut_size


# ------------------------------------------------------------------------------
def count_cut(g, state, bag):
    """state: pick nodes
    bag: total local area
    """
    cut_edges = 0
    non_state = bag - state
    for u in state:
        u_nbrs = set(g[u])
        for v in u_nbrs:
            if v in non_state:
                cut_edges += 1
        # for v in non_state:
        #     if v in u_nbrs:
        #         cut_edges += 1
    return cut_edges


def init_string(G, problem, require, num_n):
    """require: %of the graph to initiate"""
    # all init
    if require == 0:
        if problem == "mis":
            return np.zeros(num_n, dtype=int)
        elif problem == "mvc":
            return np.ones(num_n, dtype=int)
        elif problem == "mc":
            return np.zeros(num_n, dtype=int)
    elif require == 1:
        if problem == "mis":
            return random_mis(G, require)
        elif problem == "mvc":
            return random_mvc(G, require)
        elif problem == "mc":
            return np.random.binomial(n=1, p=0.5, size=(num_n,))
    elif require > 0 and require < 1:
        if problem == "mis":
            return random_greedy_mis(G, require)
        elif problem == "mvc":
            return random_greedy_mvc(G, require)
        elif problem == "mc":
            return random_greedy_mc(G, require)
    else:
        print("input error")
        exit()


def random_greedy_mis(G: nx.Graph, percentage):
    init_b = np.zeros(G.number_of_nodes(), dtype=int)
    subg_n = []
    for n in G.nodes:
        if np.random.random() < percentage:
            subg_n.append(n)
    subg = G.subgraph(subg_n).copy()
    while subg.number_of_nodes() > 0:
        v = min(subg.degree, key=lambda x: x[1])[0]
        nbrs = list(get_neighbors(subg, [v]))
        init_b[v] = 1
        subg.remove_nodes_from(nbrs + [v])
    # for i, _ in enumerate(init_b):
    #     if init_b[i] == 1 and np.random.random() > percentage:
    #         init_b[i] = 0
    return init_b


def random_mis(G: nx.Graph, percentage):
    """
    give init string a bit push
    percentage: portion for 1s, how large the graph will we initially cover?
    """
    init_b = np.ones(G.number_of_nodes(), dtype=int)
    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    for i in nodes:
        if init_b[i] == 1:
            nbrs = list(get_neighbors(G, [i]))
            init_b[nbrs] = 0
    return init_b


def random_greedy_mvc(G: nx.Graph, percentage):
    init_b = np.ones(G.number_of_nodes(), dtype=int)
    subg_n = []
    for n in G.nodes:
        if np.random.random() < percentage:
            subg_n.append(n)
    init_b[subg_n] = 0
    subg = G.subgraph(subg_n).copy()
    while subg.number_of_edges() > 0:
        v = max(subg.degree, key=lambda x: x[1])[0]
        init_b[v] = 1
        subg.remove_node(v)
    # for i, _ in enumerate(init_b):
    #     if init_b[i] == 1 and np.random.random() > percentage:
    #         init_b[i] = 0
    return init_b


def random_mvc(G: nx.Graph, percentage):
    """give init string a bit push for mvc"""
    init_b = np.zeros(G.number_of_nodes(), dtype=int)
    for u, v in G.edges:
        if init_b[u] == 0 and init_b[v] == 0:
            if np.random.random() > 0.5:
                init_b[u] = 1
            else:
                init_b[v] = 1
    return init_b


def random_greedy_mc(G: nx.Graph, percentage):
    init_b = np.random.binomial(n=1, p=0.5, size=(G.number_of_nodes(),))
    A = set(np.where(init_b == 1)[0])
    B = set(np.where(init_b == 0)[0])
    subg_n = []
    for n in G.nodes:
        if np.random.random() < percentage:
            subg_n.append(n)
    subg = G.subgraph(subg_n).copy()
    for n in subg.nodes:
        nbrs = get_neighbors(subg, [n])
        if len(nbrs & A) >= len(nbrs & B) and n in A:
            init_b[n] = 1 - init_b[n]
        elif len(nbrs & A) < len(nbrs & B) and n in B:
            init_b[n] = 1 - init_b[n]
    return init_b


def random_mc(G: nx.Graph, percentage):
    """give init string a bit push for mc (random local cut)"""
    num_n = G.number_of_nodes()
    random_subgraph_b = np.random.binomial(n=1, p=percentage, size=(num_n,))
    for i, v in enumerate(random_subgraph_b):
        if v == 1 and np.random.rand() > 0.5:
            random_subgraph_b[i] = 0
    return random_subgraph_b  # init string


def get_boundaries(G: nx.Graph, tm):
    """calculate the boundary nodes"""
    boundaries = set()
    for v in tm:
        if len(set(G[v]) - tm) > 0:
            boundaries |= {v}
    return boundaries


def cc_count(G, tm):
    subg = G.subgraph(tm).copy()
    ccs = list(nx.connected_components(subg))
    print(f"tm has {len(ccs)} ccs")


def absorb_subset(td: nx.DiGraph, root, tw, tw_ub, lb_threshold):
    """
    tw: tw of the given td
    lb_threshold > tw_upper, in general it should be larger
    but lb_threshold < tw
    """
    layered_td = sorted(list(nx.bfs_layers(td, root)), reverse=True)
    layered_td = sum(layered_td, [])
    for bag_id in layered_td:
        if bag_id not in td.nodes:
            continue
        bag = td.nodes[bag_id]["bag"]
        if len(bag) >= lb_threshold:
            cs = list(td.successors(bag_id))
            delete_c = []
            new_edges = []
            for c in cs:
                if len(td.nodes[bag_id]["bag"]) >= tw:
                    break
                c_bag = td.nodes[c]["bag"]
                if len(c_bag) >= tw_ub or td.nodes[bag_id]["bag"] >= c_bag:
                    td.nodes[bag_id]["bag"] |= c_bag  # absorb children
                    delete_c.append(c)  # delete children
                    # add new edges current_bag ---> grand children
                    new_edges += [(bag_id, gc) for gc in td.successors(c)]
            td.add_edges_from(new_edges)
            td.remove_nodes_from(delete_c)


def absorb_subset_2(td: nx.DiGraph, root, lb_threshold):
    layered_td = sorted(list(nx.bfs_layers(td, root)), reverse=True)
    layered_td = sum(layered_td, [])
    lbs = queue.Queue()

    for bag_id in layered_td:
        bag = td.nodes[bag_id]["bag"]
        if len(bag) >= lb_threshold:
            lbs.put(bag_id)

    while not lbs.empty():
        current_bag_id = lbs.get()
        bag = td.nodes[current_bag_id]["bag"]
        p = list(td.predecessors(current_bag_id))
        if p and len(td.nodes[p[0]]["bag"]) >= lb_threshold:
            td.nodes[p[0]]["bag"] |= bag
            children = list(td.successors(current_bag_id))
            new_edges = [(p[0], c) for c in children]
            td.add_edges_from(new_edges)
            td.remove_node(current_bag_id)
