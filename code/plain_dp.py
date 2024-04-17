import time
from collections import defaultdict
from itertools import combinations

import networkx as nx
from more_itertools import powerset

from share_input import get_shared_parser
from utility import (
    bfs_traversal,
    count_cut,
    delete_subset_bags,
    is_tree_decomp,
    make_tree_directed,
    read_graph,
    td_to_python,
    validate_mis,
    validate_mvc,
)


def plain_mis_dp(G: nx.Graph, td: dict, root):
    def solve_bag(bag: list, G: nx.Graph):
        bag_mISs = set([])
        bag = frozenset(bag)

        subgraph_edges = frozenset(G.subgraph(bag).edges)

        for i in range(len(bag), int(len(bag) / 2) - 1, -1):
            for ele in combinations(bag, i):
                ele = frozenset(ele)
                complement = bag - ele
                flag = True
                flag_complement = True

                # one side
                for u, v in subgraph_edges:
                    if len(ele & frozenset({u, v})) >= 2:
                        flag = False
                        break
                if flag is True:
                    bag_mISs.add(frozenset(ele))

                # the other
                for u, v in subgraph_edges:
                    if len(complement & frozenset({u, v})) >= 2:
                        flag_complement = False
                        break
                if flag_complement is True:
                    bag_mISs.add(frozenset(complement))

        bag_mISs.add(frozenset(()))  # empty set
        return frozenset(bag_mISs)

    # dp tables
    ans = tuple()
    A = defaultdict(dict)
    B = defaultdict(lambda: defaultdict(dict))

    layered_td = bfs_traversal(td, root)
    for bag_id in layered_td:
        bag = td[bag_id]["bag"]
        p_id = td[bag_id]["p"]
        children = td[bag_id]["c"]
        ISs = solve_bag(bag, G)

        for IS in ISs:
            # self-supply
            if IS not in A[bag_id]:
                A[bag_id][IS] = [len(IS), IS]
            # update A table (absorb children)
            for c_id in children:
                child = td[c_id]["bag"]
                c_intersection = IS & child
                A[bag_id][IS][0] += B[c_id][bag_id][c_intersection][0] - len(
                    c_intersection
                )
                A[bag_id][IS][1] |= B[c_id][bag_id][c_intersection][1]
            # update B table (supply to the parent)
            if p_id is not None:
                parent = td[p_id]["bag"]
                p_intersection = IS & parent
                if p_intersection not in B[bag_id][p_id]:
                    B[bag_id][p_id][p_intersection] = A[bag_id][IS]
                B[bag_id][p_id][p_intersection] = max(
                    B[bag_id][p_id][p_intersection], A[bag_id][IS]
                )
    for IS in A[root]:
        ans = max(ans, tuple(A[root][IS]))
    return ans


def plain_mvc_dp(G: nx.Graph, td: dict, root):
    def mvc_state(G: nx.Graph, bag):
        all_states = list(powerset(bag))
        all_edges = list(G.subgraph(bag).edges())
        valid_states = []
        for state in all_states:
            flag = True
            for u, v in all_edges:
                if (u not in state) and (v not in state):
                    flag = False
                    break
            if flag is True:
                valid_states.append(frozenset(state))
        return valid_states

    ans = tuple()
    A = defaultdict(dict)
    B = defaultdict(lambda: defaultdict(dict))
    layered_td = bfs_traversal(td, root)

    for bag_id in layered_td:
        bag = td[bag_id]["bag"]
        p_id = td[bag_id]["p"]
        children = td[bag_id]["c"]
        states = mvc_state(G, bag)
        for state in states:
            if state not in A[bag_id]:
                A[bag_id][state] = [len(state), state]
            # from children
            for c_id in children:
                c_bag = td[c_id]["bag"]
                c_intersection = state & c_bag
                A[bag_id][state][0] += B[c_id][bag_id][c_intersection][0] - len(
                    c_intersection
                )
                A[bag_id][state][1] |= B[c_id][bag_id][c_intersection][1]
            # to parent
            if p_id is not None:
                p_bag = td[p_id]["bag"]
                p_intersection = state & p_bag
                if p_intersection not in B[bag_id][p_id]:
                    B[bag_id][p_id][p_intersection] = A[bag_id][state]
                B[bag_id][p_id][p_intersection] = min(
                    B[bag_id][p_id][p_intersection], A[bag_id][state]
                )
    for state in A[root]:
        ans = min(ans, tuple(A[root][state])) if ans else tuple(A[root][state])
    return ans


def plain_mc_dp(G: dict, td: dict, root):
    """known: ones(N[tm]-tm)"""

    def mc_state(bag):
        """a bit special, no local constraints"""
        states = list(powerset(bag))
        return [frozenset(state) for state in states]

    ans = tuple()
    A = defaultdict(dict)
    B = defaultdict(lambda: defaultdict(dict))
    layered_td = bfs_traversal(td, root)
    for bag_id in layered_td:
        bag = td[bag_id]["bag"]
        p_id = td[bag_id]["p"]
        children = td[bag_id]["c"]
        states = mc_state(bag)
        for state in states:
            if state not in A[bag_id]:
                A[bag_id][state] = [count_cut(G, state, bag), state]
            # from children
            for c_id in children:
                child = td[c_id]["bag"]
                c_intersection = state & child  # state intersection S
                bag_intersection = bag & child  # bag intersection S', S'<=S
                dup_cuts = count_cut(G, c_intersection, bag_intersection)
                A[bag_id][state][0] += B[c_id][bag_id][c_intersection][0] - dup_cuts
                A[bag_id][state][1] |= B[c_id][bag_id][c_intersection][1]
            # to parents
            if p_id is not None:
                parent = td[p_id]["bag"]
                p_intersection = state & parent
                if p_intersection not in B[bag_id][p_id]:
                    B[bag_id][p_id][p_intersection] = A[bag_id][state]
                B[bag_id][p_id][p_intersection] = max(
                    B[bag_id][p_id][p_intersection], A[bag_id][state]
                )
    for r_state in A[root]:
        ans = tuple(A[root][r_state]) if not ans else max(ans, tuple(A[root][r_state]))
    return ans


parser = get_shared_parser()
parser.add_argument("--problem", type=str)
args = parser.parse_args()

g = read_graph(args.input, args.delimiter)

# build td
tw, td = nx.approximation.treewidth_min_degree(g.copy())
print(f"tw: {tw}")
root = max(td.nodes, key=len)
td, root = make_tree_directed(td.copy(), root)
root = delete_subset_bags(td, root)  # simplify tree
td = nx.convert_node_labels_to_integers(td, label_attribute="bag")
root = max(td.nodes, key=lambda n: len(td.nodes[n]["bag"]))
# make directed
td, root = make_tree_directed(td.copy(), root)
# assert is_tree_decomp(g.copy(), td.to_undirected())
td = td_to_python(td.copy())

if args.problem == "mis":
    start_t = time.time()
    my_mis_ans, _ = plain_mis_dp(g, td, root)
    print(f"ans: {my_mis_ans}, time:{time.time() - start_t}")
elif args.problem == "mvc":
    start_t = time.time()
    my_mvc_ans, _ = plain_mvc_dp(g, td, root)
    print(f"ans: {my_mvc_ans}, time:{time.time() - start_t}")
elif args.problem == "mc":
    start_t = time.time()
    my_mc_ans, _ = plain_mc_dp(g, td, root)
    print(f"mc ans: {my_mc_ans}, time:{time.time() - start_t}")
