import time

import networkx as nx

from setting import gp

# gurobi solvers


def gb_treewidth_modulator(G, td: nx.Graph, k):
    """k: the upper bound of bag length"""
    model = gp.Model()
    g_vertices = sorted(list(G.nodes))
    g_v = model.addVars(g_vertices, vtype=gp.GRB.BINARY)
    model.setObjective(gp.quicksum(g_v[n] for n in g_v), gp.GRB.MINIMIZE)
    for b in td.nodes:
        bag = td.nodes[b]["bag"]
        model.addConstr(gp.quicksum(g_v[v] for v in bag) >= (len(bag) - k))
        model.addConstr(gp.quicksum(g_v[v] for v in bag) <= (len(bag)))
    model.optimize()
    selected_v = [n for n in g_vertices if g_v[n].x >= 0.5]
    return selected_v


def gb_mis(G):
    model = gp.Model()
    node = model.addVars(G.nodes, vtype=gp.GRB.BINARY)
    model.setObjective(gp.quicksum(node[n] for n in G.nodes), gp.GRB.MAXIMIZE)
    model.addConstrs(node[u] + node[v] <= 1 for u, v in G.edges)
    model.optimize()
    selected_nodes = [n for n in G.nodes if node[n].x >= 0.5]
    return selected_nodes


def gb_mvc(G):
    model = gp.Model()
    node = model.addVars(G.nodes, vtype=gp.GRB.BINARY)
    model.setObjective(gp.quicksum(node[n] for n in G.nodes), gp.GRB.MINIMIZE)
    model.addConstrs(node[u] + node[v] >= 1 for u, v in G.edges)
    model.optimize()
    selected_nodes = [n for n in G.nodes if node[n].x >= 0.5]
    return selected_nodes


def gb_mc(G):
    model = gp.Model()
    node = model.addVars(G.nodes, vtype=gp.GRB.BINARY)
    model.setObjective(
        gp.quicksum(node[u] + node[v] - 2 * node[u] * node[v] for u, v in G.edges),
        gp.GRB.MAXIMIZE,
    )
    model.optimize()
    selected_nodes = [n for n in G.nodes if node[n].x >= 0.5]
    return selected_nodes


def gb_coloring(G):
    model = gp.Model()
    node = model.addVars(G.nodes, vtype=gp.GRB.INTEGRE, lb=0, ub=2)
    model.addConstrs(node[u] != node[v] for u, v in G.edges)
    model.optimize()
    if model.status == gp.GRB.OPTIMAL:
        coloring = {n: node[n].x for n in G.nodes}
        return True, coloring
    else:
        return False, None


def gb_mds(G):
    pass
