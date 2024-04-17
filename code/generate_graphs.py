import networkx as nx


def store_g(content, filename):
    with open(filename, "w") as file:
        for line in content:
            file.write(line + "\n")


def reg(num_n, deg, seed):
    G = nx.generators.random_regular_graph(deg, num_n, seed=seed)
    lines = nx.generate_edgelist(G, data=False)
    filename = f"data/reg_{num_n}_{deg}_{seed}.rawg"
    store_g(lines, filename)


def erdos(num_n, deg, seed):
    edge_p = deg / num_n
    G = nx.generators.erdos_renyi_graph(num_n, edge_p, seed=seed)
    lines = nx.generate_edgelist(G, data=False)
    filename = f"data/er_{num_n}_{deg}_{seed}.rawg"
    for n in G.nodes:
        if G.degree(n) == 0:
            G.add_edge(n, n)
    store_g(lines, filename)


seed = 100

# ## small size
# # reg(50, 3, seed)
# # reg(60, 3, seed)
# # reg(70, 3, seed)
# # reg(80, 3, seed)
# # reg(90, 3, seed)
# reg(100, 6, seed)
# reg(120, 6, seed)
# reg(140, 6, seed)
# reg(160, 6, seed)
# # reg(180, 3, seed)
# # reg(200, 3, seed)

# # reg(50, 4, seed)
# # reg(60, 4, seed)
# # reg(70, 4, seed)
# # reg(80, 4, seed)
# # reg(90, 4, seed)
# reg(100, 4, seed)
# reg(120, 4, seed)
# reg(140, 4, seed)
# reg(160, 4, seed)
# # reg(180, 4, seed)
# # reg(200, 4, seed)

# # reg(50, 5, seed)
# # reg(60, 5, seed)
# # reg(70, 5, seed)
# # reg(80, 5, seed)
# # reg(90, 5, seed)
# reg(100, 5, seed)
# reg(120, 5, seed)
# reg(140, 5, seed)
# reg(160, 5, seed)
# # reg(180, 5, seed)
# # reg(200, 5, seed)


# # erdos(50, 3, seed)
# # erdos(60, 3, seed)
# # erdos(70, 3, seed)
# # erdos(80, 3, seed)
# # erdos(90, 3, seed)
# erdos(100, 6, seed)
# erdos(120, 6, seed)
# erdos(140, 6, seed)
# erdos(160, 6, seed)
# # erdos(180, 3, seed)
# # erdos(200, 3, seed)

# # erdos(50, 4, seed)
# # erdos(60, 4, seed)
# # erdos(70, 4, seed)
# # erdos(80, 4, seed)
# # erdos(90, 4, seed)
# erdos(100, 4, seed)
# erdos(120, 4, seed)
# erdos(140, 4, seed)
# erdos(160, 4, seed)
# # erdos(180, 4, seed)
# # erdos(200, 4, seed)

# # erdos(50, 5, seed)
# # erdos(60, 5, seed)
# # erdos(70, 5, seed)
# # erdos(80, 5, seed)
# # erdos(90, 5, seed)
# erdos(100, 5, seed)
# erdos(120, 5, seed)
# erdos(140, 5, seed)
# erdos(160, 5, seed)
# # erdos(180, 5, seed)
# # erdos(200, 5, seed)


# erdos(200, 15, seed)
reg(200, 15, seed)
