import re

def make_graph(V, E):
    G = {}
    for i in V:
        G[i] = {}
        
    for (u, v) in E:
        if not ( G.get(u) and G.get(v) ):
            raise "Vertex isn't present in the list of graph vertices %" % (u if not G.get(u) else v)
        else:
            G[u][v] = 1
            G[v][u] = 1
    return G

def input_graph():
    print "Enter a list of vertices on one line"
    V = raw_input().split()
    E = []
    while True:
        try:
            pair = raw_input().split()
            E.append(pair)
        except EOFError:
            print "Inputs Accepted :)"
            break
    return make_graph(V, E)

def graph_specs(G):
    edges = 0
    for v in G:
        edges += len(G[v])
    edges /= 2
    return len(G), edges
   
def isomorphic(G, H):
    pass

