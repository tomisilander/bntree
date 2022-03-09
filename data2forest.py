#!/usr/bin/python env
from ast import fix_missing_locations
from itertools import combinations
import numpy as np
import networkx as nx
from vd import fn2valcs
from scorer import Scorer
from networkx.algorithms.tree.branchings import maximum_spanning_arborescence

def edge_weight(edge, scorer):
    (x,y) = edge
    n = len(scorer.valcounts)
    if  x == n :
        return scorer.score(y,(), update=False)
    else:
        return scorer.score(y,(x,), update=False)

def gen_edges(n:int):
    for x in range(n):
        yield(n,x)
    for (x,y) in combinations(range(n), 2):
        yield (x,y)
        yield (y,x)

def create_graph(n, scorer):
    weighted_edges = [(*edge,edge_weight(edge, scorer)) 
                      for edge in gen_edges(n)]
    G = nx.DiGraph()
    G.add_weighted_edges_from(weighted_edges)
    return G

def save(G, filename):
    with open(filename, 'w') as outf:
        print(G.number_of_nodes(), file=outf)
        for (x,y) in G.edges():
            print(x, y, file=outf)

def main(args):
    data = np.loadtxt(args.data_filename, dtype=np.int8)
    valcounts = np.array(fn2valcs(args.vd_filename), dtype=int)
    n = len(valcounts)
    scorer = Scorer(valcounts, data, score=args.score)
    G = create_graph(n, scorer)
    G_mist = maximum_spanning_arborescence(G)
    G_mist.remove_node(n)
    save(G_mist, args.out_filename)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('vd_filename')
    parser.add_argument('data_filename')
    parser.add_argument('out_filename')
    scores = 'BIC AIC HIC'.split()
    parser.add_argument('-s', '--score', choices=scores, default='BIC')
    args = parser.parse_args()
    main(args)