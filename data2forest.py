#!/usr/bin/python env
from itertools import combinations
import numpy as np
import networkx as nx
from vd import fn2valcs
from scorer import Scorer
from networkx.algorithms.tree.branchings import maximum_spanning_arborescence

def edge_weight(edge, scorer):
    (x,y) = edge
    if  x == y :
        return scorer.score(y,(), update=False)
    else:
        return scorer.score(y,(x,), update=False)

def gen_edges(n:int):
    for (x,y) in combinations(range(n), 2):
        yield (x,y)
        yield (y,x)
        if x == 0:
            yield (y,y)
            if y == 1:
                yield (x,x)

def create_graph(n, scorer):
    weighted_edges = [(*edge,edge_weight(edge, scorer)) 
                      for edge in gen_edges(n)]
    G = nx.DiGraph()
    G.add_weighted_edges_from(weighted_edges)
    return G

def main(args):
    data = np.loadtxt(args.data_filename, dtype=np.int8)
    valcounts = np.array(fn2valcs(args.vd_filename), dtype=int)
    scorer = Scorer(valcounts, data, score_fun=args.score)
    G = create_graph(len(valcounts), scorer)
    G_mist = maximum_spanning_arborescence(G)
    nx.write_edgelist(G_mist, args.out_filename, data=['weight'])

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