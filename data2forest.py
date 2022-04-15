#!/usr/bin/env python

from ast import fix_missing_locations
from itertools import product
import numpy as np
import networkx as nx
from vd import fn2valcs
from scorer import Scorer
from networkx.algorithms.tree.branchings import maximum_spanning_arborescence
from multiprocessing import Pool


def edge_weight(edge, scorer):
    (x,y) = edge
    n = len(scorer.valcounts)
    if  x == n :
        return scorer.score(y,(), update=False)
    else:
        return scorer.score(y,(x,), update=False)

def gen_edges(n:int):
    for (x,y) in product(range(n), repeat=2):
        if x==y:
            yield(n,x)
        else:
            yield (x,y)


global_scorer = None
def edge_weighter(edge):
        return (*edge,edge_weight(edge, global_scorer))

def create_graph(n, scorer):
    nof_cores = 4

    if nof_cores > 1:
        global global_scorer
        global_scorer = scorer
        chunk_len = int(np.ceil(n*n/nof_cores))
        with Pool(nof_cores) as p:
            weighted_edges = list(p.imap(edge_weighter, gen_edges(n), chunksize=chunk_len))
        global_scorer = None
    else:
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

def load(filename):
    with open(filename) as inf:
        G = nx.DiGraph()
        G.add_nodes_from(range(int(inf.readline())))
        edges = [tuple(map(int, l.split())) for l in inf]        
        G.add_edges_from(edges)
        return G

def score_graph(G, scorer):
    for node in G.nodes():
        parents = list(G.predecessors(node))
        scorer.score(node, parents)
    return scorer.total_score()

def main(args):
    data = np.loadtxt(args.data_filename, dtype=np.int8)
    valcounts = np.array(fn2valcs(args.vd_filename), dtype=int)
    n = len(valcounts)
    scorer = Scorer(valcounts, data, score=args.score)
    G = create_graph(n, scorer)
    G_mist = maximum_spanning_arborescence(G)
    G_mist.remove_node(n)
    save(G_mist, args.out_filename)
    return score_graph(G_mist, scorer)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('vd_filename')
    parser.add_argument('data_filename')
    parser.add_argument('out_filename')
    scores = 'BIC AIC HIC'.split()
    parser.add_argument('-s', '--score', choices=scores, default='BIC')
    args = parser.parse_args()
    score = main(args)
    print(score)