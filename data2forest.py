#!/usr/bin/env python

from itertools import product, combinations
import numpy as np
import networkx as nx
from vd import fn2valcs
import scorer as scr
from networkx.algorithms.tree.branchings import maximum_spanning_arborescence
from multiprocessing import Pool
from common import save_bn

def edge_weight(edge, scorer):
    (x,y) = edge
    n = len(scorer.valcounts)
    parent = () if  x == n else (x,)
    return scorer.score(y, parent, update=False)

def gen_directed_edges(n:int):
    for (x,y) in product(range(n), repeat=2):
        if x==y:
            yield(n,x)
        else:
            yield (x,y)

def gen_undirected_edges(n:int):
    for (x,y) in combinations(range(n), r=2):
        yield(x,y)

global_scorer = None
def edge_weighter(edge):        
        return (*edge,edge_weight(edge, global_scorer))

def weight_edges(n, scorer, edge_generator, nof_cores=1):
    
    if nof_cores > 1:
        global global_scorer
        global_scorer = scorer
        chunk_len = int(np.ceil(n*n/nof_cores))
        with Pool(nof_cores) as p:
            weighted_edges = list(p.imap(edge_weighter, edge_generator(n), chunksize=chunk_len))
        global_scorer = None
    else:
        weighted_edges = [(*edge,edge_weight(edge, scorer)) 
                          for edge in edge_generator(n)]

    return weighted_edges

def direct_it(G):
    G = G.copy()
    DG = nx.DiGraph()
    while G.number_of_nodes() > 0:
        # print(G.edges)
        roots_left = set([set(G.nodes).pop()])
        # print('RL', roots_left)
        while roots_left:
            root = roots_left.pop()
            # print('picked root', root)
            for child in G.neighbors(root):
                DG.add_edge(root,child,weight = G[root][child]['weight'])
                roots_left.add(child)
            # print('ROOTS', roots_left)
            G.remove_node(root)
            # print(G.edges)
            # print('DG', DG.edges)
 
    return DG
 
def main(args):
    data = np.loadtxt(args.data_filename, dtype=np.int8)
    valcounts = np.array(fn2valcs(args.vd_filename), dtype=int)
    scorer = scr.Scorer(valcounts, data, score=args.score)

    if args.score == 'MI':
        weighted_edges = weight_edges(n, scorer, gen_undirected_edges, args.nof_cores)
        G = nx.Graph()
        G.add_weighted_edges_from(weighted_edges)
        G_mst = nx.maximum_spanning_tree(G, algorithm='prim')
        G_mst = direct_it(G_mst)
    else:
        n = len(valcounts)
        weighted_edges = weight_edges(n, scorer, gen_directed_edges, args.nof_cores)
        G = nx.DiGraph()
        G.add_weighted_edges_from(weighted_edges)
        G_mst = maximum_spanning_arborescence(G)
        G_mst.remove_node(n)

    save_bn(G_mst, args.bn_filename) 

    return scr.score_graph(G_mst, scorer)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    scr.add_args(parser)
    parser.add_argument('--nof_cores', type=int, default=1)
    args = parser.parse_args()
    score = main(args)
    print(score)