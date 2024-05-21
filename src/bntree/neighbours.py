#!/usr/bin/env python

import networkx as nx
from itertools import combinations
from multiprocessing import Pool
import numpy as np

def gen_subsets(nbrs):
    ns = list(nbrs)
    for size in range(len(ns)):
        yield from combinations(ns, size)
    yield tuple(ns)

def gen_neighbor_subsets(G):
    uG = G.to_undirected()
    for n in uG.nodes:
        nbrs = list(uG.neighbors(n))
        yield gen_subsets(nbrs)


global_scorer = None
def score_family(family):
    return global_scorer.score(*family, update=False)

def score_families(scorer, families):
    global global_scorer
    global_scorer = scorer

    nof_cores = 1

    if nof_cores > 1:
        chunk_len = int(np.ceil(len(families)/nof_cores))
        with Pool(nof_cores) as p:
            res = list(p.imap(score_family, families, chunksize=chunk_len))
    else:
        res = list(map(score_family, families))

    global_scorer = None
    return res

def gen_scored_families(G, scorer):
    parent_sets = gen_neighbor_subsets(G)
    families = [[(n, ps) for ps in pss]
                for n, pss in enumerate(parent_sets)]
    scores = [score_families(scorer, fs) for fs in families]

    print(len(scores))
    for (n, (fs, ss)) in enumerate(zip(families,scores)):
        print(n, len(fs))
        for s, f in zip(ss,fs):
            print(s, len(f[1]), ' '.join(map(str, f[1])))

from data2forest import load, fn2valcs
from scorer import Scorer

def main(args):
    data = np.loadtxt(args.data_filename, dtype=np.int8)
    valcounts = np.array(fn2valcs(args.vd_filename), dtype=int)
    n = len(valcounts)
    scorer = Scorer(valcounts, data, score=args.score)
    G = load(args.bn_filename)
    gen_scored_families(G, scorer)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('vd_filename')
    parser.add_argument('data_filename')
    parser.add_argument('bn_filename')
    scores = 'BIC AIC HIC'.split()
    parser.add_argument('-s', '--score', choices=scores, default='BIC')
    args = parser.parse_args()

    main(args)