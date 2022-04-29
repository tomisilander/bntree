#!/usr/bin/env python

import numpy as np
from functools import lru_cache
from scipy.special import entr as _nlogn
from scipy.stats import entropy
from vd import fn2valcs
from common import load_bn
from cycounts import counts1d, counts2d

class Scorer():

    def __init__(self, valcounts, data, score='BIC', **kwargs):
        self.valcounts = np.array(valcounts)
        self.data = np.array(data, dtype=np.dtype("i"))
        self.rows, self.row_counts = np.unique(self.data, axis=0, return_counts=True)
        self.kwargs = kwargs
        self.score_table = [-np.inf]*len(self.valcounts)

        # some helpers

        N = data.shape[0]
        logN = np.log(N)
        loglogN = np.log(logN)

        self.xic_penalty = {'B': lambda k: 0.5*k*logN,
                            'A': lambda k: k,
                            'H': lambda k: k*loglogN}
        
        self.score_fns = {  'BIC': self.BIC,
                            'AIC': self.AIC,
                            'HIC': self.HIC,
                            'MI' : self.MI
                        }

        self.score_fn = self.score_fns[score]

    def get_counts(self, child, parents):
        family  = list(parents) + [child]
        valcs = [self.valcounts[v] for v in family]
        family_rows = self.rows[:,family]
        counts = np.zeros(valcs, dtype=int)
        if len(valcs) == 2:
            counts2d(family_rows, self.row_counts, counts)
        elif len(valcs) == 1:
            counts1d(family_rows, self.row_counts, counts)
        else:
            for ix, c in zip(map(tuple, family_rows), self.row_counts):
                counts[ix] += c
        return counts
        
    def log_ml(self, child, parents, **kwargs):
        child_freqs = self.get_counts(child, parents)
        parent_freqs = child_freqs.sum(axis=-1)                                           
        res = -np.sum(_nlogn(child_freqs))   
        res += np.sum(_nlogn(parent_freqs))
        return res

    def mutual_information(self, child, parents, **kwargs):
        child_freqs = self.get_counts(child, parents)
        marg_freqs1 = child_freqs.sum(axis=-1) # could be stored                                           
        marg_freqs2 = child_freqs.sum(axis=0)  # could be stored
        ind_freqs   = np.outer(marg_freqs1, marg_freqs2)
        res = entropy(child_freqs.flatten(), qk=ind_freqs.flatten())
        return res

    @lru_cache(maxsize=32000)
    def XIC(self, X, child, parents):
        nof_pcfgs = self.valcounts[list(parents)].prod(initial=1.0)
        nof_params = nof_pcfgs * (self.valcounts[child]-1)
        return self.log_ml(child, parents) - self.xic_penalty[X](nof_params)

    def BIC(self, child, parents, **kwargs):
        return self.XIC('B', child, parents)

    def AIC(self, child, parents):
        return self.XIC('A', child, parents, **self.kwargs)

    def HIC(self, child, parents):
        return self.XIC('H', child, parents, **self.kwargs)

    def MI(self, child, parents):
        nof_parents = len(parents)
        if nof_parents == 1:
            return self.mutual_information(child, parents, **self.kwargs)
        elif nof_parents == 0: 
            return 0.0
        else:
            raise "MI Not implemented"

    def score(self, child, parents, update=True):
        # return np.random.rand()
        res = self.score_fn(child, tuple(parents), **self.kwargs)
        if update:
            self.score_table[child] = res
        return res

    def total_score(self, new_parents={}, update=False):
        if update:
            for child, parents in new_parents.items():
                self.score_table[child] = self.score_fn(
                    child, parents, **self.kwargs)
        return sum(self.score_table)

def score_graph(G, scorer):
    for node in G.nodes():
        parents = list(G.predecessors(node))
        scorer.score(node, parents)
    return scorer.total_score()

def score_bn(args):
    data = np.loadtxt(args.data_filename, dtype=np.int8)
    valcounts = np.array(fn2valcs(args.vd_filename), dtype=int)
    scorer = Scorer(valcounts, data, score=args.score)
    G = load_bn(args.bn_filename)
    return score_graph(G, scorer)

def add_args(parser):
    parser.add_argument('vd_filename')
    parser.add_argument('data_filename')
    parser.add_argument('bn_filename')
    scores = 'BIC AIC HIC MI'.split()
    parser.add_argument('-s', '--score', choices=scores, default='BIC')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    score = score_bn(args)
    print(score)