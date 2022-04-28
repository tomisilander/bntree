from itertools import count
import numpy as np
from functools import lru_cache
from scipy.special import entr

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
                            'HIC': self.HIC
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
            print(family_rows.shape, counts.shape)
            for ix, c in zip(map(tuple, family_rows), self.row_counts):
                counts[ix] += c

        parent_freqs = counts.sum(axis=-1)                                           
        return counts, parent_freqs
        
    def log_ml(self, child, parents, **kwargs):
        child_freqs, parent_freqs = self.get_counts(child, parents)
        res = -np.sum(entr(child_freqs))  # NB! entr is -nlogn  
        res += np.sum(entr(parent_freqs)) # NB! entr is -nlogn
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
