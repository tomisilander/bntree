import numpy as np
from functools import lru_cache

class Scorer():

    def __init__(self, valcounts, data, score='BIC', **kwargs):
        self.valcounts = np.array(valcounts)
        self.data = np.array(data, dtype=int)
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

    def marginalize_counts(self, rows, row_counts):
        marg_rows, marg_row_start_ixs = np.unique(rows, axis=0, return_index=True)

        row_counts_cumsums = np.cumsum(row_counts)
        # now, selecting correct positions from cumsum
        # and the diffing them gives the counts of marg_rows
        last_poss = marg_row_start_ixs[1:] - 1
        last_cumsum = row_counts_cumsums[-1]
        marg_row_counts = np.diff(row_counts_cumsums[last_poss], prepend=0, append=last_cumsum)
        return marg_rows, marg_row_counts

    def get_counts(self, child, parents):
        parents = list(parents)
        family  = parents + [child]
        families, child_freqs  = np.unique(self.data[:, family], axis=0, return_counts=True)
        _, parent_freqs = self.marginalize_counts(families[:,:len(parents)], child_freqs)                                             
        return child_freqs, parent_freqs
        
    def log_ml(self, child, parents, **kwargs):
        child_freqs, parent_freqs = self.get_counts(child, parents)
        res = np.sum(child_freqs * np.log(child_freqs))  
        res -= np.sum(parent_freqs * np.log(parent_freqs))
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
        res = self.score_fn(child, parents, **self.kwargs)
        if update:
            self.score_table[child] = res
        return res

    def total_score(self, new_parents={}, update=False):
        if update:
            for child, parents in new_parents.items():
                self.score_table[child] = self.score_fn(
                    child, parents, **self.kwargs)
        return sum(self.score_table)
