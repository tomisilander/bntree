
from cython.view cimport array as cvarray

cpdef void counts1d(const int[:,:] rows, 
                    const long[:]  row_counts,
                          long[:]  counts_out) nogil:
    cdef size_t i, nof_rows
    nof_rows = rows.shape[0]
    for i in range(nof_rows):
        counts_out[rows[i,0]] += row_counts[i] 

cpdef void counts2d(const int[:,:] rows, 
                    const long[:]   row_counts,
                          long[:,:] counts_out) nogil:
    cdef size_t i, nof_rows
    nof_rows = rows.shape[0]
    for i in range(nof_rows):
        counts_out[rows[i,0], rows[i,1]] += row_counts[i] 