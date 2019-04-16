import numpy as np
from multiprocessing import cpu_count, Pool


def balanced_parallel(fn, a, processes=None, timeout=None):
    """ apply fn on slice of a, return concatenated result """
    if processes is None:
        processes = cpu_count()
    print('Parallel:\tstarting {} processes on input with shape {}'.format(processes, a.shape))
    results = assigned_parallel(fn, np.array_split(a, processes), timeout=timeout, verbose=False)
    return np.concatenate(results, 0)


def assigned_parallel(fn, l, processes=None, timeout=None, verbose=True):
    """ apply fn on each element of l, return list of results """
    if processes is None:
        processes = min(cpu_count(), len(l))
    pool = Pool(processes=processes)
    if verbose:
        print('Parallel:\tstarting {} processes on {} elements'.format(processes, len(l)))

    # add jobs to the pool
    handler = [pool.apply_async(fn, args=x if isinstance(x, tuple) else (x, )) for x in l]

    # pool running, join all results
    results = [handler[i].get(timeout=timeout) for i in range(processes)]

    pool.close()
    return results
