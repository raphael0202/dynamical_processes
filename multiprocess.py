import multiprocessing as mp


def apply_async_with_callback(func, args, additional_arg_list, arg_pos, nb_cpu):

    if args[arg_pos] is not None:
        raise ValueError("The argument at position arg_pos is not None.")

    pool = mp.Pool(processes=nb_cpu)
    pool_list = []

    for thread in xrange(nb_cpu):
        args[arg_pos] = additional_arg_list[thread]
        pool_list.append(pool.apply_async(func, args=tuple(args)))

    pool.close()
    results = [job.get() for job in pool_list]

    return results