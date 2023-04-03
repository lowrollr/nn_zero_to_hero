
def symmetric_derivative(f, args, argno, h=1e-7):
    orginal_value = args[argno]
    args[argno] += h
    l1 = f(*args)
    args[argno] -= (2 * h)
    l2 = f(*args)
    args[argno] = orginal_value
    return (l1 - l2) / (2 * h)


def all_derivatives(f, args, h=1e-7):
    return [symmetric_derivative(f, args, i, h) for i in range(len(args))]
        