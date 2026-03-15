def adain(x, alpha, beta):
    mean = x.mean([2, 3], keepdim=True)
    std = x.std([2, 3], keepdim=True) + 1e-8
    return alpha * ((x - mean) / std) + beta