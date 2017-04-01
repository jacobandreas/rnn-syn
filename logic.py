from data.load import Entity
import numpy as np

def flatten(lol):
    if not isinstance(lol, tuple):
        return lol
    out = []
    for l in lol:
        out.append(_flatten(l))
    return out

def enumerate_lfs(max_depth, dataset):
    if max_depth == 0:
        return
    for attr in dataset.attrs:
        yield attr
    for out in enumerate_lfs(max_depth-1, dataset):
        yield ("not", out)
    for op in ("and", "or"):
        for out1 in enumerate_lfs(max_depth-1, dataset):
            for out2 in enumerate_lfs(max_depth-1, dataset):
                if out1 != out2:
                    yield (op, out1, out2)

def eval_lf(thing, lf, dataset):
    assert isinstance(lf, str) or isinstance(lf, list) or isinstance(lf, tuple)
    if isinstance(lf, str) and isinstance(thing, Entity):
        return lf in thing.props
    elif isinstance(lf, str) and isinstance(thing, np.ndarray):
        return thing[dataset.attrs[lf]] == 1
    elif lf[0] == "not":
        return not eval_lf(thing, lf[1], dataset)
    elif lf[0] == "and":
        return all(eval_lf(thing, l, dataset) for l in lf[1:])
    elif lf[0] == "or":
        return any(eval_lf(thing, l, dataset) for l in lf[1:])

def explain_env(env, label, lfs, dataset):
    valid = []
    for lf in lfs:
        ok = True
        for i in range(env.shape[0]):
            if not any(env[i]):
                continue
            ev = 1 if eval_lf(env[i], lf, dataset) else 0
            if ev != label[i]:
                ok = False
                break
        if ok:
            valid.append(lf)
    return min(valid, key=lambda x: 1 if isinstance(x, int) else len(_flatten(x)))
