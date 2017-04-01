from data.load import load_genx, Entity

from collections import defaultdict
import tensorflow as tf
import numpy as np

#N_ATTRS = 5
N_HIDDEN = 256
N_COMM = 256
N_BATCH = 100
MAX_ITEMS = 20
INIT_SCALE = 1.47

random = np.random.RandomState(0)

def _linear(t_in, n_out):
    if len(t_in.get_shape()) == 2:
        op = "ij,jk->ik"
    elif len(t_in.get_shape()) == 3:
        op = "ijk,kl->ijl"
    else:
        assert False
    v_w = tf.get_variable(
            "w",
            shape=(t_in.get_shape()[-1], n_out),
            initializer=tf.uniform_unit_scaling_initializer(
                factor=INIT_SCALE))
    v_b = tf.get_variable(
            "b",
            shape=n_out,
            initializer=tf.constant_initializer(0))
    return tf.einsum(op, t_in, v_w) + v_b

def _embed(t_in, n_embeddings, n_out):
    v = tf.get_variable(
            "embed", shape=(n_embeddings, n_out),
            initializer=tf.uniform_unit_scaling_initializer())
    t_embed = tf.nn.embedding_lookup(v, t_in)
    return t_embed


def _mlp(t_in, widths, activations):
    assert len(widths) == len(activations)
    prev_width = t_in.get_shape()[1]
    prev_layer = t_in
    for i_layer, (width, act) in enumerate(zip(widths, activations)):
        with tf.variable_scope(str(i_layer)):
            layer = _linear(prev_layer, width)
            if act is not None:
                layer = act(layer)
        prev_layer = layer
        prev_width = width
    return prev_layer

def _flatten(lol):
    if not isinstance(lol, tuple):
        return lol
    out = []
    for l in lol:
        out.append(_flatten(l))
    return out

#def sample_lf():
#    choice = random.rand()
#    if choice < 0.7:
#        return random.randint(N_ATTRS)
#    elif choice < 0.85:
#        return ("and", sample_lf(), sample_lf())
#    else:
#        return ("or", sample_lf(), sample_lf())
#
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

def explain(env, label, lfs):
    valid = []
    for lf in lfs:
        ok = True
        for i in range(MAX_ITEMS):
            if not any(env[i]):
                continue
            ev = 1 if eval_lf(env[i], lf) else 0
            if ev != label[i]:
                ok = False
                break
        if ok:
            valid.append(lf)
    return min(valid, key=lambda x: 1 if isinstance(x, int) else len(_flatten(x)))

def sample_envs(lfs, dataset):
    envs = []
    labels = []
    attrs_by_type = defaultdict(list)
    for attr in dataset.attrs:
        typ = attr.split(":")[1]
        attrs_by_type[typ].append(attr)
    while len(envs) < N_BATCH:
        lf = lfs[random.randint(len(lfs))]
        env = []
        label = []
        accepted = False
        rejected = False
        used = []
        count = random.randint(MAX_ITEMS)
        for j in range(MAX_ITEMS):
            if j >= count:
                env.append(np.zeros(len(dataset.attrs)))
                label.append(0)
                continue
            attrs = np.zeros(len(dataset.attrs))
            for group in attrs_by_type.values():
                val = random.choice(group)
                attrs[dataset.attrs[val]] = 1
            env.append(attrs)
            if eval_lf(attrs, lf, dataset):
                accepted = True
                label.append(1)
            else:
                rejected = True
                label.append(0)
        if not (accepted and rejected):
            continue
        envs.append(env)
        labels.append(label)
    return envs, labels

def sample_annotated(dataset):
    envs = []
    labels = []
    while len(envs) < N_BATCH:
        scene_id = random.choice(dataset.labels.keys())
        scene = dataset.scenes[scene_id]
        ann = dataset.labels[scene_id]
        assert len(ann) > 0
        lf = ann[random.randint(len(ann))].lf
        env = np.zeros((MAX_ITEMS, len(dataset.attrs)))
        label = np.zeros(MAX_ITEMS)
        assert len(scene.targets) + len(scene.distractors) < MAX_ITEMS
        for j, ent in enumerate(scene.targets):
            for prop in ent.props:
                env[j, dataset.attrs[prop]] = 1
                label[j] = 1
        for j, ent in enumerate(scene.distractors):
            for prop in ent.props:
                env[j + len(scene.targets), dataset.attrs[prop]] = 1
                label[j + len(scene.targets)] = 0
        envs.append(env)
        labels.append(label)
    return envs, labels

def build_model(dataset):
    t_features = tf.placeholder(tf.float32, (None, MAX_ITEMS, len(dataset.attrs)))
    t_labels = tf.placeholder(tf.float32, (None, MAX_ITEMS))

    t_in = tf.concat((t_features, tf.expand_dims(t_labels, axis=2)), axis=2)

    cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
    with tf.variable_scope("layer1"):
        states1, hidden1 = tf.nn.dynamic_rnn(cell, t_in, dtype=tf.float32)
    #with tf.variable_scope("layer2"):
    #    states2, hidden2 = tf.nn.dynamic_rnn(cell, states1, dtype=tf.float32)
    #t_hidden = hidden2
    t_hidden = hidden1
    t_msg = tf.nn.relu(_linear(t_hidden, N_COMM))

    t_expand_msg = tf.expand_dims(t_msg, axis=1)
    t_tile_message = tf.tile(t_expand_msg, (1, MAX_ITEMS, 1))

    t_out_feats = tf.concat((t_tile_message, t_features), axis=2)
    t_pred = _mlp(t_out_feats, (N_HIDDEN, 1), (tf.nn.relu, None))
    t_pred = tf.squeeze(t_pred)
    t_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=t_labels, logits=t_pred))

    return t_features, t_labels, t_loss, t_pred

if __name__ == "__main__":
    dataset = load_genx()

    for i_scene, scene in dataset.scenes.items():
        if i_scene not in dataset.labels:
            continue
        for nl, lf in dataset.labels[i_scene]:
            if not all(eval_lf(ent, lf, dataset) for ent in scene.targets):
                print "warning: failed check with", lf
            if any(eval_lf(ent, lf, dataset) for ent in scene.distractors):
                print "warning: failed check with", lf

    t_features, t_labels, t_loss, t_pred = build_model(dataset)
    optimizer = tf.train.AdamOptimizer(0.001)
    o_train = optimizer.minimize(t_loss)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    lfs = list(enumerate_lfs(3, dataset))
    while True:
        loss = 0
        acc = 0
        ex = 0

        h_loss = 0
        h_acc = 0
        h_ex = 0
        for t in range(100):
            envs, labels = sample_envs(lfs, dataset)
            l, preds, _ = session.run(
                    [t_loss, t_pred, o_train],
                    {t_features: envs, t_labels: labels})
            match = (preds > 0) == labels
            a = np.mean(match)
            e = np.mean(np.all(match, axis=1))
            loss += l
            acc += a
            ex += e

            if t % 10 == 0:
                real_envs, real_labels = sample_annotated(dataset)
                h_l, preds = session.run(
                        [t_loss, t_pred],
                        {t_features: real_envs, t_labels: real_labels})
                match = (preds > 0) == real_labels
                h_a = np.mean(match)
                h_e = np.mean(np.all(match, axis=1))

                h_loss += h_l
                h_acc += h_a
                h_ex += h_e

        print loss / 100, acc / 100, ex / 100
        print h_loss / 10, h_acc / 10, h_ex / 10
        print
