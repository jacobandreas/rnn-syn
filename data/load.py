#!/usr/bin/env python2

from collections import defaultdict, namedtuple
import sexpdata

Label = namedtuple("Label", ["nl", "lf"])
Entity = namedtuple("Entity", ["props"])
Scene = namedtuple("Scene", ["targets", "distractors"])
Dataset = namedtuple("Dataset", ["scenes", "labels", "attrs"])

def simplify(lf):
    if isinstance(lf, list):
        head = lf[0].value()
        if "The:" in head or "Every:" in head or "A:" in head:
            assert len(lf) == 2
            return simplify(lf[1])
        if "lambda" in head:
            assert len(lf) == 3
            return simplify(lf[2])
        if "shape:" in head:
            assert len(lf) == 3
            return lf[1].value()
        if "type:" in head:
            assert len(lf) == 3
            return lf[1].value()
        if "color:" in head:
            assert len(lf) == 3
            return lf[1].value()
        if "misc:" in head:
            return None
        if "equal:" in head:
            assert len(lf) == 3
            return simplify(lf[1])
        if "plu:" in head or "sg:" in head or "cardinality:" in head:
            return None
        simp = [simplify(l) for l in lf]
        simp = [s for s in simp if s is not None]
        if simp[0] in ("and", "or") and len(simp) == 1:
            return None
        if simp[0] == "and" and len(simp) == 2:
            return simp[1]
        if simp[0] == "gminus" and len(simp) == 2:
            return ["not", simp[1]]
        if simp[0] == "gminus" and len(simp) == 3:
            simp = ["and", simp[2], ["not", simp[1]]]
        if simp[0] == "not" and len(simp) == 1:
            return None
        return simp

    val = lf.value()
    if "or:" in val:
        return "or"
    if "and:" in val:
        return "and"
    if "not:" in val:
        return "not"
    if "gminus:" in val:
        return "gminus"
    if "gplus:" in val:
        return "or"
    if isinstance(lf, sexpdata.Symbol):
        return None

    return lf


def load_genx():
    labels = defaultdict(list)
    counter = 0
    with open("data/genx/labelling/all/LABELED_TRAINING.txt") as label_f:
        lines = label_f.readlines()
        for i in range(0, len(lines), 4):
            sent, lf_str, ex_id, _ = lines[i:i+4]
            sent = sent.strip().split()
            ex_id = int(ex_id)
            lf = sexpdata.loads(lf_str)
            lf = simplify(lf)
            if lf is None:
                print "warning: unable to parse", lf_str
                continue
            labels[ex_id].append(Label(sent, lf))
            counter += 1

    all_attrs = {}
    entities = {}
    with open("data/genx/state/Attributes.tsv") as ent_f:
        for line in ent_f:
            ent_id, attrs = line.strip().split("\t")
            attrs = set(attrs.split(","))
            for attr in attrs:
                if attr not in all_attrs:
                    all_attrs[attr] = len(all_attrs)
            entities[ent_id] = Entity(attrs)

    scenes = {}
    with open("data/genx/state/SceneIndex.txt") as scene_f:
        for line in scene_f:
            ex_id, target_ids, distractor_ids = line.strip().split("::")
            ex_id = int(ex_id)
            target_ids = target_ids.split()
            distractor_ids = distractor_ids.split()
            targets = [entities[i] for i in target_ids]
            distractors = [entities[i] for i in distractor_ids]
            scenes[ex_id] = Scene(targets, distractors)

    return Dataset(scenes, dict(labels), all_attrs)
