import json
import torch
from torch.autograd import Variable
import os
import h5py
import tqdm

import configs
import backbone
from data.datamgr import SimpleDataManager
from io_utils import model_dict, get_resume_file, get_best_file, get_assigned_file

from sacred import Experiment
from sacred.observers import FileStorageObserver

EXPERIMENT_NAME = "save_features"
ex = Experiment(EXPERIMENT_NAME, ingredients=[])
ex.observers.append(FileStorageObserver(os.path.join("runs", EXPERIMENT_NAME)))


def save_features(model, data_loader, outfile):
    f = h5py.File(outfile, "w")
    max_count = len(data_loader) * data_loader.batch_size
    all_labels = f.create_dataset("all_labels", (max_count,), dtype="i")
    all_feats = None
    count = 0
    for (x, y) in tqdm.tqdm(data_loader):
        x = x.cuda()
        x_var = Variable(x)
        feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset(
                "all_feats", [max_count] + list(feats.size()[1:]), dtype="f"
            )
        all_feats[count : count + feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count : count + feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset("count", (1,), dtype="i")
    count_var[0] = count

    f.close()


#    Namespace(dataset='CUB', method='baseline', model='Conv4', n_shot=5, save_iter=-1, seed=0, split='novel', test_n_way=5, train_aug=False, train_n_way=5)


@ex.config
def get_config():
    # where runs are located
    run_dir = "runs/train"

    # job id to evaluate
    job_id = -1

    # saved feature from the model trained in x epoch, use the best model if x is -1
    save_iter = -1

    # default novel, but you can also test base/val class accuracy if you want
    # base/val/novel
    split = "novel"


@ex.capture
def validate_config(job_id):
    # job id checks
    assert job_id != -1, "must specify which job id to evaluate"


@ex.capture
def get_job_dir(run_dir, job_id):
    return os.path.join(run_dir, str(job_id))


@ex.capture
def get_job_config(run_dir, job_id):
    with open(os.path.join(get_job_dir(), "config.json")) as f:
        return json.load(f)


def get_method():
    return get_job_config()["method"]


@ex.capture
def get_checkpoint_file(save_iter):
    job_dir = get_job_dir()

    if save_iter != -1:
        return get_assigned_file(job_dir, save_iter)
    elif get_method() in ["baseline", "baseline++"]:
        return get_resume_file(job_dir)
    else:
        return get_best_file(job_dir)


@ex.capture
def get_image_size():
    model = get_job_config()["model"]
    dataset = get_job_config()["dataset"]

    if "Conv" in model:
        if dataset in ["omniglot", "cross_char"]:
            return 28
        else:
            return 84
    else:

        return 224


@ex.capture
def get_split_file(split):
    dataset = get_job_config()["dataset"]
    if dataset == "cross":
        if split == "base":
            return configs.data_dir["miniImagenet"] + "all.json"
        else:
            return configs.data_dir["CUB"] + split + ".json"
    elif dataset == "cross_char":
        if split == "base":
            return configs.data_dir["omniglot"] + "noLatin.json"
        else:
            return configs.data_dir["emnist"] + split + ".json"
    else:
        return configs.data_dir[dataset] + split + ".json"


def get_loader():
    datamgr = SimpleDataManager(get_image_size(), batch_size=64)
    return datamgr.get_data_loader(get_split_file(), aug=False)


def get_model():
    method = get_job_config()["method"]
    model = get_job_config()["model"]

    if method in ["relationnet", "relationnet_softmax"]:
        if model == "Conv4":
            return backbone.Conv4NP()
        elif model == "Conv6":
            return backbone.Conv6NP()
        elif model == "Conv4S":
            return backbone.Conv4SNP()
        else:
            return model_dict[model](flatten=False)
    elif method in ["maml", "maml_approx"]:
        raise ValueError("MAML do not support save feature")
    else:
        return model_dict[model]()


@ex.automain
def main(split, _run):
    print("using config: ", _run.config)

    validate_config()

    checkpoint_file = get_checkpoint_file()
    loader = get_loader()

    model = get_model()
    model = model.cuda()

    state = torch.load(checkpoint_file)["state"]
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace(
                "feature.", ""
            )  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

    model.load_state_dict(state)
    model.eval()

    out_file = os.path.join(get_job_dir(), "{}_features.hdf5".format(split))
    save_features(model, loader, out_file)
