import json
import tqdm
import torch
import numpy as np
import random
import torch.optim
import torch.utils.data.sampler
import os
import time

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.ove_polya_gamma_gp import OVEPolyaGammaGP, PredictiveOVEPolyaGammaGP
from methods.logistic_softmax_gp import LogisticSoftmaxGP, PredictiveLogisticSoftmaxGP
from methods.bayesian_maml import BayesianMAML, ChaserBayesianMAML
from methods.gpnet import GPNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_best_file, get_assigned_file

from methods.ove_polya_gamma_gp import kernel_ingredient

from calibrate import ECELoss

from sacred import Experiment
from sacred.observers import FileStorageObserver

EXPERIMENT_NAME = "test"


def get_save_dir():
    return os.path.join("runs", EXPERIMENT_NAME)


ex = Experiment(EXPERIMENT_NAME, ingredients=[kernel_ingredient])
ex.observers.append(FileStorageObserver(get_save_dir()))


@ex.capture
def get_checkpoint_dir(_run):
    return os.path.join(get_save_dir(), str(_run._id))


@ex.config
def get_config():
    # where runs are located
    run_dir = "runs/train"

    # job id to evaluate
    job_id = -1

    # saved feature from the model trained in x epoch, use the best model if x is -1
    save_iter = -1

    # number of episodes to test
    num_episodes = 600

    # relationnet_softmax replace L2 norm with softmax to expedite training,
    # maml_approx use first-order approximation in the gradient for efficiency
    # if default, match whatever setting was found in the job config
    # baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}
    method = "default"

    # baseline and baseline++ only use this parameter in finetuning
    # number of labeled data in each class, same as n_support
    n_shot = 5

    # baseline and baseline++ only use this parameter in finetuning
    # class num to classify for testing (validation)
    test_n_way = 5

    # default novel, but you can also test base/val class accuracy if you want
    # base/val/novel
    split = "novel"

    # further adaptation in test time or not
    adaptation = False

    # Repeat the test N times with different seeds and take the mean. The seeds range is [seed, seed+repeat]
    repeat = 5

    # number of draws for polya-gamma gps
    num_draws = None

    # number of steps for polya-gamma gps
    num_steps = None

    # Seed for Numpy and pyTorch. Default: 0 (None)
    seed = 0

    # tag (for logging purposes)
    tag = "default"

    # command allows specification of which evaluation to run
    command = "evaluate"

    # command = shot_sweep
    shot_sweep_min_shot = 1
    shot_sweep_max_shot = 20

    # None for no noise for default, otherwise 0-14
    noise_type = None

    # None for no noise, otherwise 1-5
    noise_strength = None

    run_prefix = ""


def _set_seed(seed, verbose=True):
    if seed != 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if verbose:
            print("[INFO] Setting SEED: " + str(seed))
    else:
        if verbose:
            print("[INFO] Setting SEED: None")


@ex.capture
def feature_evaluation(
    cl_data_file, model, test_n_way, n_shot, n_query=15, adaptation=False
):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list, test_n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append(
            [np.squeeze(img_feat[perm_ids[i]]) for i in range(n_shot + n_query)]
        )  # stack each batch

    z_all = torch.from_numpy(np.array(z_all))

    model.n_query = n_query
    if adaptation:
        scores = model.set_forward_adaptation(z_all, is_feature=True)
    else:
        scores = model.set_forward(z_all, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(test_n_way), n_query)
    acc = np.mean(pred == y) * 100
    return acc, {"logits": scores, "targets": y}


@ex.capture
def validate_config(job_id):
    # job id checks
    assert job_id != -1, "must specify which job id to evaluate"


@ex.capture
def get_model(test_n_way, n_shot, num_draws, num_steps):
    model = get_job_config()["model"]
    dataset = get_job_config()["dataset"]
    method = get_method()
    few_shot_params = dict(n_way=test_n_way, n_support=n_shot)

    if method == "baseline":
        return BaselineFinetune(model_dict[model], **few_shot_params)
    elif method == "baseline++":
        return BaselineFinetune(model_dict[model], loss_type="dist", **few_shot_params)
    elif method == "ove_polya_gamma_gp":
        return OVEPolyaGammaGP(model_dict[model], **few_shot_params)
    elif method == "predictive_ove_polya_gamma_gp":
        model = PredictiveOVEPolyaGammaGP(model_dict[model], **few_shot_params)
        return model
    elif method == "logistic_softmax_gp":
        return LogisticSoftmaxGP(model_dict[model], **few_shot_params)
    elif method == "predictive_logistic_softmax_gp":
        return PredictiveLogisticSoftmaxGP(model_dict[model], **few_shot_params)
    elif method == "bayesian_maml":
        return BayesianMAML(
            model_dict[model],
            num_draws=num_draws,
            num_steps=num_steps,
            **few_shot_params
        )
    elif method == "chaser_bayesian_maml":
        return ChaserBayesianMAML(
            model_dict[model],
            num_draws=num_draws,
            num_steps=num_steps,
            **few_shot_params
        )
    elif method == "gpnet":
        return GPNet(model_dict[model], **few_shot_params)
    elif method == "protonet":
        return ProtoNet(model_dict[model], **few_shot_params)
    elif method == "matchingnet":
        return MatchingNet(model_dict[model], **few_shot_params)
    elif method in ["relationnet", "relationnet_softmax"]:
        if model == "Conv4":
            feature_model = backbone.Conv4NP
        elif model == "Conv6":
            feature_model = backbone.Conv6NP
        elif model == "Conv4S":
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[model](flatten=False)
        loss_type = "mse" if method == "relationnet" else "softmax"
        return RelationNet(feature_model, loss_type=loss_type, **few_shot_params)
    elif method in ["maml", "maml_approx"]:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(
            model_dict[model], approx=(method == "maml_approx"), **few_shot_params
        )
        if dataset in [
            "omniglot",
            "cross_char",
        ]:  # maml use different parameter in omniglot
            model.n_task = 32
            model.task_update_num = 1
            model.train_lr = 0.1
        return model
    else:
        raise ValueError("unknown method {}".format(method))


@ex.capture
def get_job_dir(run_dir, job_id):
    return os.path.join(run_dir, str(job_id))


@ex.capture
def get_job_config(run_dir, job_id, run_prefix):
    with open(os.path.join(get_job_dir(), str(run_prefix), "config.json")) as f:
        return json.load(f)


@ex.capture
def get_checkpoint_file(save_iter):
    job_dir = get_job_dir()

    if save_iter != -1:
        return get_assigned_file(job_dir, save_iter)
    else:
        return get_best_file(job_dir)


@ex.capture
def load_model(n_shot, test_n_way, num_draws, num_steps, method):
    model = get_model(
        n_shot=n_shot, test_n_way=test_n_way, num_draws=num_draws, num_steps=num_steps
    )
    model = model.cuda()

    # for baseline/baseline++ just use feature evaluation
    if get_method() not in ["baseline", "baseline++"]:
        state_dict = torch.load(get_checkpoint_file())["state"]

        # model.num_steps = 1

        # # TODO: configure this better
        if method != "default":
            print("method is not default. Assuming transfer from baseline to gp...")
            state_dict["kernel.output_scale_raw"] = torch.Tensor([1.0]).log()

            for k in [
                "classifier.weight",
                "classifier.bias",
                "classifier.L.weight_g",
                "classifier.L.weight_v",
            ]:
                if k in state_dict:
                    del state_dict[k]

        model.load_state_dict(state_dict)

    model.eval()

    if num_draws is not None:
        model.num_draws = num_draws

    if num_steps is not None:
        model.num_steps = num_steps

    return model


@ex.capture
def get_method(method):
    if method == "default":
        return get_job_config()["method"]
    else:
        return method


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


@ex.capture
def get_feature_file(split):
    ret = os.path.join(get_job_dir(), "{}_features.hdf5".format(split))
    if os.path.isfile(ret):
        return ret
    else:
        return None


@ex.capture
def get_loader(
    iter_num, test_n_way, n_shot, method, noise_type, noise_strength, command
):
    print("loading with {:d} way and {:d} shot".format(test_n_way, n_shot))
    feature_file = get_feature_file()

    if feature_file is not None and noise_type is None and command != "ooe":
        return feat_loader.init_loader(feature_file)
    else:
        datamgr = SetDataManager(
            get_image_size(),
            n_eposide=iter_num,
            n_query=15,
            n_way=test_n_way,
            n_support=n_shot,
        )
        if noise_type is None:
            return datamgr.get_data_loader(get_split_file(), aug=False)
        else:
            return datamgr.get_noisy_data_loader(
                get_split_file(), noise_type, noise_strength
            )


def repeat_iterator(iterable):
    while True:
        for item in iterable:
            yield item


class EpochLoader:
    def __init__(self, iterable, num_episodes):
        self.iterable = repeat_iterator(iterable)
        self.num_episodes = num_episodes

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        for _ in range(self.num_episodes):
            yield self.convert_to_episode(next(self.iterable))

    def canonicalize(self, inputs, targets):
        assert inputs.size(0) == 1
        assert targets.size(0) == 1
        inputs = inputs[0]
        targets = targets[0]

        class_counts = torch.bincount(targets)
        assert torch.all(
            class_counts.eq(class_counts[0])
        ), "classes not balanced, cannot convert"

        shot = class_counts[0].item()
        way = class_counts.size(0)

        assert (
            targets.size(0) == shot * way
        ), "number of examples does not match shot * way"

        # reshape to class batched format
        inputs = inputs.reshape(way, shot, *inputs.size()[1:])
        targets = targets.reshape(way, shot)

        way_permutation = targets[:, 0].argsort()

        inputs = inputs[way_permutation]
        targets = targets[way_permutation]

        assert torch.all(
            targets.eq(torch.arange(way).unsqueeze(-1))
        ), "problem with class permutation"

        return inputs, targets

    def convert_to_episode(self, sample):
        train_inputs, train_targets = self.canonicalize(*sample["train"])
        test_inputs, test_targets = self.canonicalize(*sample["test"])

        return (
            torch.cat([train_inputs, test_inputs], 1),
            torch.cat([train_targets, test_targets], 1),
        )


def load_feature_extractor():
    method = get_job_config()["method"]
    model = get_job_config()["model"]

    if method in ["relationnet", "relationnet_softmax"]:
        if model == "Conv4":
            extractor = backbone.Conv4NP()
        elif model == "Conv6":
            extractor = backbone.Conv6NP()
        elif model == "Conv4S":
            extractor = backbone.Conv4SNP()
        else:
            extractor = model_dict[model](flatten=False)
    elif method in ["maml", "maml_approx"]:
        raise ValueError("MAML do not support save feature")
    else:
        extractor = model_dict[model]()

    extractor = extractor.cuda()

    state = torch.load(get_checkpoint_file())["state"]
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace(
                "feature.", ""
            )  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

    extractor.load_state_dict(state)
    extractor.eval()

    return extractor


@ex.capture
def single_test(model, n_shot, test_n_way, split, adaptation, num_episodes):
    loader = get_loader(num_episodes, n_shot=n_shot, test_n_way=test_n_way)

    if adaptation:
        # We perform adaptation on MAML simply by updating more times.
        model.task_update_num = 100

    if isinstance(loader, dict):
        acc_all = []
        stats_all = []
        pbar = tqdm.tqdm(range(num_episodes))
        for _ in pbar:
            acc, stats = feature_evaluation(
                loader,
                model,
                n_shot=n_shot,
                test_n_way=test_n_way,
                adaptation=adaptation,
            )
            acc_all.append(acc)
            stats_all.append(stats)
            pbar.set_description("Acc {:f}".format(np.mean(acc_all)))
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

        stats_final = {}
        for k in stats_all[0].keys():
            stats_final[k] = (
                torch.cat([torch.as_tensor(stats[k]) for stats in stats_all], 0)
                .detach()
                .cpu()
            )

        return {"acc_mean": acc_mean, "acc_std": acc_std, "stats": stats_final}
    else:
        if get_method() in ["baseline", "baseline++"]:
            feature_extractor = load_feature_extractor()
            return model.test_loop(
                loader,
                use_progress=True,
                return_stats=True,
                feature_extractor=feature_extractor,
            )
        else:
            return model.test_loop(loader, use_progress=True, return_stats=True)


@ex.capture
def ooe_evaluation(model, n_shot, test_n_way, split, adaptation, num_episodes):
    loader = get_loader(num_episodes, n_shot=n_shot, test_n_way=2 * test_n_way)

    if adaptation:
        # We perform adaptation on MAML simply by updating more times.
        model.task_update_num = 100

    if get_method() in ["baseline", "baseline++"]:
        feature_extractor = load_feature_extractor()
    else:
        feature_extractor = None

    targets_all = []
    logits_all = []

    pbar = tqdm.tqdm(loader)
    for x, _ in pbar:
        # 2C x N x ...
        x_support = x[:test_n_way, :n_shot]
        x_query = x[:test_n_way, n_shot:]
        x_distractor = x[test_n_way:, n_shot:]

        x = torch.cat([x_support, x_query, x_distractor], 1)
        model.n_query = x.size(1) - n_shot

        if feature_extractor is not None:
            x_flat = x.view(-1, *x.size()[2:])
            x_flat = feature_extractor(x_flat.cuda())
            x = x_flat.view(*x.size()[:2], -1)

        if isinstance(model, GPNet):
            _, _, _, scores = model.correct(x)
            logits_all.append(scores)
        else:
            scores = model.set_forward(x)
            logits_all.append(scores.cpu().detach().numpy())

        y_query = np.repeat(range(model.n_way), model.n_query)
        y_query = y_query.reshape(model.n_way, -1)
        y_query[:, y_query.shape[1] // 2 :] = -1
        y_query = y_query.reshape(-1)

        targets_all.append(y_query)

    return {
        "stats": {
            "logits": torch.as_tensor(np.concatenate(logits_all, 0)),
            "targets": torch.as_tensor(np.concatenate(targets_all, 0)),
        }
    }


@ex.capture
def shot_sweep(num_episodes, shot_sweep_min_shot, shot_sweep_max_shot, num_draws, _run):
    for shot in range(shot_sweep_min_shot, shot_sweep_max_shot + 1):
        target_file = os.path.join(
            get_checkpoint_dir(), "results_shot-{:02d}.pth".format(shot)
        )
        if os.path.isfile(target_file):
            continue

        model = load_model(n_shot=shot)
        if num_draws is not None:
            model.num_draws = num_draws

        results = single_test(model, n_shot=shot, num_episodes=num_episodes)

        _run.log_scalar("shot", shot)
        _run.log_scalar("acc_mean", results["acc_mean"])
        _run.log_scalar("acc_std", results["acc_std"])
        print(
            "{:d} shot: {:4.2f} +/- {:4.2f}".format(
                shot, results["acc_mean"], results["acc_std"]
            )
        )

        torch.save(results, target_file)


@ex.capture
def way_sweep(num_episodes, shot_sweep_min_shot, shot_sweep_max_shot, num_draws, _run):
    for way in range(max(2, shot_sweep_min_shot), shot_sweep_max_shot + 1):
        target_file = os.path.join(
            get_checkpoint_dir(), "results_way-{:02d}.pth".format(way)
        )
        if os.path.isfile(target_file):
            continue

        model = load_model(test_n_way=way)
        if num_draws is not None:
            model.num_draws = num_draws

        results = single_test(model, test_n_way=way, num_episodes=num_episodes)

        _run.log_scalar("way", way)
        _run.log_scalar("acc_mean", results["acc_mean"])
        _run.log_scalar("acc_std", results["acc_std"])

        print(
            "{:d} way: {:4.2f} +/- {:4.2f}".format(
                way, results["acc_mean"], results["acc_std"]
            )
        )

        torch.save(results, target_file)


@ex.capture
def scale_sweep(num_episodes, num_draws):
    print("running scale_sweep")

    results_all = []

    max_bias = 1.5
    num_points = 11

    for exp in torch.linspace(-max_bias, max_bias, num_points + 1):
        model = load_model()
        if num_draws is not None:
            model.num_draws = num_draws
        model.kernel.output_scale_raw.data.fill_(
            model.kernel.output_scale_raw.item() + exp
        )
        print("scale = ", model.kernel.output_scale_raw[:].exp())

        results = single_test(model)
        print("{:0.2f} scale: {:4.2f}".format(exp, results["acc_mean"]))

        results_all.append((exp, results))

    return results_all


@ex.capture
def noise_sweep(num_episodes, num_draws):
    print("running noise_sweep")

    for noise in [0.0, 1e-2, 1e-1, 1e0, 1e1]:
        model = load_model()
        if num_draws is not None:
            model.num_draws = num_draws
        model.noise = noise
        print("noise = ", model.noise)
        loader = get_loader(iter_num=num_episodes)
        acc_mean = model.test_loop(loader, use_progress=True)
        print("{:f} noise: {:4.2f}".format(noise, acc_mean))


@ex.automain
def main(command, seed, repeat, _run):
    print("using config: ", _run.config)
    print("save_dir: ", get_save_dir())

    validate_config()

    if command == "evaluate":
        accuracy_list = []
        results_all = []

        # repeat the test N times changing the seed in range [seed, seed+repeat]
        for i in range(seed, seed + repeat):
            if seed != 0:
                _set_seed(i)
            else:
                _set_seed(0)

            model = load_model()
            results = single_test(model)
            results_all.append(results)
            accuracy_list.append(results["acc_mean"])
            _run.log_scalar("acc", results["acc_mean"])
        print("-----------------------------")
        print(
            "Seeds = %d | Overall Test Acc = %4.2f%% +- %4.2f%%"
            % (repeat, np.mean(accuracy_list), np.std(accuracy_list))
        )
        print("-----------------------------")
        torch.save(results_all, os.path.join(get_checkpoint_dir(), "results.pth"))

        logits = torch.cat(
            [result["stats"]["logits"] for result in results_all], 0
        ).cuda()
        targets = torch.cat(
            [result["stats"]["targets"] for result in results_all], 0
        ).cuda()

        ece_module = ECELoss().cuda()
        ece_val = ece_module.forward(logits, targets)
        print("ece: ", ece_val)
        _run.log_scalar("ece", ece_val.item())
    elif command == "shot_sweep":
        _set_seed(seed)
        shot_sweep()
    elif command == "way_sweep":
        _set_seed(seed)
        way_sweep()
    elif command == "scale_sweep":
        _set_seed(seed)
        results_all = scale_sweep()
        torch.save(results_all, os.path.join(get_checkpoint_dir(), "results.pth"))
    elif command == "noise_sweep":
        _set_seed(seed)
        noise_sweep()
    elif command == "ooe":
        results_all = []
        # repeat the test N times changing the seed in range [seed, seed+repeat]
        for i in range(seed, seed + repeat):
            if seed != 0:
                _set_seed(i)
            else:
                _set_seed(0)

            model = load_model()
            results = ooe_evaluation(model)
            results_all.append(results)
        torch.save(results_all, os.path.join(get_checkpoint_dir(), "results.pth"))
    else:
        raise ValueError("unknown command {}".format(command))
