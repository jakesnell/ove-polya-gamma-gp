import numpy as np
import torch
import random
import torch.optim
import os

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.ove_polya_gamma_gp import OVEPolyaGammaGP, PredictiveOVEPolyaGammaGP
from methods.logistic_softmax_gp import LogisticSoftmaxGP, PredictiveLogisticSoftmaxGP
from methods.bayesian_maml import BayesianMAML, ChaserBayesianMAML
from methods.gpnet import GPNet
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict

from methods.ove_polya_gamma_gp import kernel_ingredient

from tensorboardX import SummaryWriter

from sacred import Experiment
from sacred.observers import FileStorageObserver


EXPERIMENT_NAME = "train"
RUN_DIR = "runs"


def get_save_dir():
    return os.path.join("runs", EXPERIMENT_NAME)

ex = Experiment(EXPERIMENT_NAME, ingredients=[kernel_ingredient])
ex.observers.append(FileStorageObserver(get_save_dir()))


@ex.config
def get_config():
    # Seed for Numpy and pyTorch. Default: 0 (None)
    seed = 0

    # CUB/miniImagenet/cross/omniglot/cross_char
    dataset = "CUB"

    # model: Conv{4|6} / ResNet{10|18|34|50|101}
    model = "Conv4"

    # relationnet_softmax replace L2 norm with softmax to expedite training,
    # maml_approx use first-order approximation in the gradient for efficiency
    # ove_polya_gamma_gp/predictive_ove_polya_gamma_gp/baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}
    method = "baseline"

    # baseline and baseline++ would ignore this parameter
    # class num to classify for training
    train_n_way = 5

    # baseline and baseline++ only use this parameter in finetuning
    # class num to classify for testing (validation)
    test_n_way = 5

    # baseline and baseline++ only use this parameter in finetuning
    # number of labeled data in each class, same as n_support
    n_shot = 5

    # still required for save_features.py and test.py to find the model path correctly
    # perform data augmentation or not during training
    train_aug = False

    # make it larger than the maximum label value in base class
    # total number of classes in softmax, only used in baseline
    num_classes = 200

    # Save frequency
    save_freq = 10

    # Starting epoch
    start_epoch = 0

    # for meta-learning methods, each epoch contains 100 episodes.
    # The default epoch number is dataset dependent. See train.py
    # Stopping epoch
    stop_epoch = -1

    # optimizer to use
    optimization = "Adam"

    # num_draws for ove_polya_gamma_gp
    num_draws = None

    # num_steps for ove_polya_gamma_gp
    num_steps = None

    sigma = None

    # tag (for logging purposes)
    tag = "default"


@ex.capture
def get_base_file(dataset):
    if dataset == "cross":
        return configs.data_dir["miniImagenet"] + "all.json"
    elif dataset == "cross_char":
        return configs.data_dir["omniglot"] + "noLatin.json"
    else:
        return configs.data_dir[dataset] + "base.json"


@ex.capture
def get_val_file(dataset):
    if dataset == "cross":
        return configs.data_dir["CUB"] + "val.json"
    elif dataset == "cross_char":
        return configs.data_dir["emnist"] + "val.json"
    else:
        return configs.data_dir[dataset] + "val.json"


@ex.capture
def get_image_size(model, dataset):
    if "Conv" in model:
        if dataset in ["omniglot", "cross_char"]:
            return 28
        else:
            return 84
    else:
        return 224


@ex.capture
def get_stop_epoch(n_shot, method, dataset, stop_epoch):
    if stop_epoch == -1:
        if method in ["baseline", "baseline++"]:
            if dataset in ["omniglot", "cross_char"]:
                stop_epoch = 5
            elif dataset in ["CUB"]:
                # This is different as stated in the open-review paper. However,
                # using 400 epoch in baseline actually lead to over-fitting
                stop_epoch = 200
            elif dataset in ["miniImagenet", "cross"]:
                stop_epoch = 400
            else:
                stop_epoch = 400  # default
        else:  # meta-learning methods
            if n_shot == 1:
                stop_epoch = 600
            elif n_shot == 5:
                stop_epoch = 400
            else:
                stop_epoch = 600  # default

    return stop_epoch


@ex.capture
def get_n_query(test_n_way, train_n_way):
    # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    return max(1, int(16 * test_n_way / train_n_way))


@ex.capture
def get_base_loader(method, train_n_way, n_shot, train_aug):
    if method in ["baseline", "baseline++"]:
        base_datamgr = SimpleDataManager(get_image_size(), batch_size=16)
    else:
        base_datamgr = SetDataManager(
            get_image_size(), n_query=get_n_query(), n_way=train_n_way, n_support=n_shot
        )  # n_eposide=100

    return base_datamgr.get_data_loader(get_base_file(), aug=train_aug)


@ex.capture
def get_val_loader(method, test_n_way, n_shot, train_aug):
    if method in ["baseline", "baseline++"]:
        val_datamgr = SimpleDataManager(get_image_size(), batch_size=64)
    else:
        val_datamgr = SetDataManager(
            get_image_size(), n_query=get_n_query(), n_way=test_n_way, n_support=n_shot
        )

    return val_datamgr.get_data_loader(get_val_file(), aug=False)


@ex.capture
def validate_config(dataset, model, method, num_classes, train_aug):
    # dataset checks
    if dataset in ["omniglot", "cross_char"]:
        assert (
            model == "Conv4S" and train_aug is False
        ), "omniglot only support Conv4 without augmentation"

    # method checks
    if method in ["baseline", "baseline++"]:
        if dataset == "omniglot":
            assert (
                num_classes >= 4112
            ), "class number need to be larger than max label id in base class"
        if dataset == "cross_char":
            assert (
                num_classes >= 1597
            ), "class number need to be larger than max label id in base class"


@ex.capture
def get_model(
    model,
    dataset,
    method,
    num_classes,
    train_n_way,
    n_shot,
    num_draws,
    num_steps,
    sigma,
):
    train_few_shot_params = dict(n_way=train_n_way, n_support=n_shot)

    if method == "baseline":
        return BaselineTrain(model_dict[model], num_classes)
    elif method == "baseline++":
        return BaselineTrain(model_dict[model], num_classes, loss_type="dist")
    elif method == "ove_polya_gamma_gp":
        model = OVEPolyaGammaGP(model_dict[model], **train_few_shot_params)
        if num_draws is not None:
            model.num_draws = num_draws
        if num_steps is not None:
            model.num_steps = num_steps
        if sigma is not None:
            model.kernel.sigma = sigma
        return model
    elif method == "predictive_ove_polya_gamma_gp":
        model = PredictiveOVEPolyaGammaGP(
            model_dict[model], **train_few_shot_params, fast_inference=True
        )
        if num_draws is not None:
            model.num_draws = num_draws
        if num_steps is not None:
            model.num_steps = num_steps
        if sigma is not None:
            model.kernel.sigma = sigma
        return model
    elif method == "logistic_softmax_gp":
        model = LogisticSoftmaxGP(model_dict[model], **train_few_shot_params)
        if num_draws is not None:
            model.num_draws = num_draws
        if num_steps is not None:
            model.num_steps = num_steps
        if sigma is not None:
            model.kernel.sigma = sigma
        return model
    elif method == "predictive_logistic_softmax_gp":
        model = PredictiveLogisticSoftmaxGP(model_dict[model], **train_few_shot_params)
        if num_draws is not None:
            model.num_draws = num_draws
        if num_steps is not None:
            model.num_steps = num_steps
        if sigma is not None:
            model.kernel.sigma = sigma
        return model
    elif method == "bayesian_maml":
        model = BayesianMAML(
            model_dict[model],
            **train_few_shot_params,
            num_draws=num_draws,
            num_steps=num_steps
        )
        return model
    elif method == "chaser_bayesian_maml":
        return ChaserBayesianMAML(
            model_dict[model],
            **train_few_shot_params,
            num_draws=num_draws,
            num_steps=num_steps
        )
    elif method == "gpnet":
        model = GPNet(model_dict[model], **train_few_shot_params)
        model.init_summary()
        return model
    elif method == "protonet":
        return ProtoNet(model_dict[model], **train_few_shot_params)
    elif method == "matchingnet":
        return MatchingNet(model_dict[model], **train_few_shot_params)
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
        return RelationNet(feature_model, loss_type=loss_type, **train_few_shot_params)
    elif method in ["maml", "maml_approx"]:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(
            model_dict[model], approx=(method == "maml_approx"), **train_few_shot_params
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


def train(
    base_loader,
    val_loader,
    model,
    optimizer,
    start_epoch,
    stop_epoch,
    checkpoint_dir,
    writer,
    save_freq,
    max_acc,
    _run,
):
    print("Total epochs: {:d}".format(stop_epoch))

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        train_loss = model.train_loop(epoch, base_loader, optimizer)
        _run.log_scalar("train.loss", train_loss)
        writer.add_scalar("train.loss", train_loss, epoch)

        model.eval()
        val_acc = model.test_loop(val_loader)
        _run.log_scalar("val.acc", val_acc)
        writer.add_scalar("val.acc", val_acc, epoch)

        # for baseline and baseline++, we don't use validation here so we let acc = -1
        if val_acc > max_acc:
            print("--> Best model! save...")
            max_acc = val_acc
            outfile = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "max_acc": max_acc,
                },
                outfile,
            )

        if (epoch % save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(checkpoint_dir, "last_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "max_acc": max_acc,
                },
                outfile,
            )

        writer.flush()

    return model


@ex.automain
def main(method, start_epoch, optimization, save_freq, tag, seed, _run):
    print("using config: ", _run.config)
    print("save_dir: ", get_save_dir())

    validate_config()

    _set_seed(seed)

    max_acc = 0

    base_loader = get_base_loader()
    val_loader = get_val_loader()

    model = get_model()
    model = model.cuda()

    if optimization == "Adam":
        optimizer = torch.optim.Adam(model.parameters())
    else:
        raise ValueError("Unknown optimization, please define by yourself")

    stop_epoch = get_stop_epoch()

    if method == "maml" or method == "maml_approx":
        stop_epoch *= model.n_task  # maml use multiple tasks in one update

    writer = SummaryWriter(os.path.join(RUN_DIR, EXPERIMENT_NAME, tag, _run._id))

    model = train(
        base_loader,
        val_loader,
        model,
        optimizer,
        start_epoch,
        stop_epoch,
        os.path.join(get_save_dir(), str(_run._id)),
        writer,
        save_freq,
        max_acc,
        _run,
    )
