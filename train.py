import torch
from torch.utils import data
import dataloader as d
import validate as v
import models as m
import utils
import time

def get_optimizer(cfg, model):
    optimizer = {}
    if cfg["ML"]["training"]["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["ML"]["training"]["lr"]*(1.-cfg["ML"]["training"]["momentum"]), momentum=cfg["ML"]["training"]["momentum"], weight_decay=cfg["ML"]["training"]["weight_decay"]) # Note P. Renz: have to reduce lr when using momentum to get same results
    elif cfg["ML"]["training"]["optimizer"] == "adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=cfg["ML"]["training"]["lr"], betas=(cfg["ML"]["training"]["beta1"], cfg["ML"]["training"]["beta2"]), eps=1.e-08, weight_decay=cfg["ML"]["training"]["weight_decay"])
    print('INFO: optimizer {} ready'.format(cfg["ML"]["training"]["optimizer"]))
    return optimizer

def update_ema(model, model_ema, rate_ema):
    """update parameters of exponential moving average model"""
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(rate_ema)
        p2.data.add_(p1.data * (1 - rate_ema))

def train(cfg):

    print("INFO: training settings: {}".format(cfg["ML"]["training"]))
    utils.ML.prepare_checkpoint_directory(cfg)
    dataset = d.get_dataset(cfg)

    for fold in range(cfg["ML"]["training"]["folds"]):

        if cfg["ML"]["model"]["type"] == "pm":  # analytic model --> only evaluate
            model, model_ema = m.init_model_analyze(cfg)
            v.validate(cfg, model_ema, cfg["ML"]["training"]["epochs"], dataset, fold)

        else:
            model, model_ema = m.init_model_train(cfg)
            optimizer = get_optimizer(cfg, model)
            dataset["data"].init_samples("train", fold)
            loader_train = data.DataLoader(dataset["data"], **dataset["params"])
            t_start_train = time.time()
            print("INFO: training has started. fold: {}/{}".format(fold, cfg["ML"]["training"]["folds"]-1))
            for epoch in range(cfg["ML"]["training"]["epochs"]+1):
                for batch, xz in enumerate(loader_train):
                    x, z = d.preprocess(cfg, xz, cfg["ML"]["training"]["mode"])

                    optimizer.zero_grad()
                    info = {"z0": z[0], "mode" : cfg["ML"]["training"]["mode"]}
                    out = model(x, info)
                    L = utils.ML.loss(cfg, out["y_lbl"], z, out["L_reg"])
                    L.backward()
                    m.mod_clip_grad(cfg, model)
                    optimizer.step()
                    update_ema(model, model_ema, cfg["ML"]["training"]["rate_ema"])

                if epoch == 0:
                    t_end_train = time.time()
                    print(f"INFO: required training time (epoch 0): {t_end_train - t_start_train} ({t_end_train}, {t_start_train})")

                if epoch % cfg["ML"]["training"]["verbosity"] == 0 or epoch == cfg["ML"]["training"]["epochs"]:
                    print("\nINFO: checkpoint")
                    v.validate(cfg, model_ema, epoch, dataset, fold)
                    dataset["data"].init_samples("train", fold)

    return
                    
if __name__ == '__main__':
    print('Done.')