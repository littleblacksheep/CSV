import shutil
import torch
from torch.utils import data
import utils
import numpy as np
import time
import matplotlib.pyplot as plt
import dataloader as d
import models as m

@torch.no_grad()
def validate(cfg, model_ema, epoch, dataset, fold, analyze=False):
    print("INFO: validate {}".format(cfg["ML"]["model"]["type"]))
    keys = ["mass", "fraction_mat", "position", "velocity", "tot"]
    stats = {}
    splits = ["train", "vali", "test"]
    if cfg["ML"]["model"]["type"] == "pm" or cfg["ML"]["dataset"]["name"] == "timpe":
        modes = ["last_frame"]
    else:
        modes = ["last_frame", "all_frames"]

    t0 = time.time()
    for split in splits:
        stats[split] = {}
        for mode in modes:
            print(f"INFO: mode: {mode}")
            stats[split][mode] = {}
            dataset["data"].init_samples(split, fold, verbose=True)
            params = dataset["params"]
            params["shuffle"] = False
            loader_split = data.DataLoader(dataset["data"], **params)
            for batch, xz in enumerate(loader_split):
                x, z = d.preprocess(cfg, xz, mode)
                info = {"z0": z[0], "mode" : mode}
                out = model_ema.forward(x, info)
                y = out["y_lbl"]
                stats_ = utils.ML.loss(cfg, y, z, L_reg=None, validate=True, x=x)
                if analyze and mode == "all_frames" and cfg["ML"]["model"]["type"] == "res":
                    if batch == 0: y_all = y
                    else: y_all = torch.cat((y_all, y), dim=1)
                # accumulate stats for all samples:
                for key in list(stats_.keys()):
                    if batch == 0: stats[split][mode][key] = stats_[key]
                    else: stats[split][mode][key] = torch.cat((stats[split][mode][key], stats_[key]))
            stats[split][mode]["n_samples"] = stats[split][mode]["tot"].shape[0]
            stats[split][mode]["acc_mean"] = utils.ML.balanced_acc(stats[split][mode]["cl_y"], stats[split][mode]["cl_z"])
            for key in keys:
                stats[split][mode][key + "_mean"] = stats[split][mode][key].mean()  # avg over samples
            print("INFO: stats[{}][{}]['tot_mean']: {}".format(split, mode, stats[split][mode]["tot_mean"]))
            print("INFO: stats[{}][{}]['acc_mean']: {}".format(split, mode, stats[split][mode]["acc_mean"]))

            if epoch == cfg["ML"]["training"]["epochs"]:
                dat = {}
                dat["x_setup"] = x.squeeze(1).detach().cpu()
                dat["z0"] = z[0].detach().cpu()
                dat["z1"] = z[1].detach().cpu()
                try:
                    dat["z_seq"] = z[2].detach().cpu()
                except:
                    pass
                stats[split][mode]["dat"] = dat

                for key, val in out.items():
                    try:
                        out[key] = out[key].detach().cpu()
                    except:
                        pass
                stats[split][mode]["out"] = out

            else: # dummy values
                for key in keys:
                    stats[split][mode][key] = torch.zeros(1)
                stats[split][mode]["cl_z"] = torch.zeros(1)
                stats[split][mode]["cl_y"] = torch.zeros(1)
                stats[split][mode]["impact_angle_touching_ball"] = torch.zeros(1)
                stats[split][mode]["vel_vesc_touching_ball"] = torch.zeros(1)
                stats[split][mode]["L"] = torch.zeros(1)

    if not analyze and epoch == cfg["ML"]["training"]["epochs"] and cfg["ML"]["model"]["save"] and cfg["ML"]["model"]["type"] != "pm" and cfg["ML"]["model"]["type"] != "identity":
        fname = "{}model_ema.pt".format(cfg["ML"]["system"]["dir_check"])
        torch.save(model_ema, fname)
        print(f"INFO: saved {fname}")

    t1 = time.time()
    n_samples_tot = 0
    for split in splits:
        for mode in modes:
            n_samples_tot += stats[split][mode]["n_samples"]
    stats["t_calc_mean"] = (t1 - t0) / n_samples_tot
    if cfg["ML"]["model"]["type"] == "pm": stats["n_params"] = 0
    else: stats["n_params"] = utils.ML.n_params(model_ema)
    if not analyze:
        fname = "{}stats_type{}_fold{}_epoch{}.npy".format(cfg["ML"]["system"]["dir_check"], cfg["ML"]["model"]["type"], fold, epoch)
        np.save(fname, stats)
        print(f"INFO: saved {fname}")
    return stats