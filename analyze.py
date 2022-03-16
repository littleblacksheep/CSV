import os
import sys
import yaml
try: yaml.warnings({'YAMLLoadWarning': False})
except: pass
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
import dataloader as d
import models as m
import validate as v
import utils
import seaborn as sns
import scipy
import pandas as pd
import matplotlib as mpl
mpl.rc('image', cmap='seismic')
from colour import Color

def get_classes(data, n):
    classes = []
    for i in range(n):
        m_lr = data["m_largest"][i]
        m_2lr = data["m_2nd_largest"][i]
        m_t = data["M_targ"][i]
        m_p = data["M_proj"][i]
        if m_lr < m_t:  # erosion
            if m_lr < 0.5 * m_t:
                cl = 0
            else:
                cl = 1
        elif m_lr > m_t and m_2lr > 0.1 * m_p:  # hit and run
            if m_2lr < 0.5 * m_p:
                cl = 4
            else:
                cl = 5
        elif m_lr == m_t + m_p:  # perfect accretion
            cl = 3  # dummy class (can't happen in data)
        else:  # partial accretion
            if m_lr < m_t + 0.5 * m_p:
                cl = 2
            else:
                cl = 3
        classes.append(cl)
    return classes

def analyze_data(cfg):

    if cfg["ML"]["dataset"]["analysis"]["mode"] == "gather_all":

        print("INFO: gathering winter data (all)")
        dirs = utils.system.get_subfolders("{}winter/".format(cfg["SPH"]["system"]["path_Astro_SPH"]))
        n, ID = len(dirs), {}
        setups, results, dirs_all = [], [], []
        setups_invalid, results_invalid, dirs_all_invalid, t_calc = [], [], [], {"tot" : 0., "spheres_ini" : 0., "SPH" : 0., "postprocess" : 0.}
        print(f"INFO: found {n} directories. loading files...")
        for i, dir in enumerate(dirs):
            try:
                setup = np.load("{}/setup.npy".format(dir), allow_pickle=True).item()
                result = np.load("{}/results.npy".format(dir), allow_pickle=True).item()
                valid = utils.sim.get_valid(setup, result, i, dir)
                if valid:
                    dirs_all.append(dir)
                    setups.append(setup)
                    results.append(result)
                    t_calc["spheres_ini"] += setup["t_calc_spheres_ini"]
                    t_calc["tot"] += result["t_calc_tot"]
                    t_calc["SPH"] += result["t_calc_SPH"]
                    t_calc["postprocess"] += result["t_calc_postprocess"]
                else:
                    if setup["ID"].startswith("r"):
                        dirs_all_invalid.append(dir)
                        setups_invalid.append(setup)
                        results_invalid.append(result)
            except Exception as e:
                print(e)

        print("INFO: number of valid setups: {}".format(len(setups)))
        print("INFO: number of invalid setups: {}".format(len(setups_invalid)))
        print(f"INFO: t_calc (valid): {t_calc}")

        # load aggregate information (last frame)
        results_aggregates = []
        for i, dir in enumerate(dirs_all):
            n = setups[i]["n_frames"]
            z_n = torch.zeros(0)
            with open("{}/frames/aggregates.{:04d}".format(dir, n), 'r') as f:
                lines = f.read().splitlines()
                lines = [value for value in lines if not value.startswith("#")]  # remove lines that start with "#"
                for j, line in enumerate(lines):
                    z_line = torch.from_numpy(np.array([float(item) for item in line.split()], dtype=np.float32))
                    if j < 4 and z_line.shape[0] == 3:  # add water mass fraction = 0.
                        z_line = torch.cat((z_line, torch.zeros(1)), dim=0)
                    z_n = torch.cat((z_n, z_line), dim=0)  # combine all values into one pt vector
            f.close()
            agg = {"m_largest" : z_n[0].item(), "m_2nd_largest" : z_n[4].item(), "m_rest" : z_n[8].item()}
            results_aggregates.append(agg)

        data = {}
        keys = list(setups[0].keys()) + list(results_aggregates[0].keys())
        for key in keys:
            data[key] = []
        for i in range(len(setups)):
            for key, val in setups[i].items():
                data[key].append(val)
            for key, val in results_aggregates[i].items():
                data[key].append(val)
        data_invalid = {}
        keys_invalid = list(setups_invalid[0].keys())
        for key in keys_invalid:
            data_invalid[key] = []
        for i in range(len(setups_invalid)):
            for key, val in setups_invalid[i].items():
                try:
                    data_invalid[key].append(val)
                except:
                    try:
                        data_invalid[key].append(-1.)
                    except:
                        data_invalid[key] = [-1.]

        # save to file
        fname = "{}analysis/data.npy".format(cfg["SPH"]["system"]["path_Astro_SPH"])
        np.save(fname, data)
        print("INFO: saved {}".format(fname))
        fname = "{}analysis/data_invalid.npy".format(cfg["SPH"]["system"]["path_Astro_SPH"])
        np.save(fname, data_invalid)
        print("INFO: saved {}".format(fname))

    elif cfg["ML"]["dataset"]["analysis"]["mode"] == "filter":

        fname = "{}analysis/data.npy".format(cfg["SPH"]["system"]["path_Astro_SPH"])
        data = np.load(fname, allow_pickle='TRUE').item()
        n = len(data[list(data.keys())[0]])
        # example: filter for moon-scenario:
        filters = [["M_targ", 0.7 * (0.5 * 1.19444e+25), 1.3 * (0.5 * 1.19444e+25)],  # [key, min, max]
                   ["gamma",  0.111111111111-0.05, 0.111111111111+0.05],
                   ["impact_angle_touching_ball", 40., 50.],
                   ["vel_vesc_touching_ball", 1., 1.5]]
        i, ids = list(range(n)), []
        print(f"INFO: n: {len(i)}")
        for filter in filters:
            for j in range(len(i)-1, -1, -1):
                value = data[filter[0]][i[j]]
                if value < filter[1] or value > filter[2]:
                    del i[j]
            print(f"INFO: n after filter {filter}: {len(i)}")
        for j in i:
            ids.append(data["ID"][j])
        print(f"INFO: ids: {ids}")

    elif cfg["ML"]["dataset"]["analysis"]["mode"] == "stewart":
        """
        visualize data in impact velocity - impact angle plane
        """

        # load from files
        fname = "{}analysis/data.npy".format(cfg["SPH"]["system"]["path_Astro_SPH"])
        data = np.load(fname, allow_pickle='TRUE').item()
        print("INFO: loaded {}".format(fname))
        print("INFO: data.keys(): {}".format(data.keys()))

        x = np.array(data["impact_angle_touching_ball"])
        y = np.array(data["vel_vesc_touching_ball"])
        print("INFO: number of valid samples: {}".format(x.shape[0]))

        classes = get_classes(data, x.shape[0])

        dat_burger = np.loadtxt("{}burger/RSMC_dataset_ML.txt".format(cfg["SPH"]["system"]["path_Astro_SPH"]))
        dat_burger = dat_burger[dat_burger[:, 4] < 8., :]
        # scatter plot (colos are classes)
        colors = ["silver", "gray", "tomato", "saddlebrown", "dodgerblue", "green"]
        labels = [r"erosion, $m_{la}<m_t/2$", r"erosion, $m_{la}>m_t/2$", r"accretion, $m_{la}<m_t+m_p/2$",
                  r"accretion, $m_{la}>m_t+m_p/2$", r"hit&run, $m_{2la}<m_p/2$", r"hit&run, $m_{2la}>m_p/2$"]
        size = 8
        plt.figure(figsize=(size,0.9*size))
        for cl in range(6):
            idx = [i for i, j in enumerate(classes) if j == cl]  # datapoints of respective class cl
            plt.scatter(x[idx], y[idx], color=colors[cl], alpha=0.5, marker=".", s=15, label="{}".format(labels[cl]))
            print("winter. labels, len, fraction:", labels[cl], x[idx].shape[0], x[idx].shape[0] / x.shape[0])
        sns.kdeplot(x=dat_burger[:, 5], y=dat_burger[:, 4], color="black")
        plt.xlabel("impact angle [deg]", fontsize=cfg["ML"]["analysis"]["fontsize"])
        plt.ylabel("impact velocity [$v_{esc}$]", fontsize=cfg["ML"]["analysis"]["fontsize"])
        plt.xlim([0., 90.])
        plt.ylim([1., 8.])
        plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2, fontsize=cfg["ML"]["analysis"]["fontsize"])
        plt.grid(color='gray', linestyle=':')
        fname = "{}analysis/stewart_classes.eps".format(cfg["SPH"]["system"]["path_Astro_SPH"])
        plt.tight_layout()
        plt.savefig(fname, format='eps' , dpi=600)
        plt.close()
        print(f"INFO: saved {fname}")

        #  1.RSMC-run  2.col.no.  3.time  4.impact-vel  5.impact-vel/vesc  6.impact-angle  7.mass-p  8.mass-t      9.mass-la    10.mass-sla       11.wmf-p       12.wmf-t      13.wmf-la     14.wmf-sla       15.cmf-p       16.cmf-t      17.cmf-la     18.cmf-sla
        dat_burger = dat_burger[dat_burger[:,9] > 0.,:]
        data["m_largest"] = dat_burger[:,8]
        data["m_2nd_largest"] = dat_burger[:,9]
        data["M_targ"] = dat_burger[:,7]
        data["M_proj"] = dat_burger[:,6]

        classes = get_classes(data, data["m_largest"].shape[0])

        for cl in range(6):
            idx = [i for i, j in enumerate(classes) if j == cl]  # datapoints of respective class cl
            print("burger. labels, len, fraction:", labels[cl], len(idx), len(idx) / len(classes))

    elif cfg["ML"]["dataset"]["analysis"]["mode"] == "invalid":
        """
        visualize invalid simulations
        """

        # load from files
        fname = "{}analysis/data_invalid.npy".format(cfg["SPH"]["system"]["path_Astro_SPH"])
        data_invalid = np.load(fname, allow_pickle='TRUE').item()
        print("INFO: loaded {}".format(fname))
        print("INFO: data.keys(): {}".format(data_invalid.keys()))

        x = np.array(data_invalid["impact_angle_touching_ball"])
        y = np.array(data_invalid["vel_vesc_touching_ball"])
        print("INFO: n_samples: {}".format(x.shape[0]))

        plt.figure()
        plt.scatter(x, y, alpha=0.5, marker=".", s=15)
        plt.xlabel("impact angle [deg]")
        plt.ylabel("impact velocity [$v_{esc}$]")
        plt.xlim([0., 90.])
        plt.ylim([1., 8.])
        plt.grid(color='gray', linestyle=':')
        fname = "{}analysis/data_invalid_scatter.eps".format(cfg["SPH"]["system"]["path_Astro_SPH"])
        plt.savefig(fname, format='eps' , dpi=600)
        plt.close()
        print(f"INFO: saved {fname}")
        print("INFO: invalid IDs: {}".format(data_invalid["ID"]))

def load_stats(cfg, run, typ, keys, splits, modes):

    stats, stats_comb = {}, {}
    for fold in range(cfg["ML"]["training"]["folds"]):
        stats[fold], stats_comb[fold] = {}, {}
        for epoch in range(cfg["ML"]["training"]["epochs"]+1):
            if epoch % cfg["ML"]["training"]["verbosity"] == 0 or epoch == cfg["ML"]["training"]["epochs"]:
                fname = "{}run{:03d}/stats_type{}_fold{}_epoch{}.npy".format(cfg["ML"]["system"]["dir_analyze"], run, typ, fold, epoch)
                try:
                    stats[fold][epoch] = np.load(fname, allow_pickle=True).item()
                except Exception as e:
                    if fold == 0: print(e)

        print(f"INFO: run: {run}, fold: {fold}, loaded {len(stats[fold])} checkpoints")

        # combine along epochs:
        if len(stats[fold]) > 0:
            for split in splits:
                stats_comb[fold][split] = {}
                for mode in modes:
                    stats_comb[fold][split][mode] = {"epochs" : []}
                    for key in keys:
                        stats_comb[fold][split][mode][key + "_mean"] = []
                        for epoch in list(stats[0].keys()):
                            try:
                                stats_comb[fold][split][mode][key + "_mean"].append(stats[fold][epoch][split][mode][key + "_mean"])
                            except Exception as e:
                                stats_comb[fold][split][mode][key + "_mean"].append(stats[0][epoch][split][mode][key + "_mean"])
                            if key == "tot":
                                stats_comb[fold][split][mode]["epochs"].append(epoch)
    return [stats, stats_comb]

def analyze_results(cfg):
    """
    visualize and compare results of ML models
    """

    runs = [0, 1]
    wilcox_runs = [0, 1] # within runs list
    modes = ["last_frame"]
    paper_plot = False

    keys = ['mass', 'fraction_mat', 'position', 'velocity', 'tot', 'acc']
    splits = ["train", "vali", "test"]
    y_max = {"tot": 0.05, "mass": 0.05, "fraction_mat": 0.1, "position": 0.1, "velocity": 0.1, "acc": .6}
    y_min = {"tot": 0.02, "mass" : 0., "fraction_mat" : 0., "position" : 0., "velocity" : 0., "acc" : 0.2}

    keys_tasks = ['mass', 'fraction_mat', 'position', 'velocity', 'tot']
    for typ in ["png", "eps", "pdf"]:
        command = "rm {}*.{}".format(cfg["ML"]["system"]["dir_analyze"], typ)
        utils.system.execute(command)
    label = {"linear" : "LIN", "pm" : "PIM", "ffn" : "FFN", "res" : "RES"}
    types = {}
    for run in runs:
        fname = "{}run{:03d}/cfg.npy".format(cfg["ML"]["system"]["dir_analyze"], run)
        cfg_run = np.load(fname, allow_pickle=True).item()
        types[run] = cfg_run["ML"]["model"]["type"]
        print(f"INFO: run {run} cfg: {cfg_run['ML']}")

    # load:
    results = []
    for run in runs:
        result = load_stats(cfg, run, types[run], keys, splits, modes)
        results.append(result)
    print(f"INFO: len(results): {len(results)}")
    for i, result in enumerate(results):
        run = runs[i]
        typ = types[run]
        stats = result[0]
        print(f"INFO: run: {run}, typ: {typ}")

    if paper_plot:
        splits = ["test"]
    for split in splits:
        for mode in modes:
            for key in keys:
    
                # visualize loss curves:
                #for split in ["train", "vali", "test"]: # uncomment for paper_plot
                if True: # comment for paper_plot
                    print(f"INFO: split, mode, key: {split, mode, key}")
                    for i, result in enumerate(results):
                        run = runs[i]
                        typ = types[run]
                        stats, stats_comb = result[0], result[1]
                        epochs = stats_comb[0][split][mode]["epochs"]
                        folds = []
                        for fold in range(cfg["ML"]["training"]["folds"]):
                            try:
                                folds.append(stats_comb[fold][split][mode][key + "_mean"])
                            except Exception as e:
                                pass
                        folds = np.array(folds) # [fold,epoch]
                        avg = np.mean(folds, axis=0)
                        min_ = np.min(folds, axis=0)
                        max_ = np.max(folds, axis=0)
                        if not paper_plot:
                            plt.plot(epochs, avg, label="{} {}".format(run, label[typ]))
                            plt.fill_between(epochs, min_, max_, alpha=0.4)
                        else:
                            colors = {"train": "green", "vali": "blue", "test": "black"}
                            linestyles = {"ffn" : ":", "res" : "-"}
                            plt.plot(epochs, avg, linestyle=linestyles[typ], color=colors[split], label="{} {}".format(label[typ], split))
                            plt.fill_between(epochs, min_, max_, color=colors[split], alpha=0.3)
                plt.legend(fontsize=cfg["ML"]["analysis"]["fontsize"])
                plt.grid(color='gray', linestyle=':')
                plt.xlabel("epoch", fontsize=cfg["ML"]["analysis"]["fontsize"])
                plt.ylabel("RMSE", fontsize=cfg["ML"]["analysis"]["fontsize"])
                plt.ylim([y_min[key], y_max[key]])
                fname = "{}1_RMSE_{}_{}_{}.pdf".format(cfg["ML"]["system"]["dir_analyze"], key, split, mode)
                plt.savefig(fname, dpi=600, bbox_inches='tight')
                plt.close()
                print(f"INFO: saved {fname}")
    
            # barplot of subtasks:
            n_runs = len(runs)
            n_tasks = len(keys_tasks)
            data = {"values_all" : torch.zeros((n_runs, n_tasks+1, cfg["ML"]["training"]["folds"])),
                    "values" : torch.zeros((n_runs, n_tasks+1)),
                    "errors_high" : torch.zeros((n_runs, n_tasks+1)),
                    "errors_low" : torch.zeros((n_runs, n_tasks+1))}
            for i, result in enumerate(results):
                run = runs[i]
                typ = types[run]
                for j, key in enumerate(keys_tasks + ["acc"]):
                    stats, stats_comb = result[0], result[1]
                    folds_best = []
                    for fold in range(cfg["ML"]["training"]["folds"]):
                        try:
                            if key == "acc":
                                folds_best.append(np.max(np.array(stats_comb[fold][split][mode][key + "_mean"])))  # best over all epochs
                            else:
                                folds_best.append(np.min(np.array(stats_comb[fold][split][mode][key + "_mean"])))  # best over all epochs
                        except Exception as e:
                            print(e)
                            pass
                    folds_best = torch.from_numpy(np.array(folds_best))
                    avg = torch.mean(folds_best)
                    max_ = folds_best.max()
                    min_ = folds_best.min()
                    data["values_all"][i, j, :] = folds_best
                    data["values"][i, j] = avg
                    data["errors_high"][i, j] = max_ - avg
                    data["errors_low"][i, j] = min_ - avg

                # latex table syntax:
                print("run: {} split: {}: {} ".format(run, split, typ)
                      + "& ${:.4f}^".format(data["values"][i, 0]) + repr("{") + "{:+.4f}".format(
                    data["errors_high"][i, 0]) + repr("}") + "_" + repr("{") + "{:+.4f}".format(
                    data["errors_low"][i, 0]) + repr("}") + "$"
                      + "& ${:.4f}^".format(data["values"][i, 1]) + repr("{") + "{:+.4f}".format(
                    data["errors_high"][i, 1]) + repr("}") + "_" + repr("{") + "{:+.4f}".format(
                    data["errors_low"][i, 1]) + repr("}") + "$"
                      + "& ${:.4f}^".format(data["values"][i, 2]) + repr("{") + "{:+.4f}".format(
                    data["errors_high"][i, 2]) + repr("}") + "_" + repr("{") + "{:+.4f}".format(
                    data["errors_low"][i, 2]) + repr("}") + "$"
                      + "& ${:.4f}^".format(data["values"][i, 3]) + repr("{") + "{:+.4f}".format(
                    data["errors_high"][i, 3]) + repr("}") + "_" + repr("{") + "{:+.4f}".format(
                    data["errors_low"][i, 3]) + repr("}") + "$"
                      + "& ${:.4f}^".format(data["values"][i, 4]) + repr("{") + "{:+.4f}".format(
                    data["errors_high"][i, 4]) + repr("}") + "_" + repr("{") + "{:+.4f}".format(
                    data["errors_low"][i, 4]) + repr("}") + "$"
                      + "& ${:.4f}^".format(data["values"][i, 5]) + repr("{") + "{:+.4f}".format(
                    data["errors_high"][i, 5]) + repr("}") + "_" + repr("{") + "{:+.4f}".format(
                    data["errors_low"][i, 5]) + repr("}") + "$"
                      )

            barWidth = 0.1
            fig = plt.subplots()
            for i, run in enumerate(runs):
                br = np.arange(n_tasks) + i * barWidth
                errors = torch.cat((data["errors_low"][i,:n_tasks].abs().unsqueeze(0),
                                    data["errors_high"][i,:n_tasks].abs().unsqueeze(0)), dim=0) # [2,n]
                plt.bar(br, data["values"][i,:n_tasks], yerr=errors, width=barWidth, edgecolor='black', label=f'{run} {label[types[run]]}')
                if i == 0:
                    plt.xticks(br, ['mass', 'material\nfraction', 'position', 'velocity', 'total'])
            plt.legend()
            plt.grid(color='gray', linestyle=':')
            plt.xlabel("task")
            plt.ylabel("RMSE")
            fname = "{}3_RMSE_subtasks_{}_{}.png".format(cfg["ML"]["system"]["dir_analyze"], split, mode)
            plt.savefig(fname, dpi=600, bbox_inches='tight')
            plt.close()
            print(f"INFO: saved {fname}")

            # perform significance test:
            if len(wilcox_runs) == 2:
                for task, key in enumerate(keys_tasks + ["acc"]):
                    wilcox_x = data["values_all"][wilcox_runs[0],task,:]
                    wilcox_y = data["values_all"][wilcox_runs[1],task,:]
                    w, p = scipy.stats.wilcoxon(wilcox_x, wilcox_y, zero_method='wilcox', correction=False, alternative='two-sided')
                    print("INFO: Wilcox (runs: {} vs. {}, task: {}) w: {:.4f}, p: {:.4f}".format(runs[wilcox_runs[0]], runs[wilcox_runs[1]], key, w, p))

    return

def plot_predicted_classes(cfg, run, stats, mode, epoch_final):
    splits = ["vali", "test"]
    mode = "last_frame"
    test_regions = [{"v_min": 1.5, "v_max": 2.5, "a_min": 10., "a_max": 30.},
                    {"v_min": 2., "v_max": 4., "a_min": 65., "a_max": 75.},
                    {"v_min": 1., "v_max": 2., "a_min": 80., "a_max": 90.},
                    {"v_min": 6., "v_max": 8., "a_min": 0.,
                     "a_max": 20.}]  # impact velocity, impact angle
    x = [cat_folds(stats, splits[0], mode, "impact_angle_touching_ball", epoch_final),
         cat_folds(stats, splits[1], mode, "impact_angle_touching_ball", epoch_final)]
    y = [cat_folds(stats, splits[0], mode, "vel_vesc_touching_ball", epoch_final),
         cat_folds(stats, splits[1], mode, "vel_vesc_touching_ball", epoch_final)]
    cl_y = [cat_folds(stats, splits[0], mode, "cl_y", epoch_final),
            cat_folds(stats, splits[1], mode, "cl_y", epoch_final)]
    colors = ["silver", "gray", "tomato", "saddlebrown", "dodgerblue", "green"]
    region_color = "black"
    labels = ["erosion, m_lr<m_t/2", "erosion, m_lr>m_t/2", "accretion, m_lr<m_t+m_p/2",
              "accretion, m_lr>m_t+m_p/2", "hit&run, m_2lr<m_p/2", "hit&run, m_2lr>m_p/2"]
    plt.figure()
    for i, split in enumerate(splits):
        for cl in range(6):
            idx = cl_y[i] == cl  # datapoints of respective class cl
            plt.scatter(x[i][idx], y[i][idx], color=colors[cl], alpha=0.5, marker=".", s=15)
    for r in range(len(test_regions)):
        region = test_regions[r]
        plt.plot([region["a_min"], region["a_max"]], [region["v_min"], region["v_min"]], color=region_color, linestyle="dashed")  # bottom
        plt.plot([region["a_min"], region["a_max"]], [region["v_max"], region["v_max"]], color=region_color, linestyle="dashed")  # top
        plt.plot([region["a_min"], region["a_min"]], [region["v_min"], region["v_max"]], color=region_color, linestyle="dashed")  # left
        plt.plot([region["a_max"], region["a_max"]], [region["v_min"], region["v_max"]], color=region_color, linestyle="dashed")  # right
    plt.xlabel("impact angle [deg]", fontsize=cfg["ML"]["analysis"]["fontsize"])
    plt.ylabel("impact velocity [$v_{esc}$]", fontsize=cfg["ML"]["analysis"]["fontsize"])
    plt.xlim([0., 90.])
    plt.ylim([1., 8.])
    plt.grid(color='gray', linestyle=':')
    fname = "{}run{}_classes_{}_{}.eps".format(cfg["ML"]["system"]["dir_analyze"], run, splits[0] + splits[1], mode)
    plt.tight_layout()
    plt.savefig(fname, format='eps', dpi=600)
    plt.close()
    print(f"INFO: saved {fname}")

def cat_folds(stats, split, mode, key, epoch_final):
    x = stats[0][epoch_final][split][mode][key]
    if split != "test":
        for f in range(1, 5):
            try:
                x = torch.cat((x, stats[f][epoch_final][split][mode][key]))
            except Exception as e:
                print(f"WARNING: {e}. split: {split}, fold: {f}")
    return x

def analyze_model(cfg):
    """
    investigate model. more code snipplets can be found in misc/analyze_.py
    """

    run = cfg["ML"]["system"]["run"]
    fname = "{}run{:03d}/cfg.npy".format(cfg["ML"]["system"]["dir_analyze"], run)
    cfg_run = np.load(fname, allow_pickle=True).item()
    typ = cfg_run["ML"]["model"]["type"]
    print(f"INFO: run {run} cfg: {cfg}")
    print("INFO: cfg: ", cfg_run["ML"]["model"][typ])

    dataset = d.get_dataset(cfg)
    model = m.model_load(cfg, run)
    fold = 0
    stats_analyze = v.validate(cfg, model, cfg["ML"]["training"]["epochs"], dataset, fold, analyze=True)

    #modes = ["last_frame", "all_frames"]
    modes = ["last_frame"]
    keys = ['mass', 'fraction_mat', 'position', 'velocity', 'tot', 'acc']
    for ext in ["png", "eps", "pdf"]:
        command = "rm {}*.{}".format(cfg["ML"]["system"]["dir_analyze"], ext)
        utils.system.execute(command)

    result = load_stats(cfg, run, typ, keys, ["train", "vali", "test"], modes)
    stats = result[0]

    epoch_final = max(list(stats[fold].keys()))
    print(f"INFO: epoch_final: {epoch_final}")
    plot_predicted_classes(cfg, run, stats, "last_frame", epoch_final)

if __name__ == '__main__':
    print('INFO: Done.')