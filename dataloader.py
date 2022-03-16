import os
import random
import numpy as np
import pickle
import torch
from torch.utils import data
import utils
import matplotlib.pyplot as plt
import time
import pynbody
import pylab

'''
input features:
type_x == "setup":
    0: "N" [1]
    1: "M_tot" [kg]
    2: "gamma" [1.]
    3: "core_proj" [1.]
    4: "shell_proj" [1.]
    5: "core_targ" [1.]
    6: "shell_targ" [1.]
    7 - 9: "rot_axis_proj". length of vector = omega [rad/s]
    10 - 12: "rot_axis_targ". length of vector = omega [rad/s]
    13 - 18: "S_proj". units: [m], [m/s]
    19 - 24: "S_targ". units: [m], [m/s]
    --> n_x_setup: 25

target features:
type_z == "agg":
    for all frames:
        0-2: mass, fraction_mat0, fraction_mat2 (largest). units: [kg], [1.], [1.]
        3-5: mass, fraction_mat0, fraction_mat2 (2nd largest). units: [kg], [1.], [1.]
        6-8: mass, fraction_mat0, fraction_mat2 (rest). units: [kg], [1.], [1.]
        9-14: pos and vel (largest). units: [m], [m/s]
        15-20: pos and vel (2nd largest). units: [m], [m/s]
        21-26: pos and vel (rest). units: [m], [m/s]
    --> n_z_agg : 27
'''

class Dataset_SPH_winter(data.Dataset):
    def __init__(self, cfg):
        print(f"INFO: init dataset winter")
        self.cfg = cfg
        self.hyper = {'mu': {'N': 0.,
                             'mass_agg': 0.,
                             'gamma': 0.,
                             'fraction_mat': 0.,
                             'spin': 0.,
                             'pos_vel': [0., 0., 0., 0., 0., 0.]},
                      # no shift in pos and vel --> barycenter remains at [0,0,0,0,0,0]
                      'std': {'N': 50000.,
                              'mass_agg': 1e+25,
                              'gamma': 1.,
                              'fraction_mat': 1.,
                              'spin': 6.5e-05,
                              'pos_vel': [5e+07, 2e+08, 2e+07, 2e+03, 1e+04, 6e+02]}
                      }
        self.get_all()
        self.get_x()
        self.get_z()
        self.n_samples_all = len(self.dirs_all)
        self.init_samples("train", 0)
        print("INFO: init dataset finished")

    def get_all(self):
        self.type_x = self.cfg["ML"]["dataset"]["winter"]["inputs"]
        self.type_z = self.cfg["ML"]["dataset"]["winter"]["targets"]
        if self.type_x != "setup" or self.type_z != "agg":
            raise NotImplementedError()
        fname_dirs_all = "{}misc/dirs_all.npy".format(self.cfg["SPH"]["system"]["path_Astro_SPH"])
        fname_info_all = "{}misc/info_all.npy".format(self.cfg["SPH"]["system"]["path_Astro_SPH"])
        try:  # fast
            self.dirs_all = np.load(fname_dirs_all, allow_pickle=True)
            self.info_all = np.load(fname_info_all, allow_pickle=True).item()
            print(f"INFO: loaded {fname_dirs_all} and {fname_info_all}")
            n_max = self.cfg["ML"]["dataset"]["winter"]["n_samples_max"]
            if len(self.dirs_all) > n_max:
                self.dirs_all = self.dirs_all[:n_max]
                for key in self.info_all.keys():
                    self.info_all[key] = self.info_all[key][:n_max]
                print(f"INFO: sliced to {n_max} samples")
        except Exception as e:  # takes a few minutes
            print(f"INFO: failed to load {fname_dirs_all} and {fname_info_all} ({e}). finding valid datapoints...")
            dirs = utils.system.get_subfolders("{}csv/".format(self.cfg["SPH"]["system"]["path_Astro_SPH"]))
            dirs.sort()  # sort by ID
            n, ID, self.dirs_all, dirs_invalid = len(dirs), {}, [], []
            self.info_all = {"n_frames": [], "vel_vesc_touching_ball": [], "impact_angle_touching_ball": []}
            print(f"INFO: found {n} directories")
            for i, dir in enumerate(dirs):
                try:
                    setup = np.load("{}/setup.npy".format(dir), allow_pickle=True).item()
                    result = np.load("{}/results.npy".format(dir), allow_pickle=True).item()
                    valid = utils.sim.get_valid(setup, result, i, dir)
                    if valid:
                        self.dirs_all.append(dir)
                        for key in self.info_all.keys():
                            self.info_all[key].append(setup[key])
                    else:
                        dirs_invalid.append(dir)
                except Exception as e:
                    print(f"WARNING: {e}")
                if len(self.dirs_all) == self.cfg["ML"]["dataset"]["winter"]["n_samples_max"]:
                    print("INFO: reached n_samples_max ({})".format(self.cfg["ML"]["dataset"]["winter"]["n_samples_max"]))
                    break
            n = len(self.dirs_all)
            print(f"INFO: number of valid datapoints: {n}")
            print(f"INFO: number of invalid datapoints (unused): {len(dirs_invalid)}")
            np.save(fname_dirs_all, self.dirs_all)
            np.save(fname_info_all, self.info_all)
            print(f"INFO: saved {fname_dirs_all} and {fname_info_all}")

    def get_x(self):
        print(f'INFO: get_x: type: {self.type_x}')
        processed = True
        fname_x = "{}misc/x_{}_processed{}.pt".format(self.cfg["SPH"]["system"]["path_Astro_SPH"], self.type_x, processed)
        fname_x_raw = "{}misc/x_{}_processed{}.pt".format(self.cfg["SPH"]["system"]["path_Astro_SPH"], self.type_x, False)
        if os.path.isfile(fname_x):
            self.x = torch.load(fname_x)
            if self.x.shape[0] != len(self.dirs_all):
                raise ValueError(f"ERROR: invalid self.x ({self.x.shape[0]}, {len(self.dirs_all)})")
            print(f"INFO: loaded {fname_x}")
        elif os.path.isfile(fname_x_raw):  # only apply raw2ML
            self.x = torch.load(fname_x_raw)
            if self.x.shape[0] != len(self.dirs_all):
                raise ValueError(f"ERROR: invalid self.x ({self.x.shape[0]}, {len(self.dirs_all)})")
            print(f"INFO: loading {fname_x} failed ... loaded {fname_x_raw} instead ...")
            self.x, _ = raw2ML(self.cfg, self.type_x, self.hyper, x=self.x, z=None, inverse=False)
            torch.save(self.x, fname_x)
            print(f"INFO: saved {fname_x}")
        else:
            print(f"INFO: loading {fname_x} failed ... processing data ...")
            if self.type_x == "setup":

                n_x = self.cfg["ML"]["dataset"]["winter"][f"n_x_{self.type_x}"]
                self.x = torch.zeros((0, n_x + 2))  # [b,n_x+2]

                # keys to load:
                keys = ["N", "M_tot", "gamma", "core_proj", "shell_proj", "core_targ", "shell_targ", "proj_rot_period",
                        "rot_axis_proj", "targ_rot_period", "rot_axis_targ", "S_proj", "S_targ"]

                for dir in self.dirs_all:
                    setup = np.load("{}/setup.npy".format(dir), allow_pickle=True).item()
                    x = torch.zeros(n_x + 2)
                    i = 0
                    for key in keys:
                        if type(setup[key]) == list:
                            for entry in setup[key]:
                                x[i] = float(entry)
                                i += 1
                        elif type(setup[key]) == np.ndarray:
                            for j in range(setup[key].shape[0]):
                                x[i] = float(setup[key][j])
                                i += 1
                        else:
                            x[i] = float(setup[key])
                            i += 1
                    self.x = torch.cat((self.x, x.unsqueeze(0)), dim=0)

                for c in [7, 10]:
                    self.x[:, c:c + 3] = (2. * np.pi / self.x[:, c]).unsqueeze(-1) * self.x[:,
                                                                                     c + 1:c + 4] / torch.norm(
                        self.x[:, c + 1:c + 4], dim=1, keepdim=True)
                    self.x = torch.cat((self.x[:, :c + 3], self.x[:, c + 4:]), dim=1)  # delete column c+3

                self.x = self.x.unsqueeze(1)  # [b,1,n_x]

            self.x, _ = raw2ML(self.cfg, self.type_x, self.hyper, x=self.x, z=None, inverse=False)

            torch.save(self.x.cpu(), fname_x)
            print(f"INFO: saved {fname_x}")

    def get_z_n_agg(self, dir, n):
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

        # delete columns for distance, v_rel, v_rel/v_esc, S_scenario:
        z_n = z_n[:34]
        # delete columns 12-15 (scenario):
        z_n = torch.cat((z_n[:12], z_n[12 + 4:]), dim=0)
        # delete columns 10, 6, 2 (fraction_mat1):
        cols = [10, 6, 2]
        for c in cols:
            z_n = torch.cat((z_n[:c], z_n[c + 1:]), dim=0)

        z_n = z_n.unsqueeze(0)  # [1,n_z]
        return z_n

    def get_z(self):
        print(f'INFO: get_z: type: {self.type_z}')
        self.z = []
        random_setup = self.cfg["SPH"]["sim"]["random_setup"]
        frame_interval = random_setup["frame_interval"]
        n_z_agg = self.cfg["ML"]["dataset"]["winter"][f"n_z_agg"]
        n_frames_nonzero_max = self.cfg["ML"]["dataset"]["winter"]["n_frames_nonzero_max"]
        processed = True
        fname_z = "{}misc/z_{}_processed{}.pickle".format(self.cfg["SPH"]["system"]["path_Astro_SPH"], self.type_z, processed)
        fname_z_raw = "{}misc/z_{}_processed{}.pickle".format(self.cfg["SPH"]["system"]["path_Astro_SPH"], self.type_z, False)
        if os.path.isfile(fname_z):
            with open(fname_z, 'rb') as handle:
                self.z = pickle.load(handle)
            if len(self.z) != len(self.dirs_all):
                raise ValueError(f"ERROR: invalid self.z ({len(self.z)}, {len(self.dirs_all)})")
            print(f"INFO: loaded {fname_z}")

        elif os.path.isfile(fname_z_raw):  # only apply raw2ML
            with open(fname_z_raw, 'rb') as handle:
                self.z = pickle.load(handle)
            if len(self.z) != len(self.dirs_all):
                raise ValueError(f"ERROR: invalid self.z ({len(self.z)}, {len(self.dirs_all)})")
            print(f"INFO: loading {fname_z} failed ... loaded {fname_z_raw} instead ...")

            for i in range(len(self.z)):
                n_frames_nonzero = int(self.z[i][0][1].item())
                if self.type_z == "agg":
                    z_agg = self.z[i][1][:n_frames_nonzero]
                    z_sph = None
                    _, z = raw2ML(self.cfg, self.type_z, self.hyper, x=None, z=z_agg, inverse=False)

                n_frames_nonzero = z.shape[0]
                n_rows_nonzero = z.shape[1]
                if n_frames_nonzero < n_frames_nonzero_max:
                    z = torch.cat((z, -666 * torch.ones((n_frames_nonzero_max - z.shape[0], z.shape[1], z.shape[2]))),
                                  dim=0)  # pad with -666
                self.z[i][1] = z

            with open(fname_z, 'wb') as handle:
                pickle.dump(self.z, handle)
            print(f"INFO: saved {fname_z}")

        else:
            print(f"INFO: loading {fname_z} failed ... processing data ...")
            for i, dir in enumerate(self.dirs_all):
                n_frames = self.info_all["n_frames"][i]
                result = np.load("{}/results.npy".format(dir), allow_pickle=True).item()
                z_agg = torch.zeros((0, 1, n_z_agg))  # time series (CAUTION: last frame is generally non-equidistant)
                # load all frames:
                for n in range(1, n_frames + 1):
                    if n % frame_interval == 0 or n == n_frames:

                        if self.type_z == "agg":
                            z_n_agg = self.get_z_n_agg(dir, n)
                            z_agg = torch.cat((z_agg, z_n_agg.unsqueeze(0)), dim=0)  # z.shape: [time, rows, features]

                # preprocess all frames:
                _, z = raw2ML(self.cfg, self.type_z, self.hyper, x=None, z=z_agg, inverse=False)

                s, device = z.shape, z.device
                n_frames_nonzero = s[0]
                n_z = s[2]
                if n_frames_nonzero < n_frames_nonzero_max:
                    z = torch.cat((z,
                                   -666 * torch.ones((n_frames_nonzero_max - n_frames_nonzero, z.shape[1], z.shape[2]),
                                                     device=device)), dim=0)  # pad with -666
                elif n_frames_nonzero > n_frames_nonzero_max:
                    z = z[:n_frames_nonzero_max]  # cut
                    n_frames_nonzero = n_frames_nonzero_max
                if z.shape[0] != n_frames_nonzero_max:
                    raise ValueError()
                setup = np.load("{}/setup.npy".format(dir), allow_pickle=True).item()
                if torch.isnan(z.mean()):
                    raise ValueError()
                info = torch.tensor([n_frames, n_frames_nonzero, 0, setup["impact_angle_touching_ball"],
                                     setup["vel_vesc_touching_ball"]])
                self.z.append([info, z.cpu()])

            with open(fname_z, 'wb') as handle:
                pickle.dump(self.z, handle)
            print(f"INFO: saved {fname_z}")

    def init_samples(self, split, fold, verbose=True):
        """
        + split_method == 0: classic random, but fixed 80/10/10 splits
        + split_method == 1: crossvalidation. ood test split
        + split_method == 2: crossvalidation. iid test split
        + split_method == 3: train on entire development set (train + vali), ood test split
        """

        self.split = split
        self.fold = fold
        i_fold, folds = [], self.cfg["ML"]["training"]["folds"]

        # select samples:
        if self.split == "trainvalitest":
            i_fold = list(range(0, self.n_samples_all))
        else:
            split_method = self.cfg["ML"]["training"]["split_method"]
            if split_method == 0:
                n_samples_test = int(0.1 * self.n_samples_all)
                n_samples_vali = int(0.1 * self.n_samples_all)
                n_samples_train = self.n_samples_all - n_samples_vali - n_samples_test
                n_samples_trainvali = n_samples_train + n_samples_vali
                if self.split == "train":
                    i_fold = list(range(0, n_samples_train))
                elif self.split == "vali":
                    i_fold = list(range(n_samples_train, n_samples_trainvali))
                elif self.split == "test":
                    i_fold = list(range(n_samples_trainvali, self.n_samples_all))

            elif split_method == 1 or split_method == 2 or split_method == 3:
                try:
                    i_test = self.i_test
                    i_trainvali = self.i_trainvali
                except:
                    if verbose: print("INFO: init self.i_test and self.i_trainvali")
                    if split_method == 1 or split_method == 3:
                        i_all = list(range(0, self.n_samples_all))
                        # find indices within test regions:
                        i_trainvali = []
                        i_test = []
                        test_regions = [{"v_min": 1.5, "v_max": 2.5, "a_min": 10., "a_max": 30.},
                                        {"v_min": 2., "v_max": 4., "a_min": 65., "a_max": 75.},
                                        {"v_min": 1., "v_max": 2., "a_min": 80., "a_max": 90.},
                                        {"v_min": 6., "v_max": 8., "a_min": 0.,
                                         "a_max": 20.}]  # impact velocity, impact angle
                        for i in i_all:
                            v, a = self.info_all["vel_vesc_touching_ball"][i], \
                                   self.info_all["impact_angle_touching_ball"][i]
                            test = False
                            for r in range(len(test_regions)):
                                if test_regions[r]["v_min"] < v < test_regions[r]["v_max"] and test_regions[r]["a_min"] < a < test_regions[r]["a_max"]:
                                    test = True
                            if test:
                                i_test.append(i)
                            else:
                                i_trainvali.append(i)
                    elif split_method == 2:
                        n_samples_test = int(0.1 * self.n_samples_all)
                        n_samples_trainvali = self.n_samples_all - n_samples_test
                        i_trainvali = list(range(0, n_samples_trainvali))
                        i_test = list(range(n_samples_trainvali, self.n_samples_all))
                    self.i_test = i_test
                    self.i_trainvali = i_trainvali
                # prepare fold:
                if split_method == 3:
                    if folds != 1:
                        raise ValueError("invalid number of folds")
                    if self.split == "train" or self.split == "vali":
                        i_fold = self.i_trainvali
                    elif self.split == "test":
                        i_fold = i_test
                else:
                    n_samples_test = len(i_test)
                    n_samples_vali = len(i_test)
                    n_samples_train = self.n_samples_all - n_samples_vali - n_samples_test
                    n_samples_trainvali = len(i_trainvali)
                    if n_samples_trainvali != n_samples_train + n_samples_vali:
                        raise ValueError()
                    if folds == 1:
                        if self.split == "train":
                            i_fold = []
                            for i in range(0, n_samples_train):
                                i_fold.append(i_trainvali[i])
                        elif self.split == "vali":
                            i_fold = []
                            for i in range(n_samples_train, n_samples_trainvali):
                                i_fold.append(i_trainvali[i])
                        elif self.split == "test":
                            i_fold = i_test
                    else:
                        n_fold = int(n_samples_trainvali / folds)
                        if self.split == "train":
                            for i in range(n_samples_trainvali):
                                if not (self.fold * n_fold <= i < (self.fold + 1) * n_fold):
                                    i_fold.append(i_trainvali[i])
                        elif self.split == "vali":
                            for i in range(n_samples_trainvali):
                                if self.fold * n_fold <= i < (self.fold + 1) * n_fold:
                                    i_fold.append(i_trainvali[i])
                        elif self.split == "test":
                            i_fold = i_test

        # (optional) use less training data:
        a = self.cfg["ML"]["training"]["training_data_amount"]
        if self.split == "train" and a != 1:
            print(f"WARNING: using training_data_amount = {a}")
            i_fold_ = []
            for i in range(int(a * len(i_fold))):
                i_fold_.append(i_fold[i])
            i_fold = i_fold_

        # init samples for current split and fold:
        self.index2i = []
        for i in i_fold:
            self.index2i.append(i)  # note: i can point to any dir in self.dirs_all, whereas index operates only within current split and fold
        self.n_samples = len(self.index2i)
        if verbose:
            print(f"INFO: init samples finished (split: {self.split}, fold: {self.fold}, n_samples: {self.n_samples})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        i = self.index2i[index]
        return [self.x[i], self.z[i]]

class Dataset_SPH_timpe(data.Dataset):
    def __init__(self, cfg):
        print(f"INFO: init dataset timpe")
        self.cfg = cfg
        self.hyper = {'mu': {'N': 0.,
                             'mass_agg': 0.,
                             'gamma': 0.,
                             'fraction_mat': 0.,
                             'spin': 0.,
                             'pos_vel': [0., 0., 0., 0., 0., 0.]},
                      # no shift in pos and vel --> barycenter remains at [0,0,0,0,0,0]
                      'std': {'N': 230000.,
                              'mass_agg': 100.,
                              'gamma': 1.,
                              'fraction_mat': 1.,
                              'spin': 100.,
                              'pos_vel': [6., 41., 3., 2., 16., 0.5]}
                      }
        self.get_all()
        self.get_x()
        self.get_z()
        self.init_samples("train", 0)
        print("INFO: init dataset finished")

    def get_all(self):
        self.type_x = "setup"
        self.type_z = "agg"
        if self.type_x != "setup" or self.type_z != "agg":
            raise NotImplementedError()
        fname_trainvali = "{}misc/i_trainvali_timpe.pickle".format(self.cfg["SPH"]["system"]["path_Astro_SPH"])
        fname_test = "{}misc/i_test_timpe.pickle".format(self.cfg["SPH"]["system"]["path_Astro_SPH"])
        try:
            with open(fname_trainvali, "rb") as f:
                self.i_trainvali = pickle.load(f)
            with open(fname_test, "rb") as f:
                self.i_test = pickle.load(f)
            print(f"INFO: loaded {fname_trainvali}")
            print(f"INFO: loaded {fname_test}")
        except:
            self.dirs_trainvali = utils.system.get_subfolders("{}lhs10k/".format(self.cfg["ML"]["dataset"]["timpe"]["path"]))
            self.dirs_test = utils.system.get_subfolders("{}lhs500/".format(self.cfg["ML"]["dataset"]["timpe"]["path"]))
            self.dirs_trainvali.sort()
            self.dirs_test.sort()
            self.n_samples_trainvali = len(self.dirs_trainvali)
            self.n_samples_test = len(self.dirs_test)
            self.i_trainvali = list(range(0, self.n_samples_trainvali))
            self.i_test = list(range(0, self.n_samples_test))
            with open(fname_trainvali, "wb") as f:
                pickle.dump(self.i_trainvali, f)
            with open(fname_test, "wb") as f:
                pickle.dump(self.i_test, f)
            print(f"INFO: saved {fname_trainvali}")
            print(f"INFO: saved {fname_test}")

    def get_x_i(self, dir_):
        x = torch.zeros(25)  # [n_x]
        data = pynbody.load("{}/collision.std".format(dir_))  # ['grp' : remnant idx, 'eps' : sml length? useless, 'vel', 'mass', 'phi': potential engergy of particle, 'pos']
        d = {}
        for key in ["pos", "vel", "mass", "metals", "grp"]:
            d[key] = torch.from_numpy(np.array(data[key])).float()
        sph = torch.cat((d["pos"][:, :], d["vel"][:, :], d["mass"][:].unsqueeze(-1), d["metals"][:].unsqueeze(-1)), dim=-1)
        idx_proj = data["grp"] == 1  # projectile
        idx_targ = data["grp"] == 2  # target
        sph_proj = sph[idx_proj]
        sph_targ = sph[idx_targ]
        m_proj = sph_proj[:, 6].sum()
        m_targ = sph_targ[:, 6].sum()
        x[0] = len(idx_proj) + len(idx_targ)
        x[1] = m_proj + m_targ
        x[2] = m_proj / m_targ
        idx_proj_mat0 = sph_proj[:, 7] == 2.  # core proj (iron)
        # idx_proj_mat1 = sph_proj[:,7] == 1. # mantle proj (granite)
        x[3] = len(idx_proj_mat0) / len(idx_proj)  # core proj
        x[4] = 0.  # shell proj
        idx_targ_mat0 = sph_targ[:, 7] == 2.  # core targ (iron)
        x[5] = len(idx_targ_mat0) / len(idx_targ)  # core targ
        x[6] = 0.  # shell targ
        S_proj = (sph_proj[:, :6] * sph_proj[:, 6].unsqueeze(-1)).sum(dim=0) / m_proj  # barycenter
        S_targ = (sph_targ[:, :6] * sph_targ[:, 6].unsqueeze(-1)).sum(dim=0) / m_targ  # barycenter
        ang_proj = utils.sim.angular_momentum(sph_proj[:, :7].numpy(), S_proj.numpy())
        ang_targ = utils.sim.angular_momentum(sph_targ[:, :7].numpy(), S_targ.numpy())
        x[7:7 + 3] = torch.from_numpy(ang_proj["vector"])
        x[10:10 + 3] = torch.from_numpy(ang_targ["vector"])
        x[13:13 + 6] = S_proj
        x[19:19 + 6] = S_targ
        return x

    def get_x(self):
        fname_x_trainvali = "{}misc/x_{}_trainvali_timpe.pt".format(self.cfg["SPH"]["system"]["path_Astro_SPH"], self.type_x)
        fname_x_test = "{}misc/x_{}_test_timpe.pt".format(self.cfg["SPH"]["system"]["path_Astro_SPH"], self.type_x)
        try:
            self.x_trainvali = torch.load(fname_x_trainvali)
            self.x_test = torch.load(fname_x_test)
            print(f"INFO: loaded {fname_x_trainvali}")
            print(f"INFO: loaded {fname_x_test}")
        except Exception as e:
            print(e)
            self.x_trainvali = torch.zeros((0, 25))  # [b,n_x]
            for i, dir_ in enumerate(self.dirs_trainvali):
                x = self.get_x_i(dir_)
                self.x_trainvali = torch.cat((self.x_trainvali, x.unsqueeze(0)), dim=0)
            self.x_test = torch.zeros((0, 25))  # [b,n_x]
            for i, dir_ in enumerate(self.dirs_test):
                x = self.get_x_i(dir_)
                self.x_test = torch.cat((self.x_test, x.unsqueeze(0)), dim=0)

            self.x_trainvali = self.x_trainvali.unsqueeze(1) # [b,1,n_x]
            self.x_test = self.x_test.unsqueeze(1)  # [b,1,n_x]
            self.x_trainvali, _ = raw2ML(self.cfg, self.type_x, self.hyper, x=self.x_trainvali, z=None, inverse=False)
            self.x_test, _ = raw2ML(self.cfg, self.type_x, self.hyper, x=self.x_test, z=None, inverse=False)

            torch.save(self.x_trainvali.cpu(), fname_x_trainvali)
            torch.save(self.x_test.cpu(), fname_x_test)
            print(f"INFO: saved {fname_x_trainvali}")
            print(f"INFO: saved {fname_x_test}")

    def get_z_i(self, dir_):
        param = {}
        with open("{}/collision.param".format(dir_)) as f:
            for line in f:
                if line[0] != "#":
                    (key, _, val) = line.split()
                    param[key] = val
        n_frames = param["nSteps"] # simulated steps (each step with time interval param["dDelta"])
        info = torch.tensor([int(n_frames)])

        z = torch.zeros(27)  # [n_z]
        data = pynbody.load("{}/outcome.std".format(dir_))
        d = {}
        for key in ["pos", "vel", "mass", "metals", "grp"]:
            d[key] = torch.from_numpy(np.array(data[key])).float()
        sph = torch.cat((d["pos"][:, :], d["vel"][:, :], d["mass"][:].unsqueeze(-1), d["metals"][:].unsqueeze(-1)), dim=-1)

        idx_lr = data["grp"] == 1  # largest remnant
        idx_2lr = data["grp"] == 2  # 2nd-largest remnant
        idx_rest = data["grp"] == 0  # rest
        sph_lr = sph[idx_lr]
        sph_2lr = sph[idx_2lr]
        sph_rest = sph[idx_rest]

        z[0] = sph_lr[:, 6].sum()
        z[3] = sph_2lr[:, 6].sum()
        z[6] = sph_rest[:, 6].sum()
        idx_mat0 = sph_lr[:, 7] == 2.
        z[1] = len(idx_mat0) / len(idx_lr)  # mat0 (iron) fraction
        z[2] = 0.
        idx_mat0 = sph_2lr[:, 7] == 2.
        z[4] = len(idx_mat0) / len(idx_2lr)  # mat0 (iron) fraction
        z[5] = 0.
        idx_mat0 = sph_rest[:, 7] == 2.
        z[7] = len(idx_mat0) / len(idx_rest)  # mat0 (iron) fraction
        z[8] = 0.
        if sph_lr.shape[0] > 0:
            z[9:9 + 6] = (sph_lr[:, :6] * sph_lr[:, 6].unsqueeze(-1)).sum(dim=0) / z[0]  # barycenter
        if sph_2lr.shape[0] > 0:
            z[15:15 + 6] = (sph_2lr[:, :6] * sph_2lr[:, 6].unsqueeze(-1)).sum(dim=0) / z[3]  # barycenter
        if sph_rest.shape[0] > 0:
            z[21:21 + 6] = (sph_rest[:, :6] * sph_rest[:, 6].unsqueeze(-1)).sum(dim=0) / z[6]  # barycenter

        z = z.unsqueeze(0).unsqueeze(0)
        _, z = raw2ML(self.cfg, self.type_z, self.hyper, x=None, z=z, inverse=False)
        z = z.squeeze(0).squeeze(0)
        return info, z  # tensor: n_frames, tensor: z features

    def get_z(self):
        self.keys_param = ["dSoft", "bPeriodic", "bParaRead", "bParaWrite", "dTheta", "nReplicas", "achInFile",
                           "achOutName", "dConstAlpha", "dConstBeta", "nSmooth", "iCheckInterval", "bRestart",
                           "dExtraStore", "iMaxRung", "dEta", "dEtaCourant", "bStandard", "bDoGravity", "bDoGas",
                           "bFastGas", "bComove", "bGasCondensed", "dKpcUnit", "dMsolUnit", "dVelocityDamper",
                           "dhMax", "dMeanMolWeight", "nSteps", "dDelta", "iOutInterval", "iLogInterval",
                           "bOverwrite", "bVDetails", "bVRungStat", "bViscosityLimiter", "bViscosityLimitdt"]

        fname_z_trainvali = "{}misc/z_{}_trainvali_timpe.pickle".format(self.cfg["SPH"]["system"]["path_Astro_SPH"], self.type_z)
        fname_z_test = "{}misc/z_{}_test_timpe.pickle".format(self.cfg["SPH"]["system"]["path_Astro_SPH"], self.type_z)
        try:
            with open(fname_z_trainvali, 'rb') as handle:
                self.z_trainvali = pickle.load(handle)
            with open(fname_z_test, 'rb') as handle:
                self.z_test = pickle.load(handle)
            print(f"INFO: loaded {fname_z_trainvali}")
            print(f"INFO: loaded {fname_z_test}")
        except Exception as e:
            print(e)
            self.z_trainvali = []
            for i, dir_ in enumerate(self.dirs_trainvali):
                info, z = self.get_z_i(dir_)
                self.z_trainvali.append([info, z])
            self.z_test = []
            for i, dir_ in enumerate(self.dirs_test):
                info, z = self.get_z_i(dir_)
                self.z_test.append([info, z])

            with open(fname_z_trainvali, 'wb') as handle:
                pickle.dump(self.z_trainvali, handle)
            with open(fname_z_test, 'wb') as handle:
                pickle.dump(self.z_test, handle)
            print(f"INFO: saved {fname_z_trainvali}")
            print(f"INFO: saved {fname_z_test}")

    def init_samples(self, split, fold, verbose=True):
        self.split = split
        self.fold = fold
        folds = self.cfg["ML"]["training"]["folds"]

        # select samples:
        if self.cfg["ML"]["training"]["split_method"] != 1:
            raise NotImplementedError("ERROR: split_method = 1 required")

        # prepare fold:
        if folds < 2:
            raise NotImplementedError()
        n_samples_test = len(self.i_test)
        n_samples_trainvali = len(self.i_trainvali)
        n_samples_vali = int(n_samples_trainvali / folds)
        n_samples_train = n_samples_trainvali - n_samples_vali

        n_fold = int(n_samples_trainvali / folds)
        i_fold = []
        if self.split == "train":
            for i in range(n_samples_trainvali):
                if not (self.fold * n_fold <= i < (self.fold + 1) * n_fold):
                    i_fold.append(self.i_trainvali[i])
        elif self.split == "vali":
            for i in range(n_samples_trainvali):
                if self.fold * n_fold <= i < (self.fold + 1) * n_fold:
                    i_fold.append(self.i_trainvali[i])
        elif self.split == "test":
            i_fold = self.i_test

        # use less training data:
        a = self.cfg["ML"]["training"]["training_data_amount"]
        if self.split == "train" and a != 1:
            print(f"WARNING: using training_data_amount = {a}")
            i_fold_ = []
            for i in range(int(a * len(i_fold))):
                i_fold_.append(i_fold[i])
            i_fold = i_fold_

        # init samples for current split and fold:
        self.index2i = []
        for i in i_fold:
            self.index2i.append(i)  # note: i can point to any datapoint depending on split, whereas index operates only within current split and fold
        self.n_samples = len(self.index2i)
        if verbose: print(f"INFO: init samples finished (split: {self.split}, fold: {self.fold}, n_samples: {self.n_samples})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        i = self.index2i[index]
        if self.split == "test":
            return [self.x_test[i], self.z_test[i]]
        else:
            return [self.x_trainvali[i], self.z_trainvali[i]]

@torch.no_grad()
def preprocess(cfg, xz, mode):

    # to device:
    x = xz[0].cuda()  # inputs
    z = [xz[1][0].cuda(), xz[1][1].cuda()]  # labels. [info, z_agg]

    bs, device = z[1].shape[0], z[1].device

    if cfg["ML"]["dataset"]["name"] == "winter":

        type_z = cfg["ML"]["dataset"]["winter"]["targets"]
        if cfg["ML"]["dataset"]["winter"]["augmentation"]:
            # init random rotation:
            bs = x.shape[0]
            q = torch.zeros((bs, 4), device=device)
            euler = 2. * np.pi * torch.rand((bs, 3), device=device)  # yaw, pitch, roll
            q = utils.ML.euler_to_quaternion(euler)
            # select colums to be rotated:
            type_x = cfg["ML"]["dataset"]["winter"]["inputs"]
            if type_x == "setup":
                cols_x = [[7, 7+6], [13, 13+6], [19, 19+6]]
            if type_z == "agg":
                cols_z = [[9, 9+6], [15, 15+6], [21, 21+6]]
            # apply rotation:
            for c in cols_x:
                x[:,:,c[0]:c[1]] = utils.ML.quaternion_rotate(x[:,:,c[0]:c[1]], q)  # input shape: [b,n,k*3], [b,4]
            for c in cols_z:
                z[1][:,:,:,c[0]:c[1]] = utils.ML.quaternion_rotate(z[1][:,:,:,c[0]:c[1]].squeeze(2), q).unsqueeze(2)  # input shape: [b,t,1,k*3], [b,4]

        if type_z == "agg":
            z.append(z[1][:,:,0,:])  # ground truth sequence before reshaping. shape: [b,t,n_z]

        # flatten labels to avoid dummy values:
        z_combine = torch.zeros((1, 0, z[1].shape[2], z[1].shape[3]), device=device)
        if mode == "all_frames":
            # combine all samples in non-padded (in time) representation
            for b in range(bs):
                n_frames_nonzero = int(z[0][b, 1])
                z_combine = torch.cat((z_combine, z[1][b,:n_frames_nonzero,:,:].unsqueeze(0)), dim=1)
            z[1] = z_combine.clone()  # [1,T,n_nodes,n_z]
        elif mode == "last_frame":
            # combine all samples (last frame only)
            for b in range(bs):
                n_frames_nonzero = int(z[0][b, 1])
                z_combine = torch.cat((z_combine, z[1][b, n_frames_nonzero-1:n_frames_nonzero, :, :].unsqueeze(0)), dim=1)
            z[1] = z_combine.clone()  # [1,b,n_nodes,n_z]
        elif mode == "inference":
            z = None

    elif cfg["ML"]["dataset"]["name"] == "timpe":

        z[0] = torch.div(z[0], 1000, rounding_mode='floor')
        z[1] = z[1].unsqueeze(0).unsqueeze(2)  # [1,b,1,n_z]

    return x, z

@torch.no_grad()
def raw2ML(cfg, typ, hyper, x=None, z=None, inverse=False):
    """
    transform raw values to ML-friendly values
    inverse == False: raw --> ML
    inverse == True: ML --> raw
    """
    def normalize(x, mu, std, inverse):
        if inverse:  # ML --> raw
            x = std * x + mu
        else:  # raw --> ML
            x = (x - mu) / std
        return x

    if x is not None:
        if x.dim() != 3:  # [b,*,n_x]
            raise ValueError(x.shape)
        if typ == "setup":
            x[:, :, 0] = normalize(x[:, :, 0], hyper["mu"]["N"], hyper["std"]["N"], inverse)
            x[:, :, 1] = normalize(x[:, :, 1], hyper["mu"]["mass_agg"], hyper["std"]["mass_agg"], inverse)
            x[:, :, 2] = normalize(x[:, :, 2], hyper["mu"]["gamma"], hyper["std"]["gamma"], inverse)
            x[:, :, 3:3+4] = normalize(x[:, :, 3:3+4], hyper["mu"]["fraction_mat"], hyper["std"]["fraction_mat"], inverse)
            for dim in range(3):  # sx,sy,sz
                x[:, :, dim + 7] = normalize(x[:, :, dim + 7], hyper["mu"]["spin"], hyper["std"]["spin"], inverse)  # proj
                x[:, :, dim + 10] = normalize(x[:, :, dim + 10], hyper["mu"]["spin"], hyper["std"]["spin"], inverse)  # targ
            for dim in range(6):  # x,y,z,vx,vy,vz
                x[:, :, dim + 13] = normalize(x[:, :, dim + 13], hyper["mu"]["pos_vel"][dim], hyper["std"]["pos_vel"][dim], inverse)  # proj
                x[:, :, dim + 19] = normalize(x[:, :, dim + 19], hyper["mu"]["pos_vel"][dim], hyper["std"]["pos_vel"][dim], inverse)  # targ
        elif typ == "sph":
            raise NotImplementedError()

    if z is not None:
        if z.dim() != 3:  # [t,*,n_z]
            raise ValueError(z.shape)
        if typ == "agg":
            z[:, :, 0] = normalize(z[:, :, 0], hyper["mu"]["mass_agg"], hyper["std"]["mass_agg"], inverse)
            z[:, :, 3] = normalize(z[:, :, 3], hyper["mu"]["mass_agg"], hyper["std"]["mass_agg"], inverse)
            z[:, :, 6] = normalize(z[:, :, 6], hyper["mu"]["mass_agg"], hyper["std"]["mass_agg"], inverse)
            z[:, :, 1] = normalize(z[:, :, 1], hyper["mu"]["fraction_mat"], hyper["std"]["fraction_mat"], inverse)
            z[:, :, 2] = normalize(z[:, :, 2], hyper["mu"]["fraction_mat"], hyper["std"]["fraction_mat"], inverse)
            z[:, :, 4] = normalize(z[:, :, 4], hyper["mu"]["fraction_mat"], hyper["std"]["fraction_mat"], inverse)
            z[:, :, 5] = normalize(z[:, :, 5], hyper["mu"]["fraction_mat"], hyper["std"]["fraction_mat"], inverse)
            z[:, :, 7] = normalize(z[:, :, 7], hyper["mu"]["fraction_mat"], hyper["std"]["fraction_mat"], inverse)
            z[:, :, 8] = normalize(z[:, :, 8], hyper["mu"]["fraction_mat"], hyper["std"]["fraction_mat"], inverse)
            for dim in range(6):  # x,y,z,vx,vy,vz
                z[:, :, dim + 9] = normalize(z[:, :, dim + 9], hyper["mu"]["pos_vel"][dim], hyper["std"]["pos_vel"][dim], inverse)  # largest
                z[:, :, dim + 15] = normalize(z[:, :, dim + 15], hyper["mu"]["pos_vel"][dim], hyper["std"]["pos_vel"][dim], inverse)  # 2nd largest
                z[:, :, dim + 21] = normalize(z[:, :, dim + 21], hyper["mu"]["pos_vel"][dim], hyper["std"]["pos_vel"][dim], inverse)  # rest
        else:
            raise NotImplementedError()
        
    return x, z

def get_dataset(cfg):
    if cfg["ML"]["dataset"]["name"] == "winter":
        dataset = {"data": Dataset_SPH_winter(cfg), "params": {'batch_size': cfg["ML"]["training"]["bs"],
                                                        'shuffle': cfg["ML"]["dataset"]["shuffle"],
                                                        'num_workers': cfg["ML"]["dataset"]["num_workers"],
                                                        'drop_last': False}}
    elif cfg["ML"]["dataset"]["name"] == "timpe":
        if cfg["ML"]["training"]["mode"] != "last_frame":
            raise NotImplementedError()
        dataset = {"data": Dataset_SPH_timpe(cfg), "params": {'batch_size': cfg["ML"]["training"]["bs"],
                                                        'shuffle': cfg["ML"]["dataset"]["shuffle"],
                                                        'num_workers': cfg["ML"]["dataset"]["num_workers"],
                                                        'drop_last': False}}
    else:
        raise NotImplementedError()

    return dataset

if __name__ == '__main__':
    print('Done.')