import os
import sys
import torch
import shutil
import subprocess
import matplotlib
import socket
import yaml
if socket.gethostname() != "philip-dell": matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import random
import time
import json
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

class system:

    def prepare_cfg(args):
        with open(args.config) as f:
            cfg = yaml.load(f)
        if socket.gethostname() == "philip-dell": x = "local"
        else: x = "server"

        cfg["system"]["config"] = args.config
        cfg["system"]["machine"] = x
        cfg["system"]["path_project"] = cfg["system"][x]["path_project"]
        cfg["system"]["config_abs"] = "{}code{}".format(cfg["system"]["path_project"], cfg["system"]["config"].replace("./", "/"))
        cfg["system"]["gpu_arch"] = cfg["system"][x]["gpu_arch"]
        cfg["SPH"]["system"]["path_Astro_SPH"] = cfg["SPH"]["system"][x]["path_Astro_SPH"]
        cfg["SPH"]["system"]["path_spheres_ini"] = "{}code/SPH/spheres_ini/".format(cfg["system"]["path_project"])
        cfg["SPH"]["system"]["path_miluphcuda"] = "{}code/SPH/miluphcuda/".format(cfg["system"]["path_project"])
        cfg["ML"]["dataset"]["info"] = cfg["ML"]["dataset"]["{}".format(cfg["ML"]["dataset"]["name"])]
        cfg["ML"]["dataset"]["info"]["path"] = cfg["ML"]["dataset"]["info"]["path_{}".format(x)]
        cfg["ML"]["dataset"]["timpe"]["path"] = cfg["ML"]["dataset"]["timpe"]["path_{}".format(x)]
        if cfg["ML"]["dataset"]["name"] == "winter":
            cfg["ML"]["dataset"]["info"]["n_x"] = cfg["ML"]["dataset"]["winter"]["n_x_{}".format(cfg["ML"]["dataset"]["winter"]["inputs"])]
            cfg["ML"]["dataset"]["info"]["n_z"] = cfg["ML"]["dataset"]["winter"]["n_z_{}".format(cfg["ML"]["dataset"]["winter"]["targets"])]
        cfg["ML"]["dataset"]["analysis"]["path"] = cfg["ML"]["dataset"]["analysis"]["path_{}".format(x)]
        cfg["ML"]["system"]["dir_check_all"] = cfg["ML"]["system"]["dir_check_{}".format(x)]
        cfg["ML"]["system"]["dir_check"] = "{}run{:03d}/".format(cfg["ML"]["system"]["dir_check_{}".format(x)], cfg["ML"]["system"]["run"])
        cfg["ML"]["system"]["dir_analyze_all"] = cfg["ML"]["system"]["dir_analyze_{}".format(x)]
        cfg["ML"]["system"]["dir_analyze"] = cfg["ML"]["system"]["dir_analyze_{}".format(x)]
        return cfg
    
    def number_of_files_in(dir_, identifier):
        return len([name for name in os.listdir(dir_) if identifier in name and os.path.isfile(os.path.join(dir_, name))])
    
    def check_mem(dir_, min_GB=10):
        # check disk usage of given dir_
        total, used, free = shutil.disk_usage(dir_)
        total, used, free = total // (2**30), used // (2**30), free // (2**30) # in GB
        if free < min_GB: raise ValueError('ERROR: not enough disk space available on {}: total: {}, used: {}, free: {} [GB]'.format(dir_, total, used, free))

    def execute(command, dic=None):
        print("INFO: command: {}".format(command))
        t0 = time.time()
        e = subprocess.call(command, shell=True)
        t1 = time.time()
        if dic is not None:
            if "commands" in dic: dic["commands"].append(command)
            else: dic["commands"] = [command] # init
        if e != 0:
            print("ERROR: {}".format(e))
            try: dic["error"] = e
            except: dic = {"error" : e}
        return dic, t1-t0
    
    def get_subfolders(path): return [f.path for f in os.scandir(path) if f.is_dir()]
    
class sim:  # version 1 (functions related to SPH simulations)
    
    def rand_min_max(min_, max_): return min_ + np.random.random() * (max_ - min_)
    def rand_int_min_max(min_, max_): return np.random.randint(min_, max_+1)
    gravitational_constant = 6.674e-11 # SI
    
    def get_ID(cfg):
        if cfg["SPH"]["sim"]["setup_type"] == "custom": ext = "c"
        elif cfg["SPH"]["sim"]["setup_type"] == "random": ext = "r"
        while True:
            ID = "{}{:015d}".format(ext, random.randint(0, int(1e15))) # 15-digit code
            path_sim = "{}winter/{}/".format(cfg["SPH"]["system"]["path_Astro_SPH"], ID)
            if not os.path.exists(path_sim): break
            else: print("WARNING: {} already exists --> sampling new ID".format(path_sim))
        return ID

    def get_valid(setup, result, i, dir):
        valid = False
        err = os.path.getsize("{}/err.txt".format(dir)) > 0
        if not err:
            if not "error" in setup.keys() and not "error" in result.keys():
                if setup["ID"].startswith("r"):
                    if "valid" in result.keys():
                        if result["valid"]:
                            valid = True
                        else:
                            print(f"INFO: get_valid (dir #{i}, ID: {setup['ID']}): result['valid'] = False")
                    else:
                        print(f"INFO: get_valid (dir #{i}, ID: {setup['ID']}): 'valid' not in result.keys")
                else:
                    print(f"INFO: get_valid (dir #{i}, ID: {setup['ID']}): setup['ID'] does not start with 'r'")
            else:
                print(f"INFO: get_valid (dir #{i}, ID: {setup['ID']}): error in setup.keys or result.keys")
        else:
            print(f"INFO: get_valid (dir #{i}, ID: {setup['ID']}): err")
        return valid
    
    class NumpyArrayEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    
    def save_dic(dic, fname):
        '''save given dictionary to .txt, .npy, and .json'''
        with open(fname + ".txt", 'w') as f:
            for key, val in dic.items(): f.write("{} : {}\n".format(key, val))
        f.close()
        np.save("{}".format(fname + ".npy"), dic)
        print("INFO: saved {}".format(fname))
        return
    
    def save_frag_large(setup, frag_large, frame):
        f_frag_large = "{}fragments_large.{:04d}".format(setup["path_sim"], frame)
        if frame == 0:
            if frag_large.shape[1] != 7: raise ValueError(frag_large.shape)
            header = "target and projectile (SPH input frame)\nx1, x2, x3, v1, v2, v3, mass"
        else:
            if setup["n_mat"] == 2:
                if frag_large.shape[1] != 10: raise ValueError(frag_large.shape)
                header = "largest fragments (containing >= {} SPH particles), frame {} (SPH output frame)\nx1, x2, x3, v1, v2, v3, mass, rel_mass, mat0_frac, mat1_frac".format(setup["n_sph_min"], frame)
            else:
                if frag_large.shape[1] != 11: raise ValueError(frag_large.shape)
                header = "largest fragments (containing >= {} SPH particles), frame {} (SPH output frame)\nx1, x2, x3, v1, v2, v3, mass, rel_mass, mat0_frac, mat1_frac, mat2_frac".format(setup["n_sph_min"], frame)
        np.savetxt(f_frag_large, frag_large, header=header)
        np.save("{}.npy".format(f_frag_large), frag_large)
        with open("{}.json".format(f_frag_large), 'w') as json_file: json.dump(frag_large, json_file, cls=sim.NumpyArrayEncoder)
        print("INFO: saved {}".format(f_frag_large))
        return
    
    def save_frame(cfg, setup, results, frame):
        """save all relevant data for given frame"""
        f_a = "{}impact.{:04d}".format(setup["path_sim"], frame)
        d = np.loadtxt(f_a)
        subsampling = cfg["SPH"]["sim"]["{}_setup".format(cfg["SPH"]["sim"]["setup_type"])]["subsampling"]
        if frame == 0:
            if d.shape[1] != 9: raise ValueError(d.shape)
            header = "raw output, subsampled with factor {}\nx1, x2, x3, v1, v2, v3, mass, energy, material type".format(subsampling)
        else:
            if d.shape[1] != 13: raise ValueError(d.shape)
            header = "raw output, subsampled with factor {}\nx1, x2, x3, v1, v2, v3, mass, density, energy, smoothing length, number of interaction partners, material type, pressure".format(subsampling)
        np.savetxt(f_a, d[::subsampling,:], header=header)
        # saving with identical format as sph raw frames:
        """
        f = open(f_a, "w")
        for i in range(d[::subsampling,:].shape[0]):
            str = ""
            for c in range(d[::subsampling,:].shape[1]):
                if c == 10 or c == 11:
                    value = int(d[i,c])
                elif c < 6:
                    if int(np.sign(d[i,c])) == 1:
                        value = '+{:.15e}'.format(d[i,c])
                    else:
                        value = '{:.15e}'.format(d[i, c])
                else:
                    value = '{:.6e}'.format(d[i, c])
                str += "{}\t".format(value)
            str = str[:-2]
            str += "\n"
            f.write(str)
        f.close()
        """
        print("INFO: saved {}".format(f_a)) # overwrite
        if frame == 0:
            fnames = ["impact.{:04d}".format(frame),
                      "fragments_large.{:04d}".format(frame),
                      "fragments_large.{:04d}.npy".format(frame),
                      "fragments_large.{:04d}.json".format(frame),
                      "fragments_vis.{:04d}.png".format(frame)]
        else:
            fnames = ["impact.{:04d}".format(frame),
                      "impact.{:04d}.info".format(frame),
                      "fragments_large.{:04d}".format(frame),
                      "fragments_large.{:04d}.npy".format(frame),
                      "fragments_large.{:04d}.json".format(frame),
                      
                      "aggregates.{:04d}".format(frame)]
            if results["frame_{}".format(frame)]["n_frag_large"] != 0: fnames.append("fragments_vis.{:04d}.png".format(frame))
        for fname in fnames:
            f_a = "{}{}".format(setup["path_sim"], fname)
            f_b = "{}frames/".format(setup["path_sim"])
            system.execute("mv {} {}".format(f_a, f_b))
            print("INFO: saved {}".format(f_b))
        return
    
    def write_spheres_ini_input(setup, fname):
        with open(fname, 'w') as f:
            f.write("N = {}\n".format(setup["N"]))
            f.write("M_tot = {}\n".format(setup["M_tot"]))
            f.write("M_proj = {}\n".format(setup["M_proj"]))
            f.write("mantle_proj = {}\n".format(setup["mantle_proj"]))
            f.write("shell_proj = {}\n".format(setup["shell_proj"]))
            f.write("mantle_target = {}\n".format(setup["mantle_targ"]))
            f.write("shell_target = {}\n".format(setup["shell_targ"]))
            f.write("vel_vesc = {}\n".format(setup["vel_vesc_touching_ball"]))
            f.write("impact_angle = {}\n".format(setup["impact_angle_touching_ball"]))
            f.write("ini_dist_fact = {}\n".format(setup["ini_dist_fact"]))
            f.write("weibull_core = {}\n".format(setup["weibull_core"]))
            f.write("weibull_mantle = {}\n".format(setup["weibull_mantle"]))
            f.write("weibull_shell = {}\n".format(setup["weibull_shell"]))
            f.write("core_eos = T\n")
            f.write("mantle_eos = T\n")
            f.write("shell_eos = T\n")
            f.write("core_mat = {}\n".format(setup["core_mat"]))
            f.write("mantle_mat = {}\n".format(setup["mantle_mat"]))
            f.write("shell_mat = {}\n".format(setup["shell_mat"]))
            f.write("proj_rot_period = {}\n".format(setup["proj_rot_period"]))
            f.write("targ_rot_period = {}\n".format(setup["targ_rot_period"]))
            ax2sph = {0 : "x", 1 : "y", 2 : "z"}
            for obj in ["proj", "targ"]:
                for ax in range(3): f.write("{} = {}\n".format("{}_rot_axis_{}".format(obj, ax2sph[ax]), setup["rot_axis_{}".format(obj)][ax]))
            f.close()
        print("INFO: saved {}".format(fname))
        return
    
    def rot_max(m, r, value):
        '''
        maximum rotation period = orbital period at sea level
        m : mass of object in [kg]
        r : radius of object in [m] (extracted from SPH)
        value : number between 0 and 1 that defines how close object is to critical rotation speed
        '''
        T_crit = np.sqrt(4.*(np.pi**2)*(r**3) / (sim.gravitational_constant*m)) # [sec]. 3rd Kepler law
        omega_crit = 2*np.pi / T_crit # = sqrt(G*M / (r**3)) [rad/sec]
        omega = value * omega_crit
        if omega == 0.: T = -1 # no rotation
        else: T = 2*np.pi / omega
        return T_crit, T # critical rotation period [sec]
    
    def rot_axis(setup, obj):
        # init randomly oriented rotation axis. norm(axis) = rotation period
        T = setup["{}_rot_period".format(obj)]
        axis = 2 * np.random.random(size=3) - 1.
        axis = T * axis / np.linalg.norm(axis) # length of rotation axis vector = T
        setup["rot_axis_{}".format(obj)] = axis
        return setup
    
    def get_t_sim(setup):
        '''
        calculate simulation time depending on dynamical collision timescale R_sum / v_esc 
        '''
        t_sim = setup["coll_timescale"]  * (setup["t_sim_fact"] + setup["ini_dist_fact"])  # [sec]
        setup["t_sim"] = round(t_sim / 3600. + 1, 0) # round up to next full hour
        setup["n_frames"] = max(1, int(setup["n_frames"] * int(setup["t_sim"])))
        if setup["n_frames_video"] == -1: pass
        else: setup["n_frames_video"] = max(1, int(setup["n_frames_video"] * int(setup["t_sim"])))
        setup["t_delta"] = (setup["t_sim"] * 3600.) / setup["n_frames"] # [sec]
        return setup
    
    def identify_fragments(cfg, setup, results, frame):
        frame_key = "frame_{}".format(frame) 
        results[frame_key] = {} # misc infos
        f_frag_inp = setup["f_impact"].replace("impact.0000", "impact.{:04d}".format(frame))
        f_frag_out = "{}fragments_out.{:04d}".format(setup["path_sim"], frame)
        f_frag_idx = "{}fragments_idx.{:04d}".format(setup["path_sim"], frame)
        results, t_calc = system.execute("{}utils/postprocessing/fast_identify_fragments_and_calc_aggregates/fast_identify_fragments_{} -i {} -o {} -m {} -l {}".format(cfg["SPH"]["system"]["path_miluphcuda"], cfg["system"]["machine"], f_frag_inp, f_frag_out, setup["n_mat"], f_frag_idx), results)
        print("INFO: saved {}".format(f_frag_out))
        print("INFO: saved {}".format(f_frag_idx))
        frag_out = np.loadtxt(f_frag_out) # (sorted by mass)
        if len(frag_out.shape) == 1: frag_out = np.expand_dims(frag_out, axis=0)
        # select most massive fragments until the first one with less than n_sph_min particles
        n_frag_large = 0
        with open(f_frag_idx, "r") as f:
            for line in f:
                if line[0] != "#":
                    l = [int(x) for x in line.split()] # index, number of particles, sph indices
                    if l[1] < setup["n_sph_min"]: break
                    else: n_frag_large += 1        
        f.close()
        results[frame_key]["n_frag_tot"] = frag_out.shape[0]
        results[frame_key]["n_frag_large"] = n_frag_large
        if n_frag_large > 0:
            frag_large = frag_out[0:n_frag_large,:] # largest fragments
            sim.save_frag_large(setup, frag_large, frame)
        return results
    
    def sample_frames(setup):
        '''select frames for in-depth analysis'''
        frames = []
        for t in range(0, setup["n_frames"]+1, setup["frame_interval"]): frames.append(t)
        if frames[-1] != setup["n_frames"]: frames.append(setup["n_frames"]) # append last if not already included
        return frames
    
    def get_bary(p):
        """
        calculate barycenter of given SPH particles p
        # p: np.array of shape [n,7] (px, py, pz, vx, vy, vz, m)
        """
        S = np.sum(p[:,:6]*p[:,-1:], axis=0) / np.sum(p[:,-1:]) # barycenter position and velocity
        return S        
    
    def get_angular_momentum(cfg, setup, results, frame):
        if frame == 0: # calculate angular momentum for input frame
            d_raw = np.loadtxt(setup["f_impact"], usecols=(0,1,2,3,4,5,6,8)) # px, py, pz, vx, vy, vz, m, mat
            d = {"proj" : d_raw[:setup["N_proj"],:], "targ" : d_raw[setup["N_proj"]:,:]} # ordering of sph particles in impact.0000: proj, targ, point masses
            # calculate angular momentum
            setup["S_system_raw"] = sim.get_bary(d_raw[:,:7])
            frag_out = {}
            for obj in ["proj", "targ"]:
                setup["S_{}".format(obj)] = sim.get_bary(d[obj][:,:7])
                setup["L_orbit_{}".format(obj)] = sim.angular_momentum(d[obj][:,:7], np.array([0.,0.,0.,0.,0.,0.])) # reference point = system barycenter
                setup["L_spin_{}".format(obj)] = sim.angular_momentum(d[obj][:,:7], setup["S_{}".format(obj)]) # reference point = fragment barycenter
                frag_out[obj] = np.zeros(7) # init
                frag_out[obj][:6] = setup["S_{}".format(obj)]
                frag_out[obj][6] = setup["M_{}".format(obj)]
            # save fragments_large
            frag_large = np.stack((frag_out["targ"], frag_out["proj"])) # shape: [2,7]
            sim.save_frag_large(setup, frag_large, frame)            
            return setup
        
        else: # calculate angular momentum for output frames
            # system:
            frame_key = "frame_{}".format(frame)
            f_frag_inp = setup["f_impact"].replace("impact.0000", "impact.{:04d}".format(frame))
            d_raw = np.loadtxt(f_frag_inp, usecols=(0,1,2,3,4,5,6,11)) # px, py, pz, vx, vy, vz, m, mat
            results[frame_key]["S_system_raw"] = sim.get_bary(d_raw[:,:7]) # includes single-particle fragments. should be same as initial angular momentum (measure numerical effects)
            results[frame_key]["L_system_raw"] = sim.angular_momentum(d_raw[:,:7], results[frame_key]["S_system_raw"])
            f_info = "{}impact.{:04d}.info".format(setup["path_sim"], frame)
            with open(f_info, 'r') as f:
                for line in f:
                    if "Total angular momentum: norm(L) =" in line: results[frame_key]["L_system_raw_info"] = float(line.split()[-1])
            f.close()
            
            # find sph particle indices of largest fragments
            i, n = [], 0 
            f_frag_idx = "{}fragments_idx.{:04d}".format(setup["path_sim"], frame)
            frag_idx = []    
            with open(f_frag_idx, "r") as f:
                for line in f:
                    if line[0] != "#":
                        l = [int(x) for x in line.split()] # index, number of particles, sph indices
                        if l[1] < setup["n_sph_min"]: break
                        else: frag_idx.append({"n" : l[1], "idx" : l[2:]}) 
            f.close()
            for val in frag_idx:
                n = n + val["n"]
                i = i + val["idx"]
            if len(frag_idx) != results[frame_key]["n_frag_large"]: raise ValueError()
            if len(i) != n: raise ValueError(len(i), n)
            d = d_raw[i[:],:] # without debris
            results[frame_key]["S_system"] = sim.get_bary(d[:,:7]) # system barycenter without small fragments
            for i in range(results[frame_key]["n_frag_large"]):
                d_i = d_raw[frag_idx[i]["idx"][:],:] # all particles of a single fragment
                results[frame_key]["S_frag{}".format(i)] = sim.get_bary(d_i[:,:7])
                results[frame_key]["L_orbit_frag{}".format(i)] = sim.angular_momentum(d_i[:,:7], results[frame_key]["S_system"]) # reference point = system barycenter
                results[frame_key]["L_spin_frag{}".format(i)] = sim.angular_momentum(d_i[:,:7], results[frame_key]["S_frag{}".format(i)]) # reference point = fragment barycenter
            return results
        
    def angular_momentum(p, S):
        '''
        p : particle information: shape [n,6]: absolute position, absolute velocity, mass
        S : absolute position and velocity of reference point (e.g. barycenter of fragment or system). shape [6]
        returns: norm of angular momentum
        '''
        r, v, m = p[:,:3], p[:,3:6], p[:,6]
        m = np.expand_dims(m, axis=-1) # [n,1]
        r_bary, v_bary = S[:3], S[3:]
        r_vec = r - r_bary
        v_vec = v - v_bary
        p_vec = m * v_vec
        L_vec = np.cross(r_vec, p_vec) # [n,3]
        L_vec = np.sum(L_vec, axis=0) # [3]
        L = np.linalg.norm(L_vec) # [1]
        return {"scalar" : L, "vector" : L_vec}
    
    def calc_aggregates(cfg, setup, results, frame):
        f_aggregates = "{}aggregates.{:04d}".format(setup["path_sim"], frame)
        f_frag_out = "{}fragments_out.{:04d}".format(setup["path_sim"], frame)
        results, t_calc = system.execute("{}utils/postprocessing/fast_identify_fragments_and_calc_aggregates/calc_aggregates -f {} -n {} -t > {}".format(cfg["SPH"]["system"]["path_miluphcuda"], f_frag_out, setup["n_mat"], f_aggregates), results)
        print("INFO: saved {}".format(f_aggregates))
        return results
    
    def visualize_frag(setup, results, frame, every=1):
        '''visualize fragments'''
        if frame > 0:
            if results["frame_{}".format(frame)]["n_frag_large"] == 0: return # nothing to visualize
        f_frag_large = "{}fragments_large.{:04d}".format(setup["path_sim"], frame)
        d = np.loadtxt(f_frag_large)
        if len(d.shape) == 1: d = np.expand_dims(d, axis=0)
        d[:,6] *= (2. / 1e23) # visually pleasing marker sizes
        plt.scatter(d[:,0], d[:,1], s=d[:,6], marker="s", color="black", label="fragments ({})".format(d.shape[0]))
        for i in range(d.shape[0]): plt.arrow(d[i,0], d[i,1], 1000*d[i,3], 1000*d[i,4], head_width=0, head_length=0, color="red")
        plt.xlabel("x-pos [m]")
        plt.ylabel("y-pos [m]")
        plt.legend()
        plt.grid(linestyle='dotted')
        plt.axis('equal')
        scale_factor = 0.1
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        dx, dy = xmax - xmin, ymax - ymin
        xymin, xymax, dxy = min(xmin, ymin), max(xmax, ymax), max(dx, dy)
        plt.xlim([xymin - dxy * scale_factor, xymax + dxy * scale_factor])
        plt.ylim([xymin - dxy * scale_factor, xymax + dxy * scale_factor])
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.title("{}, frame: {}".format(setup["ID"], frame))
        fname = "{}fragments_vis.{:04d}.png".format(setup["path_sim"], frame)
        plt.savefig(fname)
        plt.close()
        print("INFO: saved {}".format(fname))
        return
    
    def check_validity(setup):
        '''check validity of SPH simulation'''
        line = str(subprocess.check_output(['tail', '-1', "{}miluphcuda_error".format(setup["path_sim"])]))
        line = line.split()
        if "end" not in line[0] or str(setup["n_frames"]) not in line[2]: valid = False
        else: valid = True
        if not valid:
            print("WARNING: invalid sim: {}. clearing {}".format(setup["ID"], setup["path_sim"]))
            system.execute("rm -r {}".format(setup["path_sim"]))
        return valid
    
    def postprocess_frames(cfg, setup, results, frames):
        for frame in frames:
            if frame == 0:
                sim.visualize_frag(setup, results, frame)
                sim.save_frame(cfg, setup, results, frame)
            else:
                results = sim.identify_fragments(cfg, setup, results, frame)
                results = sim.get_angular_momentum(cfg, setup, results, frame)
                results = sim.calc_aggregates(cfg, setup, results, frame)
                sim.visualize_frag(setup, results, frame)
                sim.save_frame(cfg, setup, results, frame)
        return setup, results
    
    def load_checkpoint(cfg, ID, checkpoint):
        '''continue pipeline post run_sim'''
        setup = {"path_sim" : "{}winter/{}/".format(cfg["SPH"]["system"]["path_Astro_SPH"], ID)}
        f_setup_npy = "{}setup.npy".format(setup["path_sim"])
        setup = np.load(f_setup_npy, allow_pickle=True).item()
        print("INFO: loaded {}".format(f_setup_npy))
        if checkpoint == 1: results = None
        elif checkpoint == 2: results = {"wall_clock_time_sph" : None}
        return setup, results
    
class ML:

    def download_results():
        dir_source = "winter@orca:/system/user/publicwork/winter/output/19_CSV/"
        dir_target = "/mnt/data/output/19_CSV/"
        command = "rsync -av {} {}".format(dir_source, dir_target)
        system.execute(command)

    def prepare_checkpoint_directory(cfg):
        if os.path.isdir(cfg["ML"]["system"]["dir_check"]):
            shutil.rmtree(cfg["ML"]["system"]["dir_check"], ignore_errors=False)
            print('INFO: deleted checkpoint directory')
        os.makedirs(cfg["ML"]["system"]["dir_check"])
        print('INFO: created checkpoint directory: {}'.format(cfg["ML"]["system"]["dir_check"]))
        fname = "{}cfg".format(cfg["ML"]["system"]["dir_check"])
        sim.save_dic(cfg, fname)
        return

    def stats(x, min=None, max=None):
        """get basic statistics of tensor"""
        if min is not None:
            if x.min().item() < min:
                print(f"WARNING: {x.min().item()} < {min}")
        if max is not None:
            if x.max().item() > max:
                print(f"WARNING: {x.max().item()} > {max}")
        out = "shape: {}, mean{:.09f}, std{:.09f}, min{:.09f}, max{:.09f}".format(x.shape, x.mean().item(), x.std().item(), x.min().item(), x.max().item())
        return out

    def scenario(m_t, m_p, m_lr, m_2lr):
        """
        m_t.shape: [b]
        m_p.shape: [b]
        m_lr.shape: [b]
        m_2lr.shape: [b]

        classes:
        erosion:
            general condition: largest aggregate < target
            subclasses:
                0: largest aggregate < 0.5 * target
                1: largest aggregate > 0.5 * target
        accretion:
            general condition: largest aggregate > target
            subclasses:
                2: largest aggregate < target + 0.5 projectile
                3: largest aggregate > target + 0.5 projectile
        hit&run:
            general condition: largest aggregate > target and 2nd largest aggregate > 0.1 * projectile
            subclasses:
                4: 2nd largest aggregate < 0.5 * projectile
                5: 2nd largest aggregate > 0.5 * projectile
        perfect merging:
            6: general condition: largest aggregate = target + projectile
        """
        bs = m_t.shape[0]
        cl = torch.zeros(bs, dtype=torch.long)
        for i in range(bs):
            if m_lr[i] < m_t[i]:  # erosion
                if m_lr[i] < 0.5 * m_t[i]:
                    cl[i] = 0
                else:
                    cl[i] = 1
            elif m_lr[i] > m_t[i] and m_2lr[i] > 0.1 * m_p[i]:  # hit and run
                if m_2lr[i] < 0.5 * m_p[i]:
                    cl[i] = 4
                else:
                    cl[i] = 5
            elif m_lr[i] == m_t[i] + m_p[i]:  # perfect accretion
                cl[i] = 3  # dummy class (can't happen in data)
            else:  # partial accretion
                if m_lr[i] < m_t[i] + 0.5 * m_p[i]:
                    cl[i] = 2
                else:
                    cl[i] = 3
        return cl

    def balanced_acc(cl_y, cl_z, n_cl=6):
        '''
        calculates balanced accuracy
        '''
        bacc = torch.zeros(1)
        for c in range(n_cl):
            idx = cl_z == c
            y, z = cl_y[idx], cl_z[idx]
            n_tot_c = y.shape[0]  # total number of samples for class c
            if n_tot_c == 0:
                print(f"INFO: balanced_acc: encountered {n_tot_c} samples for class {c}")
            else:
                n_cor_c = torch.where(y == z, 1, 0).sum()  # number of correct samples for class c
                bacc += (n_cor_c / n_tot_c)
        bacc = bacc / n_cl
        return bacc.item()

    def loss(cfg, y, z, L_reg=None, validate=False, x=None):
        info, z = z[0], z[1]

        if cfg["ML"]["dataset"]["winter"]["targets"] != "agg":
            raise NotImplementedError()
        if y.shape != z.shape or y.dim() != 4:
            raise ValueError(y.shape, z.shape)

        # weights according to object masses:
        w = torch.zeros((z.shape[0], z.shape[1], z.shape[2], 3), device=y.device)
        w[:,:,:,0] = z[:, :, :, 0].detach()  # la
        w[:,:,:,1] = z[:, :, :, 3].detach()  # 2la
        w[:,:,:,2] = z[:, :, :, 6].detach()  # rest
        w = w / w.sum(dim=-1).unsqueeze(-1)
        idx = {0: [0, 1, 2, 9, 10, 11, 12, 13, 14],
               1: [3, 4, 5, 15, 16, 17, 18, 19, 20],
               2: [6, 7, 8, 21, 22, 23, 24, 25, 26]}

        if "ae" in cfg["ML"]["training"]["loss_fn"]:
            err = (y - z).abs()  # absolute error for each sample and each feature
        elif "se" in cfg["ML"]["training"]["loss_fn"]:
            err = (y - z) ** 2  # squared error for each sample and each feature
        for o in range(3):
            err[:, :, :, idx[o]] *= w[:,:,:,o].unsqueeze(-1)  # for training

        if validate:
            SE = (y - z) ** 2  # squared error for each sample and each feature
            for o in range(3):
                SE[:, :, :, idx[o]] *= w[:,:,:,o].unsqueeze(-1)  # for validation

        # subselect error depending on task
        if cfg["ML"]["training"]["loss_fn"] == "ae":
            pass
        elif cfg["ML"]["training"]["loss_fn"] == "se":
            pass
        elif cfg["ML"]["training"]["loss_fn"] == "ae_mass":
            idx = [0, 3, 6]
            err = err[:,:,:,idx]
        elif cfg["ML"]["training"]["loss_fn"] == "ae_mat":
            idx = [1, 2, 4, 5, 7, 8]
            err = err[:,:,:,idx]
        elif cfg["ML"]["training"]["loss_fn"] == "ae_pos":
            idx = [9, 10, 11, 15, 16, 17, 21, 22, 23]
            err = err[:,:,:,idx]
        elif cfg["ML"]["training"]["loss_fn"] == "ae_vel":
            idx = [12, 13, 14, 18, 19, 20, 24, 25, 26]
            err = err[:,:,:,idx]
        elif cfg["ML"]["training"]["loss_fn"] == "ae_mass_mat":
            err = err[:,:,:,:9]
        elif cfg["ML"]["training"]["loss_fn"] == "ae_pos_vel":
            err = err[:,:,:,9:]
        elif cfg["ML"]["training"]["loss_fn"] == "ae_mass_mat_pos":
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 21, 22, 23]
            err = err[:,:,:,idx]  # subselect for task

        L = err.mean()  # total loss over minibatch
        if L_reg is not None: L = L + L_reg

        if validate:

            stats = {}

            # validation performance:
            RMSE = {}
            keycols = {"mass": [0, 3, 6], "fraction_mat": [1, 2, 4, 5, 7, 8],
                       "position": [9, 10, 11, 15, 16, 17, 21, 22, 23],
                       "velocity": [12, 13, 14, 18, 19, 20, 24, 25, 26],
                       "tot": list(range(0, 27))}
            for key, cols in keycols.items():
                stats[key] = torch.sqrt(SE[:, :, :, cols].detach().mean(dim=(0, 2, 3))).cpu()  # [b]

            # classification accuracy:
            y, z = y.squeeze(0).squeeze(1), z.squeeze(0).squeeze(1)
            x = x[:, 0, :]
            m_tot = x[:, 1]
            gamma = x[:, 2]
            m_t = m_tot / (gamma + 1.)
            m_p = gamma * m_t
            cl_y = ML.scenario(m_t, m_p, y[:, 0], y[:, 3])
            cl_z = ML.scenario(m_t, m_p, z[:, 0], z[:, 3])
            stats["cl_z"] = cl_z.cpu()
            stats["cl_y"] = cl_y.cpu()
            try:
                stats["impact_angle_touching_ball"] = info[:, 3].cpu()
                stats["vel_vesc_touching_ball"] = info[:, 4].cpu()
            except:
                stats["impact_angle_touching_ball"] = torch.zeros(1)
                stats["vel_vesc_touching_ball"] = torch.zeros(1)
            stats["L"] = L.detach().unsqueeze(0).cpu()

            return stats

        else:
            return L

    def n_params(model, info=False):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)  # number of learnable parameters of given model

    # Quaternions:
    def euler_to_quaternion(euler):
        '''
        euler angles yaw (Z), pitch (Y), roll (X), given in rad
        from: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        Body 3-2-1 sequence:
            The airplane first does yaw (Body-Z) turn during taxiing onto the runway,
            then pitches (Body-Y) during take-off,
            and finally rolls (Body-X) in the air.
        '''
        if euler.dim() != 2:
            raise ValueError(euler.shape)
        yaw, pitch, roll = euler[:,0], euler[:,1], euler[:,2]
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr
        q = torch.stack((qw, qx, qy, qz), dim=1)  # [b,4]
        return q

    def conjugate_quaternion(q):
        '''returns the conjugate q_ of quaternion q'''
        q_ = torch.cat((q[:,:,:1], -q[:,:,1:]), dim=-1)
        return q_

    def prepare(x):
        '''prepends zeros to the last (2nd) dimension'''
        if x.shape[-1] == 3:  # will be true for pc, not for q or q_
            return torch.cat((torch.zeros_like(x[:, :, :1]), x), dim=2)
        else:
            return x

    def quaternion_multiply(a, b):
        '''
        multiply two quaternion tensors
        note: also bring tensors into correct shape
        '''
        a = ML.prepare(a)
        b = ML.prepare(b)
        w1, x1, y1, z1 = a[:, :, 0:1], a[:, :, 1:2], a[:, :, 2:3], a[:, :, 3:]
        w2, x2, y2, z2 = b[:, :, 0:1], b[:, :, 1:2], b[:, :, 2:3], b[:, :, 3:]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return torch.cat((w, x, y, z), dim=-1)

    def quaternion_rotate(pc, q, inverse=False):
        '''
        Rotates 3D point cloud pc, represented as a quaternion q.
        Args:
            pc: [b,n,3] point cloud
            q: [b,4] rotation quaternion. dim 1: (w,x,y,z)
        Returns:
            q * pc * q_
        '''
        if pc.dim() != 3:
            raise ValueError(pc.shape)
        if q.dim() != 2 or q.shape[-1] != 4:
            raise ValueError(q.shape)

        if pc.shape[-1] == 3:
            k = 1
        elif pc.shape[-1] == 6:  # stack along dim 1 and rotate several vectors in parallel
            k = 2
            pc = torch.cat((pc[:,:,:3], pc[:,:,3:]), dim=1)
        else:
            raise NotImplementedError()
        # normalize q:
        q_norm = torch.norm(q, dim=1).unsqueeze(-1)
        q = q / q_norm
        # calculate conjugate:
        q = q.unsqueeze(dim=1)  # [b,1,4]
        q_ = ML.conjugate_quaternion(q)
        # calculate transformation:
        if not inverse:
            wxyz = ML.quaternion_multiply(ML.quaternion_multiply(q, pc), q_)  # [b,n,4]
        else:
            wxyz = ML.quaternion_multiply(ML.quaternion_multiply(q_, pc), q)  # [b,n,4]
        pc = wxyz[:, :, 1:]  # [b,n,3]
        if k == 2:  # unstack
            s = pc.shape[1]
            pc = torch.cat((pc[:,:s//2,:], pc[:,s//2:,:]), dim=2)
        return pc  # transformed point cloud

if __name__ == "__main__":
    print('Done.')

