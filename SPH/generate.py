import sys
import os
import time
import utils
from datetime import datetime
import numpy as np

def get_setup(cfg):
    '''prepares setup for a specific SPH run'''    
    if cfg["SPH"]["sim"]["setup_type"] == "custom":
        # system:
        setup = cfg["SPH"]["sim"]["custom_setup"]
        setup["t_calc_tot_temp"] = time.time()
        setup["setup_type"] = cfg["SPH"]["sim"]["setup_type"]
        
        # sizes:
        setup["M_targ"] = setup["M_tot"] - setup["M_proj"]
        setup["gamma"] = setup["M_proj"] / setup["M_targ"]
        
        # materials:
        for key in ["proj", "targ"]: setup["core_{}".format(key)] = 1 - setup["mantle_{}".format(key)] - setup["shell_{}".format(key)] # core mass fraction
        
        # damage:
        setup["weibull_core"] = 0
        setup["weibull_mantle"] = 0
        setup["weibull_shell"] = 0
        
        # rotation (rot period already specified):
        setup["rot_axis_proj"] = np.array([setup["proj_rot_axis_x"], setup["proj_rot_axis_y"], setup["proj_rot_axis_z"]])
        setup["rot_axis_targ"] = np.array([setup["targ_rot_axis_x"], setup["targ_rot_axis_y"], setup["targ_rot_axis_z"]])

        # clean-up
        del(setup["n_simulations"])
        del(setup["checkpoint"])
    
    elif cfg["SPH"]["sim"]["setup_type"] == "random":
        # system:
        setup = {}
        setup["t_calc_tot_temp"] = time.time()
        setup["setup_type"] = cfg["SPH"]["sim"]["setup_type"]
        random_setup = cfg["SPH"]["sim"]["random_setup"]
        
        # SPH settings:
        setup["t_sim_fact"] = utils.sim.rand_min_max(random_setup["t_sim_fact_min"], random_setup["t_sim_fact_max"])
        setup["ini_dist_fact"] = utils.sim.rand_min_max(random_setup["ini_dist_fact_min"], random_setup["ini_dist_fact_max"])
        setup["n_frames"] = random_setup["n_frames"]
        setup["frame_interval"] = random_setup["frame_interval"]
        setup["N"] = utils.sim.rand_int_min_max(random_setup["N_min"], random_setup["N_max"])  # number of SPH particles
        
        # sizes:
        setup["M_tot"] = utils.sim.rand_min_max(random_setup["M_tot_min"], random_setup["M_tot_max"])  # total mass [kg]
        setup["gamma"] = utils.sim.rand_min_max(random_setup["gamma_min"], random_setup["gamma_max"])  # mass ratio projectile / target
        setup["M_targ"] = setup["M_tot"] / (setup["gamma"] + 1.)
        setup["M_proj"] = setup["M_tot"] - setup["M_targ"]
        
        # materials:
        setup["n_mat"] = 3
        setup["f_m_key"] = "iron_basalt_water"
        setup["core_mat"] = "Iron" # ID=0
        setup["mantle_mat"] = "BasaltNakamura" # ID=1
        setup["shell_mat"] = "Water" # ID=2
        for key in ["proj", "targ"]:
            setup["core_{}".format(key)] = utils.sim.rand_min_max(random_setup["iron_fraction_min"], random_setup["iron_fraction_max"])
            setup["shell_{}".format(key)] = utils.sim.rand_min_max(random_setup["water_fraction_min"], random_setup["water_fraction_max"])
            setup["mantle_{}".format(key)] = 1. - setup["core_{}".format(key)] - setup["shell_{}".format(key)]
            if setup["shell_{}".format(key)] < 0.1: # remove water shell for obj
                setup["mantle_{}".format(key)] += setup["shell_{}".format(key)]
                setup["shell_{}".format(key)] = 0.
        if setup["shell_proj"] < 0.1 and setup["shell_targ"] < 0.1: # no water present at all --> 2 materials
            setup["n_mat"] = 2
            setup["f_m_key"] = "iron_basalt"
            setup["shell_mat"] = "None"
        
        # damage:
        setup["weibull_core"] = 0
        setup["weibull_mantle"] = 0
        setup["weibull_shell"] = 0
        
        # no rotation (yet):
        for obj in ["proj", "targ"]:
            setup["{}_rot_period".format(obj)] = -1.0
            setup["rot_axis_{}".format(obj)] = np.zeros(3)
            
        # imact geometry:
        setup["vel_vesc_touching_ball"] = utils.sim.rand_min_max(random_setup["vel_vesc_min"], random_setup["vel_vesc_max"]) # impact velocity [v_esc]
        setup["impact_angle_touching_ball"] = utils.sim.rand_min_max(random_setup["impact_angle_min"], random_setup["impact_angle_max"]) # impact angle [deg]
        
        # fragments:
        setup["n_sph_min"] = random_setup["n_sph_min"]
        
    setup["ID"] = utils.sim.get_ID(cfg)
    setup["GPU_name"] = cfg["system"]["GPU_name"]
    return setup

def init_sim(cfg, setup):
    # create simulation directory
    setup["path_sim"] = "{}winter/{}/".format(cfg["SPH"]["system"]["path_Astro_SPH"], setup["ID"])
    os.mkdir(setup["path_sim"])
    os.chdir(setup["path_sim"])
    sys.stderr = open('{}err.txt'.format(setup["path_sim"]), 'w')
    
    # pick correct material.cfg
    cp_source = "{}material_{}.cfg".format(cfg["SPH"]["system"]["path_spheres_ini"], setup["f_m_key"])
    setup["f_mat"] = "{}material.cfg".format(setup["path_sim"])
    os.system("cp {} {}".format(cp_source, setup["f_mat"]))
    
    # prepare filenames
    f_spheres_ini_input = "{}spheres_ini.input".format(setup["path_sim"])
    f_spheres_ini_log = "{}spheres_ini.log".format(setup["path_sim"])
    setup["f_impact"] = "{}impact.0000".format(setup["path_sim"])
        
    # create impact.0000 with spheres_ini
    utils.sim.write_spheres_ini_input(setup, f_spheres_ini_input)
    command = "{}spheres_ini -S {} -f {} -o {} -m {} -H -G 2 -O 3 -L 0.5 1> {}".format(cfg["SPH"]["system"]["path_spheres_ini"], cfg["SPH"]["system"]["path_spheres_ini"], f_spheres_ini_input, setup["f_impact"], setup["f_mat"], f_spheres_ini_log) # -G 2: spherical shells, -O 0: HYDRO
    setup, t_calc = utils.system.execute(command, setup)
    setup["t_calc_spheres_ini"] = t_calc
    if "error" in setup: return setup
    
    # extract important information from spheres_ini log
    with open(f_spheres_ini_log, 'r') as f:
        for line in f:
            if "projectile: N_des =" in line: setup["N_proj"] = int(line.split()[6])
            if "target:     N_des =" in line: setup["N_targ"] = int(line.split()[6])
            if "projectile: desired:      R =" in line: setup["R_proj"] = float(line.split()[4]) # required for rotation
            if "target: desired:      R =" in line: setup["R_targ"] = float(line.split()[4]) # required for rotation
            if "collision timescale (R_p+R_t)/|v_imp| =" in line: setup["coll_timescale"] = float(line.split()[4]) # required for simulation time
    f.close()
    
    if cfg["SPH"]["sim"]["setup_type"] == "random": # re-initialize rotating configurations
        
        # rotation:
        for obj in ["proj", "targ"]:
            rot = utils.sim.rand_min_max(0, cfg["SPH"]["sim"]["random_setup"]["rot_limit"])
            setup["{}_rot_period_crit".format(obj)], setup["{}_rot_period".format(obj)] = utils.sim.rot_max(setup["M_{}".format(obj)], setup["R_{}".format(obj)], rot)
            setup = utils.sim.rot_axis(setup, obj)
                    
        # overwrite impact.0000 with spheres_ini
        utils.sim.write_spheres_ini_input(setup, f_spheres_ini_input)
        command = "{}spheres_ini -S {} -f {} -o {} -m {} -H -G 2 -O 3 -L 0.5 1> {}".format(cfg["SPH"]["system"]["path_spheres_ini"], cfg["SPH"]["system"]["path_spheres_ini"], f_spheres_ini_input, setup["f_impact"], setup["f_mat"], f_spheres_ini_log) # -G 2: spherical shells, -O 0: HYDRO
        setup, t_calc = utils.system.execute(command, setup)
        setup["t_calc_spheres_ini"] += t_calc
        if "error" in setup: return setup
    
    print("INFO: updated {}".format(setup["f_impact"]))
    print("INFO: updated {}".format(f_spheres_ini_log))
    
    # angular momentum:
    setup = utils.sim.get_angular_momentum(cfg, setup, None, 0)

    # calculate simulation time:
    setup = utils.sim.get_t_sim(setup)

    # save setup:
    utils.sim.save_dic(setup, "{}setup".format(setup["path_sim"]))

    return setup

def run_sim(cfg, setup):
    if "error" in setup: return {"error" : setup["error"]}
    results = {}
    command = f'{cfg["SPH"]["system"]["path_miluphcuda"]}miluphcuda_{cfg["system"]["gpu_arch"]}_{cfg["system"]["machine"]} -v -I rk2_adaptive -Q 1e-4 -n {setup["n_frames"]} -a 0.5 -t {setup["t_delta"]} -f {setup["path_sim"]}impact.0000 -m {setup["path_sim"]}material.cfg -s -g > {setup["path_sim"]}miluphcuda_output 2> {setup["path_sim"]}miluphcuda_error'
    results, t_calc = utils.system.execute(command, results)
    results["t_calc_SPH"] = t_calc
    return results

def exit_sim(cfg, setup, results, checkpoint):    
    # safety:
    if "error" in results:
        return setup, results
    if checkpoint == 0:
        results["valid"] = utils.sim.check_validity(setup)
        if not results["valid"]:
            return setup, results
    
    # postprocessing:
    results["ID"] = setup["ID"]
    results["t_calc_postprocess"] = time.time()
    dir = "{}frames/".format(setup["path_sim"])
    if not os.path.isdir(dir):
        os.mkdir(dir)
    frames = utils.sim.sample_frames(setup)
    setup, results = utils.sim.postprocess_frames(cfg, setup, results, frames)
    t_cur = time.time()
    results["t_calc_postprocess"] = t_cur - results["t_calc_postprocess"]
    results["t_calc_visualize"] = t_cur
    t_cur = time.time()
    results["t_calc_visualize"] = t_cur - results["t_calc_visualize"]
    results["t_calc_tot"] = t_cur - setup["t_calc_tot_temp"]
    del setup["t_calc_tot_temp"]
    return setup, results

def save_dic(cfg, setup, results):
    utils.sim.save_dic(cfg, "{}cfg".format(setup["path_sim"]))
    utils.sim.save_dic(setup, "{}setup".format(setup["path_sim"]))
    utils.sim.save_dic(results, "{}results".format(setup["path_sim"]))
    return

def clean(cfg, setup, results):
    if "error" in setup or "error" in results: return # skip cleanup
    if cfg["SPH"]["sim"]["{}_setup".format(cfg["SPH"]["sim"]["setup_type"])]["cleanup"]:
        files_del = ["impact.*", "fragments_*", "miluphcuda_output", "miluphcuda_error", "conserved_quantities.log", "target.structure", "target.SEAGen", "projectile.structure", "projectile.SEAGen"]
        for file in files_del: utils.system.execute("rm {}{}".format(setup["path_sim"], file, results))
    return

def generate(cfg):
    a = input("CAUTION: You are about to add new datapoints! Are you sure what you are doing? ")
    if a != "yes":
        sys.exit()
    del cfg["ML"]
    
    if cfg["system"]["machine"] == "server": utils.system.check_mem("/publicdata/", min_GB=100)
    n_simulations = cfg["SPH"]["sim"]["{}_setup".format(cfg["SPH"]["sim"]["setup_type"])]["n_simulations"]
    checkpoint = cfg["SPH"]["sim"]["{}_setup".format(cfg["SPH"]["sim"]["setup_type"])]["checkpoint"]
    checkpoint_ID = cfg["SPH"]["sim"]["{}_setup".format(cfg["SPH"]["sim"]["setup_type"])]["checkpoint_ID"]
    if checkpoint != 0: n_simulations = 1  # debug mode
    date_start = str(datetime.now())
    
    for i in range(n_simulations):
        sys.stderr = sys.__stderr__
        print("\nINFO: running datapoint #{}/{} @{} ...".format(i+1, n_simulations, str(datetime.now())))
        if checkpoint == 0: # default: preprocessing + simulation + postprocessing
            setup = get_setup(cfg)
            setup = init_sim(cfg, setup)
            results = run_sim(cfg, setup)
            setup, results = exit_sim(cfg, setup, results, checkpoint)
            save_dic(cfg, setup, results)
            clean(cfg, setup, results)
        elif checkpoint == 1: # for debugging: simulation + postprocessing
            setup, _ = utils.sim.load_checkpoint(cfg, checkpoint_ID, checkpoint)
            results = run_sim(cfg, setup)
            setup, results = exit_sim(cfg, setup, results, checkpoint)
            save_dic(cfg, setup, results)
            clean(cfg, setup, results)
        elif checkpoint == 2: # for debugging: postprocessing
            setup, results = utils.sim.load_checkpoint(cfg, checkpoint_ID, checkpoint)
            setup, results = exit_sim(cfg, setup, results, checkpoint)
            save_dic(cfg, setup, results)
            clean(cfg, setup, results)
        checkpoint = 0
    print("INFO: performed {} simulations between {} and {}".format(n_simulations, date_start, str(datetime.now())))

if __name__ == '__main__':
    print('INFO: Done.')