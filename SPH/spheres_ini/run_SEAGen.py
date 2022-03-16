#!/usr/bin/env python3

"""
Generates a sphere of particles arranged in spherical shells by calling SEAGen.
The particles are approximately equidistant and thus have approximately equal "volumes".

"""


try:
    import sys
    import numpy as np
    import traceback
    import argparse
except Exception as e:
    print("ERROR! Cannot properly import Python modules in 'run_SEAGen.py'. Exiting ...")
    print(str(type(e).__name__))
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)


parser = argparse.ArgumentParser(description="Script to run SEAGen to create a sphere of approx. equidistant particles arranged in spherical shells.")
parser.add_argument("--SEAGen_dir", help = "specify directory containing SEAGen module 'seagen.py'", type = str, metavar = "")
parser.add_argument("--N_des", help = "specify desired particle number in the sphere", type = int, metavar = "")
parser.add_argument("--R_total", help = "specify total radius of sphere", type = float, metavar = "")
parser.add_argument("--R_core", help = "specify core radius of sphere", type = float, metavar = "")
parser.add_argument("--R_mantle", help = "specify mantle radius of sphere", type = float, metavar = "")
parser.add_argument("--outfile", help = "specify output file, where the first 3 lines are comments and start with '#' (the first line contains the total particle number after '# '), followed by lines with  1.x  2.y  3.z  4.mat-type  5.radius  6.density  7.mass, default is 'particles.txt'", type = str, metavar = "", default = "particles.txt")
parser.add_argument("-v", help= "be verbose" , action='store_true')
args = parser.parse_args()


try:
    sys.path.append(args.SEAGen_dir)
    import seagen
except Exception as e:
    print("ERROR! Cannot properly import SEAGen module in 'run_SEAGen.py'. Exiting ...")
    print(str(type(e).__name__))
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)


# determine structure of sphere
has_core = has_mantle = has_shell = False
if args.R_core/args.R_total > 1.0e-6:
    has_core = True
if (args.R_mantle-args.R_core)/args.R_total > 1.0e-6:
    has_mantle = True
if (args.R_total-args.R_mantle)/args.R_total > 1.0e-6:
    has_shell = True

# create array for radii (providing only the outer radius of core/mantle/shell seems sufficient for SEAGen)
radii_list = [1.0e-6*args.R_total]  # small inner value is necessary for SEAGen

if has_core:    # if there is a core
    radii_list.append(args.R_core)
if has_mantle:    # if there is a mantle
    radii_list.append(args.R_mantle)
if has_shell:    # if there is a shell
    radii_list.append(args.R_total)
radii = np.asarray(radii_list, dtype = float)   # was built as Python list 'radii_list' and now converted to numpy ndarray 'radii'

# create array for densities
densities = np.ones(len(radii))

# create array for mat-types
# NOTE: assuming mat-types 0/1/2 for core/mantle/shell
if has_core:
    mat_types_list = [0, 0]
    if has_mantle:
        mat_types_list.append(1)
    if has_shell:
        mat_types_list.append(2)
elif has_mantle:
    mat_types_list = [1, 1]
    if has_shell:
        mat_types_list.append(2)
elif has_shell:
    mat_types_list = [2, 2]
else:
    print("ERROR. Sphere to generate with SEAGen appears to have neither core nor mantle nor shell ...")
    sys.exit(1)
mat_types = np.asarray(mat_types_list, dtype = int)     # was built as Python list 'mat_types_list' and now converted to numpy ndarray 'mat_types'

# consistency check
if len(mat_types) != len(radii):
    print("ERROR when generating sphere with SEAGen. Length of array 'mat_types' is different than those of 'radii' ...")
    sys.exit(1)

# run SEAGen
if args.v:
    particles = seagen.GenSphere(N_picle_des = args.N_des, A1_r_prof = radii, A1_rho_prof = densities, A1_mat_prof = mat_types, verb=1)
else:
    particles = seagen.GenSphere(N_picle_des = args.N_des, A1_r_prof = radii, A1_rho_prof = densities, A1_mat_prof = mat_types, verb=0)

# write particles to output file
ofl=open(args.outfile, "w")
ofl.write("# {0}\n".format(particles.N_picle) )
ofl.write("#\n")
ofl.write("# 1.x  2.y  3.z  4.mat-type  5.radius  6.density  7.mass\n")
for i in range(0,len(particles.x)):
    ofl.write("{0:.16e}\t{1:25.16e}{2:25.16e}{3:5d}{4:25.16e}{5:20g}{6:20g}\n".format(particles.x[i], particles.y[i], particles.z[i], particles.mat[i], particles.r[i], particles.rho[i], particles.m[i]) )
ofl.close()

if args.v:
    volumes = np.divide(particles.m, particles.rho)
    print("\nParticle volume statistics (computed from SEAGen rho and m):")
    print("    minimum  = {0:g}".format(np.amin(volumes)) )
    print("    maximum  = {0:g}".format(np.amax(volumes)) )
    print("    mean     = {0:g}".format(np.mean(volumes)) )
    print("    median   = {0:g}".format(np.median(volumes)) )
    print("    std.dev. = {0:g}".format(np.std(volumes)) )


# debugging
#print("\n\nradii = {0}\ndensities = {1}\nmat_types = {2}".format(radii, densities, mat_types) )
#print("\nrho_min = {0:g}\nrho_max = {1:g}\nrho_mean = {2:g}\nrho_median = {3:g}\nrho_std.dev. = {4:g}".format(np.amin(particles.rho), np.amax(particles.rho), np.mean(particles.rho), np.median(particles.rho), np.std(particles.rho) ) )
#print("\nmass_min = {0:g}\nmass_max = {1:g}\nmass_mean = {2:g}\nmass_median = {3:g}\nmass_std.dev. = {4:g}\n".format(np.amin(particles.m), np.amax(particles.m), np.mean(particles.m), np.median(particles.m), np.std(particles.m) ) )

