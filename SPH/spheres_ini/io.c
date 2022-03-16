// Contains spheres_ini functions related to input/output.

// Christoph Burger 07/Nov/2020


#include <stdio.h>
#include <string.h>
#include <math.h>

#include "spheres_ini.h"

#ifdef MILUPHCUDA
#include <libconfig.h>
#endif


void help(char* programname)
{
    fprintf(stdout, "\nspheres_ini creates an initial SPH particle distribution of two colliding spheres, for use with ");
#ifdef MILUPH
    fprintf(stdout, "miluph.\n");
#endif
#ifdef MILUPHCUDA
    fprintf(stdout, "miluphcuda.\n");
#endif
    fprintf(stdout, "\nBoth bodies can in general consist of three different layers/materials (core/mantle/shell),"
                    " as defined in the input parameter file (see the -f flag).\n"
                    "\n"
                    "The bodies' internal/radial structure can be:\n"
                    "    o homogeneous (in each layer)\n"
                    "    o initialized following computed hydrostatic profiles (see the -H flag)\n"
                    "    o follow some given radial profiles (see the -R flag)\n"
                    "\n"
                    "Different geometrical particle arrangements are available (see the -G flag):\n"
                    "    o simple-cubic lattice\n"
                    "    o hexagonally close-packed lattice\n"
                    "    o spherical shells setup (computed via the SEAGen tool)\n"
                    "\n"
                    "Initial rotation about arbitrary axes is possible and set via the input parameter file (see the -f flag).\n"
                    "Additional pointmasses can be added (to act gravitationally on the colliding bodies during the SPH simulation). See the -N and -M flags.\n"
                    "As output (i.e., input file for the SPH code) only ASCII files are currently supported. See the -O flag to set the correct format.\n");
    
    fprintf(stdout, "\n\n    Usage: %s [Options]\n", programname);

    fprintf(stdout, "\nMain options:\n\n");
    fprintf(stdout, "        -?, -h                   displays this help message and exits\n");
    fprintf(stdout, "        -f input-file            specify input parameter file, default is 'spheres_ini.input'\n");
    fprintf(stdout, "        -o output-file           specify output file, default is 'impact.0000'\n");
#ifdef MILUPH
    fprintf(stdout, "        -m material-file         specify materialconstants file, default is 'materialconstants.data'\n");
    fprintf(stdout, "        -s scenario-file         specify scenario file for use as input file for miluph, default is 'materialscenario.data'\n");
#endif
#ifdef MILUPHCUDA
    fprintf(stdout, "        -m materialfile          specify materialconfiguration file, default is 'material.cfg'\n");
#endif
    fprintf(stdout, "        -G particle-geometry     specify geometric arrangement of particles (0/1/2):\n"
                    "                                     0 ... SIMPLE-CUBIC\n"
                    "                                     1 ... HEXAGONALLY CLOSE-PACKED (default)\n"
                    "                                     2 ... SPHERICAL SHELLS (approx. equal particle distances) using SEAGen\n");
    fprintf(stdout, "        -S source-directory      only required for SPHERICAL SHELL particle geometry:\n"
                    "                                     specify spheres_ini source directory, where 'run_SEAGen.py' and\n"
                    "                                     'SEAGen/' directory are located, default is '../spheres_ini/'\n");
    fprintf(stdout, "        -O output-format         specify output file format (input file for SPH simulation):\n"
                    "                                     0 ... HYDRO (default)\n"
                    "                                     1 ... SOLID WITHOUT FRAGMENTATION\n"
                    "                                     2 ... SOLID WITH FRAGMENTATION\n"
                    "                                     3 ... HYDRO WITHOUT DENSITY COLUMN\n");
    fprintf(stdout, "        -H                       Set this flag to calculate consistent hydrostatic structures as a method of relaxation for\n"
                    "                                 self-gravity dominated bodies. This is possible with the Tillotson, ANEOS, and ideal-gas EoS.\n"
                    "                                 If neither this nor '-R' are set, homogeneous values for the density, etc. are assumed by default.\n");
    
    fprintf(stdout, "\nAdditional options:\n\n");
    fprintf(stdout, "        -N coordinates-file      Set this flag for including additional pointmasses and input in cartesian coordinates.\n"
                    "                                 The coordinates-file has to provide positions, velocities and masses of all bodies, in the format:\n"
                    "                                     rows:  comments - projectile - target - additional-bodies\n"
                    "                                     columns:  x - y - z - vx - vy - vz - mass\n"
                    "                                 If used, all redundant data from the input parameter file is overwritten.\n"
                    "                                 By default, the additional pointmasses are represented by single SPH particles.\n"
                    "                                 Alternatively, they can be written to a separate file via the -M flag.\n");
    fprintf(stdout, "        -M pointmasses-file      If set, the additional pointmasses (-N flag) are not included as single SPH particles,\n"
                    "                                 but written to the specified file (readable by miluphcuda).\n");
    fprintf(stdout, "        -R                       Set this flag to use some radial profiles for projectile/target, given in the files specified in 'spheres_ini.h'.\n"
                    "                                 If neither this nor '-H' are set then homogeneous values for density, etc. are assumed by default.\n"
                    "                                 If used, all redundant data from the input parameter file (and from the -N flag, if applicable) is overwritten.\n");

    fprintf(stdout, "\nSpecial options (where the defaults work best in most situations):\n\n");
    fprintf(stdout, "        -x sml-factor            set smoothing-length-factor, where  sml = mean-particle-distance * sml-factor, default is 2.1\n");
    fprintf(stdout, "        -b                       Set up the whole arrangement defined by the -N flag (i.e., the coordinates-file) in a frame that is\n"
                    "                                 (initially) barycentric w.r.t. projectile + target alone (otherwise and by default the coordinate system\n"
                    "                                 defined via the coordinates-file is used). NOTE: This works not in combination with '-M'!\n");
    fprintf(stdout, "        -L lower-radius-bound    this times the uncompressed radius gives the lower starting value for the\n"
                    "                                 radius-iteration to find the hydrostatic structure, default is 0.1\n");
    fprintf(stdout, "        -U upper-radius-bound    this times the uncompressed radius gives the upper starting value for the\n"
                    "                                 radius-iteration to find the hydrostatic structure, default is 1.1\n");
    
    fprintf(stdout, "\n\n");
#ifdef MILUPH
    fprintf(stdout, "Output file format (for miluph):\n");
#endif
#ifdef MILUPHCUDA
    fprintf(stdout, "Output file format (for miluphcuda):\n");
#endif
    fprintf(stdout, "    coordinates (0-2)\n"
                    "    velocities (3-5)\n"
                    "    mass (6)\n"
                    "    density (omitted if output-format (-O flag) is set to 3)\n"
                    "    energy\n"
                    "    material type\n"
                    "    for SOLID WITHOUT FRAGMENTATION additionally:\n"
#ifdef MILUPH
                    "        plastic strain\n"
                    "        temperature\n"
#endif
                    "        S-tensor\n"
                    "    for SOLID WITH FRAGMENTATION additionally:\n"
                    "        number of flaws\n"
                    "        DIM-root of damage\n"
#ifdef MILUPH
                    "        plastic strain\n"
                    "        temperature\n"
#endif
                    "        S-tensor\n"
                    "        flaw activation thresholds\n\n");
}




void read_inputfile(char* infile, int* N, double* M, double* M_p, double* c_p_m, double* c_p_s, double* c_t_m, double* c_t_s, double* ini_vel, 
    double* impact_par, double* vel_vesc, double* impact_angle, double *ini_dist_fact, int* weibull_core, int* weibull_mantle, int* weibull_shell, char* core_eos, 
    char* mantle_eos, char* shell_eos, char* mat_core, char* mat_mantle, char* mat_shell,
    double *rot_period_p, double *rot_period_t, double *rot_axis_p, double *rot_axis_t)
// Reads the input parameter file 'infile'.
{
    FILE* ifl;
    int n_req = 0;    // number of those found values which are strictly required
    const int N_req = 19;   // number of overall strictly required values
    const int BUFSIZE = 256;
    char linebuf[BUFSIZE], name[BUFSIZE], value[BUFSIZE];    // strings for whole line and for name and value in non-comment lines
    
    if( (ifl = fopen(infile, "r")) == NULL )
        ERRORVAR("FILE ERROR! Cannot open '%s' for reading!\n", infile)
    
    while( fgets(linebuf, BUFSIZE, ifl) )    // reading line after line, fgets reads string of size BUFSIZE, returns pointer to first char and NULL at EOF
        if( (*linebuf) != '#' )
        {
            if ( sscanf(linebuf, "%s = %s\n", name, value) != 2 )    // sscanf returns the number of successfully assigned var
            {
                fprintf(stderr, "ERROR when reading input parameter file, probably wrong format!\n");
                fprintf(stderr, "Each line has to either start with '#' or be of the format 'name = value'.\n");
                exit(1);
            }
            else if( !strcmp(name,"N") )                { *N = atoi(value); n_req++; }    // strcmp returns 0 if equal
            else if( !strcmp(name,"M_tot") )            { *M = atof(value); n_req++; }
            else if( !strcmp(name,"M_proj") )            { *M_p = atof(value); n_req++; }
            else if( !strcmp(name,"mantle_proj") )        { *c_p_m = atof(value); n_req++; }
            else if( !strcmp(name,"shell_proj") )        { *c_p_s = atof(value); n_req++; }
            else if( !strcmp(name,"mantle_target") )    { *c_t_m = atof(value); n_req++; }
            else if( !strcmp(name,"shell_target") )        { *c_t_s = atof(value); n_req++; }
            else if( !strcmp(name,"ini_vel") )            { *ini_vel = atof(value); n_req++; }
            else if( !strcmp(name,"impact_par") )        { *impact_par = atof(value); n_req++; }
            else if( !strcmp(name,"vel_vesc") )         { *vel_vesc = atof(value); n_req++; }
            else if( !strcmp(name,"impact_angle") )     { *impact_angle = atof(value); n_req++; }
            else if( !strcmp(name,"ini_dist_fact") )     { *ini_dist_fact = atof(value); n_req++; }
            else if( !strcmp(name,"weibull_core") )        { *weibull_core = atoi(value); n_req++; }
            else if( !strcmp(name,"weibull_mantle") )    { *weibull_mantle = atoi(value); n_req++; }
            else if( !strcmp(name,"weibull_shell") )    { *weibull_shell = atoi(value); n_req++; }
            else if( !strcmp(name,"core_eos") )            { *core_eos = *value; n_req++; }
            else if( !strcmp(name,"mantle_eos") )        { *mantle_eos = *value; n_req++; }
            else if( !strcmp(name,"shell_eos") )        { *shell_eos = *value; n_req++; }
            else if( !strcmp(name,"core_mat") )            { strcpy(mat_core, value); n_req++; }
            else if( !strcmp(name,"mantle_mat") )        { strcpy(mat_mantle, value); n_req++; }
            else if( !strcmp(name,"shell_mat") )        { strcpy(mat_shell, value); n_req++; }
            else if( !strcmp(name,"proj_rot_period") )  { *rot_period_p = atof(value); }
            else if( !strcmp(name,"targ_rot_period") )  { *rot_period_t = atof(value); }
            else if( !strcmp(name,"proj_rot_axis_x") )  { rot_axis_p[0] = atof(value); }
            else if( !strcmp(name,"proj_rot_axis_y") )  { rot_axis_p[1] = atof(value); }
            else if( !strcmp(name,"proj_rot_axis_z") )  { rot_axis_p[2] = atof(value); }
            else if( !strcmp(name,"targ_rot_axis_x") )  { rot_axis_t[0] = atof(value); }
            else if( !strcmp(name,"targ_rot_axis_y") )  { rot_axis_t[1] = atof(value); }
            else if( !strcmp(name,"targ_rot_axis_z") )  { rot_axis_t[2] = atof(value); }
            else
                ERRORVAR("ERROR when reading input parameter file '%s', probably wrong format!\n", infile)
        }
    
    fclose(ifl);
    
    if( n_req != N_req )
        ERRORVAR3("ERROR when reading data from input parameter file '%s'. %d entries are strictly required and must be set, but found %d ... check 'spheres_ini.input' in the repo!\n", infile, N_req, n_req)
}




#ifdef MILUPHCUDA
void readMaterialConfiguration(char *matfile, material *mat, int weibull_it)
// Reads material.cfg (used in case of miluphcuda setups), for one material, specified via 'mat', which also takes the results.
// For materials except ideal gas it reads the (uncompressed) density, weibull parameters m and k (only if weibull_it == 1), the bulk modulus if
// eos != ANEOS (for computing the sound speed), the Tillotson eos parameters (only if eos == Tillotson) and ANEOS parameters (only if eos == ANEOS).
// For an ideal-gas it reads gamma, rho_0 and p_0 (values used usually for the top of an atmosphere), and the conversion factor from e to T.
{
    int i;
    double a;
    config_t config;    // configuration structure
    config_setting_t *all_materials, *one_material, *subset; // configuration settings structures
    int ID;
    const char *tempstring;     // allocation of storage is done automatically by libconfig (when running 'config_setting_lookup_string')
    const char *tmp_table_file;     // allocation of storage is done automatically by libconfig (when running 'config_setting_lookup_string')
    int n_mat;  // number of materials found in file
    int found_mat = FALSE;
    
    config_init(&config);
    if( !config_read_file(&config, matfile) )
        ERRORVAR("ERROR when reading materialconfiguration file '%s'!\n", matfile)
    all_materials = config_lookup(&config, "materials");
    if( all_materials == NULL )
        ERRORVAR("ERROR! Couldn't find materialconfiguration settings in '%s' ...\n", matfile)
    n_mat = config_setting_length(all_materials);
    
    fprintf(stdout, "--------------------------------\n");
    fprintf(stdout, "Searching for material '%s', material type %d, in materialconfiguration file '%s' ... ", mat->mat_name, mat->mat_type, matfile);
    
    // find material defined by mat:
    for(i=0; i<n_mat; i++)   // loop over all materials in file
    {
        one_material = config_setting_get_elem(all_materials, i);
        if( one_material == NULL )
            ERRORVAR("ERROR! Something's messed up with the settings in '%s' ...\n", matfile)
        
        if( !config_setting_lookup_int(one_material, "ID", &ID) )
            ERRORVAR("ERROR! Found material without ID in '%s' ...\n", matfile)
        if( ID == (mat->mat_type) )
        {
            found_mat = TRUE;
            fprintf(stdout, "Found.\n");
            break;
        }
    }
    if( found_mat == FALSE )
        ERRORVAR2("ERROR. Didn't find material with mat_type %d in '%s'!\n", mat->mat_type, matfile)
    
    config_setting_lookup_string(one_material, "name", &tempstring );
    if( strcmp(mat->mat_name, tempstring) )  // strcmp returns 0 if equal
        ERRORVAR4("ERROR. Material with mat_type %d should be '%s' but was found to be '%s' in '%s'!\n", mat->mat_type, mat->mat_name, tempstring, matfile)
    
    subset = config_setting_get_member(one_material, "eos");
    if( subset == NULL )
        ERRORVAR3("ERROR. Can't find eos parameters for material %d ('%s') in '%s' ...\n", ID, mat->mat_name, matfile)
    config_setting_lookup_int(subset, "type", &(mat->eos_type) );
    if( !( ((mat->eos)=='M' && (mat->eos_type)==EOS_TYPE_MURNAGHAN) || ((mat->eos)=='T' && (mat->eos_type)==EOS_TYPE_TILLOTSON) || ((mat->eos)=='A' && (mat->eos_type)==EOS_TYPE_ANEOS) || ((mat->eos)=='I' && (mat->eos_type)==EOS_TYPE_IDEAL_GAS) ) )
        ERRORVAR3("ERROR. Material with mat_type %d should have eos '%c' but has a contradicting eos type in '%s'!\n", mat->mat_type, mat->eos, matfile)
    
    if( (mat->eos) == 'M')
    {
        if( !config_setting_lookup_float(subset, "rho_0", &(mat->rho_0) ) )
            ERRORVAR2("ERROR! Didn't find 'rho_0' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "bulk_modulus", &a) )
            ERRORVAR2("ERROR! Didn't find 'bulk_modulus' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        mat->cs = sqrt(a/(mat->rho_0));    // sound speed = sqrt(bulk_modulus/rho_0)
        fprintf(stdout, "Found Murnaghan eos parameters:\nrho_0 = %g\nbulk_modulus = %g\ncs = %g (=sqrt(bulk_modulus/rho_0))\n", mat->rho_0, a, mat->cs);
    }
    else if( (mat->eos) == 'T' )
    {
        if( !config_setting_lookup_float(subset, "till_rho_0", &(mat->till.rho_0) ) )
            ERRORVAR2("ERROR! Didn't find 'till_rho_0' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        mat->rho_0 = mat->till.rho_0;
        if( !config_setting_lookup_float(subset, "till_A", &(mat->till.A) ) )
            ERRORVAR2("ERROR! Didn't find 'till_A' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "till_B", &(mat->till.B) ) )
            ERRORVAR2("ERROR! Didn't find 'till_B' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "till_E_0", &(mat->till.e_0) ) )
            ERRORVAR2("ERROR! Didn't find 'till_E_0' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "till_E_iv", &(mat->till.e_iv) ) )
            ERRORVAR2("ERROR! Didn't find 'till_E_iv' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "till_E_cv", &(mat->till.e_cv) ) )
            ERRORVAR2("ERROR! Didn't find 'till_E_cv' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "till_a", &(mat->till.a) ) )
            ERRORVAR2("ERROR! Didn't find 'till_a' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "till_b", &(mat->till.b) ) )
            ERRORVAR2("ERROR! Didn't find 'till_b' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "till_alpha", &(mat->till.alpha) ) )
            ERRORVAR2("ERROR! Didn't find 'till_alpha' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "till_beta", &(mat->till.beta) ) )
            ERRORVAR2("ERROR! Didn't find 'till_beta' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "rho_limit", &(mat->till.rho_limit) ) )
            ERRORVAR2("ERROR! Didn't find 'rho_limit' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "bulk_modulus", &a ) )
            ERRORVAR2("ERROR! Didn't find 'bulk_modulus' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        mat->cs = sqrt(a/(mat->till.rho_0));    // sound speed = sqrt(bulk_modulus/rho_0)
        fprintf(stdout, "Found Tillotson eos parameters:\nrho_0 = %g\nA = %g\nB = %g\ne_0 = %g\ne_iv = %g\ne_cv = %g\n", mat->till.rho_0, mat->till.A, mat->till.B, mat->till.e_0, mat->till.e_iv, mat->till.e_cv);
        fprintf(stdout, "a = %g\nb = %g\nalpha = %g\nbeta = %g\nrho_limit = %g\nbulk_modulus = %g\ncs = %g (=sqrt(bulk_modulus/rho_0))\n", mat->till.a, mat->till.b, mat->till.alpha, mat->till.beta, mat->till.rho_limit, a, mat->cs);
    }
    else if( (mat->eos) == 'A' )
    {
        if( !config_setting_lookup_float(subset, "aneos_rho_0", &(mat->aneos.rho_0) ) )
            ERRORVAR2("ERROR! Didn't find 'aneos_rho_0' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        mat->rho_0 = mat->aneos.rho_0;
        if( !config_setting_lookup_string(subset, "table_path", &tmp_table_file ) )
            ERRORVAR2("ERROR! Didn't find 'table_path' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        strcpy(mat->aneos.table_file, tmp_table_file);  // it's necessary to go via 'tmp_table_file' because 'config_destroy()' deallocates all memory that was used in 'config_setting_lookup_string' - even if it didn't allocate it ...
        if( !config_setting_lookup_int(subset, "n_rho", &(mat->aneos.n_rho) ) )
            ERRORVAR2("ERROR! Didn't find 'n_rho' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_int(subset, "n_e", &(mat->aneos.n_e) ) )
            ERRORVAR2("ERROR! Didn't find 'n_e' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "aneos_e_norm", &(mat->aneos.e_norm) ) )
            ERRORVAR2("ERROR! Didn't find 'aneos_e_norm' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "aneos_bulk_cs", &(mat->aneos.bulk_cs) ) )
            ERRORVAR2("ERROR! Didn't find 'aneos_bulk_cs' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        mat->cs = mat->aneos.bulk_cs;
        fprintf(stdout, "Found ANEOS eos parameters:\nrho_0 = %g\ntable_path = %s\nn_rho = %d\nn_e = %d\ne_norm = %g\ncs = %g (=aneos_bulk_cs)\n",
                mat->aneos.rho_0, mat->aneos.table_file, mat->aneos.n_rho, mat->aneos.n_e, mat->aneos.e_norm, mat->cs);
    }
    else if( (mat->eos) == 'I' )
    {
        if( !config_setting_lookup_float(subset, "polytropic_gamma", &(mat->ideal_gas.gamma) ) )
            ERRORVAR2("ERROR! Didn't find 'polytropic_gamma' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "ideal_gas_rho_0", &(mat->ideal_gas.rho_0) ) )
            ERRORVAR2("ERROR! Didn't find 'ideal_gas_rho_0' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        mat->rho_0 = mat->ideal_gas.rho_0;
        if( !config_setting_lookup_float(subset, "ideal_gas_p_0", &(mat->ideal_gas.p_0) ) )
            ERRORVAR2("ERROR! Didn't find 'ideal_gas_p_0' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "ideal_gas_conv_e_to_T", &(mat->ideal_gas.conv_e_to_T) ) )
            ERRORVAR2("ERROR! Didn't find 'ideal_gas_conv_e_to_T' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        // compute cs with rho_0 and p_0 for now ...
        mat->cs = sqrt( (mat->ideal_gas.gamma) * (mat->ideal_gas.p_0) / (mat->ideal_gas.rho_0) );
        mat->ideal_gas.polytropic_K = mat->ideal_gas.p_0/pow(mat->ideal_gas.rho_0,mat->ideal_gas.gamma);
        fprintf(stdout, "Found Ideal gas eos parameters:\ngamma = %g\nrho_0 = %g\np_0 = %g\nconv_e_to_T = %g\n", mat->ideal_gas.gamma, mat->ideal_gas.rho_0, mat->ideal_gas.p_0, mat->ideal_gas.conv_e_to_T);
        fprintf(stdout, "cs = %g (=sqrt(gamma*p_0/rho_0))\npolytropic_K = %g (=p_0/rho_0**gamma)\n", mat->cs, mat->ideal_gas.polytropic_K);
    }
    else
        ERRORVAR2("ERROR. EOS of material with material_type %d is '%c', which is not supported ...\n", mat->mat_type, mat->eos)
    
    if( weibull_it == TRUE )
    {
        if( !config_setting_lookup_float(subset, "W_M", &(mat->m) ) )
            ERRORVAR2("ERROR! Didn't find 'W_M' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        if( !config_setting_lookup_float(subset, "W_K", &(mat->k) ) )
            ERRORVAR2("ERROR! Didn't find 'W_K' for material with mat_type %d in '%s' ...\n", mat->mat_type, matfile)
        fprintf(stdout, "Found Weibull parameters:\nm = %g\nk = %g\n", mat->m, mat->k);
    }
    
    // clean up
    config_destroy(&config);
}
#endif  // end ifdef MILUPHCUDA




#ifdef MILUPHCUDA
void pasteSml(char *matfile, material *mat)
// Pastes the sml into all (relevant) materials in a (miluphcuda) materialconfiguration file.
{
    int i;
    config_t config;    // configuration structure
    config_setting_t *all_materials, *one_material, *one_setting; // configuration settings structures
    int ID;
    int n_mat;  // number of materials found in file
    
    config_init(&config);
    if( !config_read_file(&config, matfile) )
        ERRORVAR("ERROR when reading materialconfiguration file '%s'!\n", matfile)
    all_materials = config_lookup(&config, "materials");
    if( all_materials == NULL )
        ERRORVAR("ERROR! Couldn't find materialconfiguration settings in '%s' ...\n", matfile)
    n_mat = config_setting_length(all_materials);
    
    for(i=0; i<n_mat; i++)   // loop over all materials in file
    {
        one_material = config_setting_get_elem(all_materials, i);
        if( one_material == NULL )
            ERRORVAR("ERROR! Something's messed up with the settings in '%s' ...\n", matfile)
        
        // read ID of current material and locate sml setting:
        if( !config_setting_lookup_int(one_material, "ID", &ID) )
            ERRORVAR("ERROR! Found material without ID in '%s' ...\n", matfile)
        if( ID == mat[CORE].mat_type || ID == mat[MANTLE].mat_type || ID == mat[SHELL].mat_type )
            if( (one_setting = config_setting_lookup(one_material, "sml")) == NULL )
                ERRORVAR2("ERROR! Couldn't find 'sml' in material with mat_type %d in '%s'.\n", mat[ID].mat_type, matfile)
        
        if( ID == mat[CORE].mat_type )
        {
            if( !config_setting_set_float(one_setting, mat[CORE].sml) )
                ERRORVAR2("ERROR! Unknown problem encountered when writing sml for material with mat_type %d to '%s' ...\n", mat[CORE].mat_type, matfile)
        }
        else if( ID == mat[MANTLE].mat_type )
        {
            if( !config_setting_set_float(one_setting, mat[MANTLE].sml) )
                ERRORVAR2("ERROR! Unknown problem encountered when writing sml for material with mat_type %d to '%s' ...\n", mat[MANTLE].mat_type, matfile)
        }
        else if( ID == mat[SHELL].mat_type )
        {
            if( !config_setting_set_float(one_setting, mat[SHELL].sml) )
                ERRORVAR2("ERROR! Unknown problem encountered when writing sml for material with mat_type %d to '%s' ...\n", mat[SHELL].mat_type, matfile)
        }
    }
    
    // write to file again and clean up:
    config_set_option(&config, CONFIG_OPTION_SEMICOLON_SEPARATORS, FALSE);
    config_set_option(&config, CONFIG_OPTION_COLON_ASSIGNMENT_FOR_GROUPS, FALSE);
    config_set_option(&config, CONFIG_OPTION_ALLOW_SCIENTIFIC_NOTATION, TRUE);
    config_set_option(&config, CONFIG_OPTION_OPEN_BRACE_ON_SEPARATE_LINE, FALSE);
    if( !config_write_file(&config, matfile) )
        ERRORVAR("ERROR when writing materialconfiguration to file '%s'!\n", matfile)
    config_destroy(&config);
}
#endif  // end ifdef MILUPHCUDA




#ifdef MILUPH
void readMaterialConstants(FILE* f, material* mat, int weibull_it)
// Reads 'materialconstants.data' (used in case of miluph setups). Reads the (uncompressed) density, weibull parameters m and k (only if weibull_it == 1),
// the bulk modulus if not using ANEOS (for computing the sound speed), the Tillotson eos parameters (only if using Tillotson) and ANEOS parameters (only if
// using ANEOS) for one material from the materialconstants file. The desired material is specified in mat, which also takes the results.
{
    const int BUFSIZE = PATHLENGTH;
    char linebuf[BUFSIZE], name[BUFSIZE], value[BUFSIZE];    // strings for whole line and for name and value in non-comment lines
    int mat_found = FALSE, rho_0_found = FALSE, wm_found = FALSE, wk_found = FALSE, bm_found = FALSE;
    int till_found = 0;        // will be incremented whenever a further Tillotson eos parameter is found
    int aneos_found = 0;    // will be incremented whenever a further ANEOS eos parameter is found
    
    if( weibull_it != 1 )    // makes sure that no search for weibull parameters is conducted if weibulling is not desired
    {
        wm_found = wk_found = TRUE;
        mat->m = -1.0;
        mat->k = -1.0;
    }
    if( mat->eos != 'T' )   // makes sure that no search for Tillotson eos parameters is conducted if eos != Tillotson
        till_found = 10;
    if( mat->eos != 'A' )   // makes sure that no search for ANEOS eos parameters is conducted if eos != ANEOS
        aneos_found = 6;
    if( mat->eos == 'A' )   // makes sure that no search for the bulk modulus is conducted if eos == ANEOS (the sound speed is read directly for ANEOS)
        bm_found = TRUE;
    
    fprintf(stdout, "--------------------------------\n");
    fprintf(stdout, "Searching for material '%s', material type '%d', in material file ...\n", mat->mat_name, mat->mat_type);
    
    rewind(f);    // sets the file pointer back to the beginning of the file (important if this function is called more than once)
    
    while( fgets(linebuf, BUFSIZE, f) && !(rho_0_found && wm_found && wk_found && bm_found && (till_found==10) && (aneos_found==6)) )    // reading line after line, fgets reads string of size BUFSIZE, returns pointer to first char and NULL at EOF
        if ( (*linebuf) != '#' )
        {
            if ( sscanf(linebuf, "%s = %s\n", name, value) != 2 )    // sscanf returns the number of successfully assigned var
                ERRORTEXT("ERROR when reading material file, probably wrong format!\n")
            if ( !strcmp(name,"MAT") )    // TRUE if a new material is found, strcmp returns 0 if equal
            {
                fprintf(stdout, "New material '%s' found ... ", value);
                if (mat_found)    // if the right material was already found the loop should not reach another material
                    ERRORTEXT("ERROR when reading material file, probably wrong format!\n")
                else if ( strcmp(value,mat->mat_name) )    // FALSE if new material == desired material, strcmp returns 0 if equal
                    fprintf(stdout, "and ignored.\n");
                else
                {
                    fprintf(stdout, "and identified with material type %d.\n", mat->mat_type);
                    fprintf(stdout, "Searching for (uncompressed) density%s%s%s%s ... ", ( (weibull_it==1) ? ", weibull parameters" : "" ), ( (mat->eos=='T') ? ", Tillotson eos parameters" : "" ), ( (mat->eos=='A') ? ", ANEOS parameters" : "" ), ( (mat->eos!='A') ? " and bulk modulus (for calculating sound speed)" : "" ) );
                    mat_found = TRUE;
                }
            }
            else if (mat_found)
            {
                if ( !strcmp(name,"till_rho_0") && ( mat->eos == 'T' ) )
                {
                    mat->rho_0 = mat->till.rho_0 = atof(value);
                    rho_0_found = TRUE;
                    till_found++;
                }
                else if ( !strcmp(name,"aneos_rho_0") && ( mat->eos == 'A' ) )
                {
                    mat->rho_0 = mat->aneos.rho_0 = atof(value);
                    rho_0_found = TRUE;
                    aneos_found++;
                }
                else if ( !strcmp(name,"rho_0") && ( mat->eos == 'M' ) )
                {
                    mat->rho_0 = atof(value);
                    rho_0_found = TRUE;
                }
                else if ( !strcmp(name,"W_M") )
                {
                    mat->m = atof(value);
                    wm_found = TRUE;
                }
                else if ( !strcmp(name,"W_K") )
                {
                    mat->k = atof(value);
                    wk_found = TRUE;
                }
                else if ( !strcmp(name,"bulk_modulus") && ( mat->eos != 'A' ) )
                {
                    mat->cs = atof(value);    // use cs as temporary storage for the bulk modulus
                    bm_found = TRUE;
                }
                else if ( !strcmp(name,"till_E_0") && ( mat->eos == 'T' ) )
                {
                    mat->till.e_0 = atof(value);
                    till_found++;
                }
                else if ( !strcmp(name,"till_E_iv") && ( mat->eos == 'T' ) )
                {
                    mat->till.e_iv = atof(value);
                    till_found++;
                }
                else if ( !strcmp(name,"till_E_cv") && ( mat->eos == 'T' ) )
                {
                    mat->till.e_cv = atof(value);
                    till_found++;
                }
                else if ( !strcmp(name,"till_a") && ( mat->eos == 'T' ) )
                {
                    mat->till.a = atof(value);
                    till_found++;
                }
                else if ( !strcmp(name,"till_b") && ( mat->eos == 'T' ) )
                {
                    mat->till.b = atof(value);
                    till_found++;
                }
                else if ( !strcmp(name,"till_A") && ( mat->eos == 'T' ) )
                {
                    mat->till.A = atof(value);
                    till_found++;
                }
                else if ( !strcmp(name,"till_B") && ( mat->eos == 'T' ) )
                {
                    mat->till.B = atof(value);
                    till_found++;
                }
                else if ( !strcmp(name,"till_alpha") && ( mat->eos == 'T' ) )
                {
                    mat->till.alpha = atof(value);
                    till_found++;
                }
                else if ( !strcmp(name,"till_beta") && ( mat->eos == 'T' ) )
                {
                    mat->till.beta = atof(value);
                    till_found++;
                }
                else if ( !strcmp(name,"table_path") && ( mat->eos == 'A' ) )
                {
                    strcpy(mat->aneos.table_file, value);
                    aneos_found++;
                }
                else if ( !strcmp(name,"n_rho") && ( mat->eos == 'A' ) )
                {
                    mat->aneos.n_rho = atoi(value);
                    aneos_found++;
                }
                else if ( !strcmp(name,"n_e") && ( mat->eos == 'A' ) )
                {
                    mat->aneos.n_e = atoi(value);
                    aneos_found++;
                }
                else if ( !strcmp(name,"aneos_bulk_cs") && ( mat->eos == 'A' ) )
                {
                    mat->cs = mat->aneos.bulk_cs = atof(value);
                    aneos_found++;
                }
                else if ( !strcmp(name,"aneos_e_norm") && ( mat->eos == 'A' ) )
                {
                    mat->aneos.e_norm = atof(value);
                    aneos_found++;
                }
            }
        }
    if ( rho_0_found && wm_found && wk_found && bm_found && (till_found == 10) && (aneos_found == 6) )
        fprintf(stdout, "successfully completed.\n");
    else
        ERRORTEXT("ERROR when reading material file, probably wrong format!\n")
    if( mat->eos != 'A' )
        mat->cs = sqrt((mat->cs)/(mat->rho_0));    // sound speed = sqrt(bulk_modulus/rho_0)
}
#endif  // end ifdef MILUPH




#ifdef MILUPH
void write_scenario_data(FILE* f, material* mat, int nmat)
// Writes scenario data to file *f for use as input file for miluph. The file has to be opened/closed externally.
// The function can be used generally to write data for nmat different materials, passed via the address *mat of the struct array.
{
    fprintf(f, "# These are material-dependent scenario data for use as input file for miluph.\n#\n");
    fprintf(f, "# DO NOT CHANGE THE DESCRIPTORS!\n#\n#");
    while (nmat > 0)
    {
        fprintf(f, "\nMTYPE = %i\nMAT = %s\nEOS = %c\n", mat->mat_type, mat->mat_name, mat->eos);
        fprintf(f, "sml = %e\n# Artificial viscosity parameters:\nalpha = %e\nbeta = %e\n#", mat->sml, mat->alpha, mat->beta);
        mat++;
        nmat--;
    }
}
#endif




void write_outfile(FILE* f, particle* p, int n, int OutputMode)
// Writes the whole particle vector p (n components) to file f.
{
    int i,j,k;
    
    fprintf(stdout, "--------------------------------\n");
#ifdef MILUPH
    fprintf(stdout, "Writing %d particles to the output file (for use with miluph ", n);
#endif
#ifdef MILUPHCUDA
    fprintf(stdout, "Writing %d particles to the output file (for use with miluphcuda ", n);
#endif
    if( OutputMode == 0 )
        fprintf(stdout, "in HYDRO runs!) ... ");
    else if( OutputMode == 1 )
        fprintf(stdout, "in SOLID runs WITHOUT FRAGMENTATION!) ... ");
    else if( OutputMode == 2 )
        fprintf(stdout, "in SOLID runs WITH FRAGMENTATION!) ... ");
    else if( OutputMode == 3 )
        fprintf(stdout, "in HYDRO runs, but OMITTING THE DENSITY COLUMN!) ... ");
    else
        ERRORVAR("ERROR. Invalid 'OutputMode' = %d ...\n", OutputMode)
    
    for( i=0; i<n; i++)
    {
        for( j=0; j<DIM; j++)
            fprintf(f, "%.16le\t", p[i].x[j]);
        for( j=0; j<DIM; j++)
            fprintf(f, "%.16le\t", p[i].v[j]);
        
        if( OutputMode != 3 )
            fprintf(f, "%.16le\t%.16le\t%.16le\t%d", p[i].mass, p[i].rho, p[i].e, p[i].mat_type);
        else   // omit density column
            fprintf(f, "%.16le\t%.16le\t%d", p[i].mass, p[i].e, p[i].mat_type);
        
        if( OutputMode == 2 )
            fprintf(f, "\t%d\t%.16le", p[i].flaws.n_flaws, p[i].damage);
#ifdef MILUPH
        if( OutputMode == 1 || OutputMode == 2 )
            fprintf(f, "\t%.16le\t%.16le", p[i].plastic_strain, p[i].temp);
#endif
        if( OutputMode == 1 || OutputMode == 2 )
        {
            for( j=0; j<DIM; j++)
                for( k=0; k<DIM; k++)
                    fprintf(f, "\t%e", p[i].S[j][k]);
        }
        if( OutputMode == 2 )
        {
            for( j=0; j<p[i].flaws.n_flaws; j++)
                fprintf(f, "\t%e", p[i].flaws.act_thr[j]);
        }
        fprintf(f, "\n");
    }
    fprintf(stdout, "Done.\n");
}




void allocate_ANEOS_table_memory(material *mat)
// Allocates memory for ANEOS lookup-table for material 'mat'.
{
    int i;
    int n_rho = mat->aneos.n_rho;
    int n_e = mat->aneos.n_e;
    
    // rho
    if( (mat->aneos.rho = (double*)malloc(n_rho*sizeof(double))) == NULL )
        ERRORVAR("ERROR during memory allocation for 'rho' in ANEOS lookup table in '%s'!\n", mat->aneos.table_file)
    // e
    if( (mat->aneos.e = (double*)malloc(n_e*sizeof(double))) == NULL )
        ERRORVAR("ERROR during memory allocation for 'e' in ANEOS lookup table in '%s'!\n",mat->aneos.table_file)
    // p
    if( (mat->aneos.p = (double**)malloc(n_rho*sizeof(double*))) == NULL )
        ERRORVAR("ERROR during memory allocation for 'p' in ANEOS lookup table in '%s'!\n",mat->aneos.table_file)
    for(i=0; i<n_rho; i++)
        if( (mat->aneos.p[i] = (double*)malloc(n_e*sizeof(double))) == NULL )
            ERRORVAR("ERROR during memory allocation for 'p' in ANEOS lookup table in '%s'!\n",mat->aneos.table_file)
    // T
    if( (mat->aneos.T = (double**)malloc(n_rho*sizeof(double*))) == NULL )
        ERRORVAR("ERROR during memory allocation for 'T' in ANEOS lookup table in '%s'!\n",mat->aneos.table_file)
    for(i=0; i<n_rho; i++)
        if( (mat->aneos.T[i] = (double*)malloc(n_e*sizeof(double))) == NULL )
            ERRORVAR("ERROR during memory allocation for 'T' in ANEOS lookup table in '%s'!\n",mat->aneos.table_file)
    // cs
    if( (mat->aneos.cs = (double**)malloc(n_rho*sizeof(double*))) == NULL )
        ERRORVAR("ERROR during memory allocation for 'cs' in ANEOS lookup table in '%s'!\n",mat->aneos.table_file)
    for(i=0; i<n_rho; i++)
        if( (mat->aneos.cs[i] = (double*)malloc(n_e*sizeof(double))) == NULL )
            ERRORVAR("ERROR during memory allocation for 'cs' in ANEOS lookup table in '%s'!\n",mat->aneos.table_file)
    // entropy
    if( (mat->aneos.entropy = (double**)malloc(n_rho*sizeof(double*))) == NULL )
        ERRORVAR("ERROR during memory allocation for 'entropy' in ANEOS lookup table in '%s'!\n",mat->aneos.table_file)
    for(i=0; i<n_rho; i++)
        if( (mat->aneos.entropy[i] = (double*)malloc(n_e*sizeof(double))) == NULL )
            ERRORVAR("ERROR during memory allocation for 'entropy' in ANEOS lookup table in '%s'!\n",mat->aneos.table_file)
    // phase-flag
    if( (mat->aneos.phase_flag = (int**)malloc(n_rho*sizeof(int*))) == NULL )
        ERRORVAR("ERROR during memory allocation for 'phase_flag' in ANEOS lookup table in '%s'!\n",mat->aneos.table_file)
    for(i=0; i<n_rho; i++)
        if( (mat->aneos.phase_flag[i] = (int*)malloc(n_e*sizeof(int))) == NULL )
            ERRORVAR("ERROR during memory allocation for 'phase_flag' in ANEOS lookup table in '%s'!\n",mat->aneos.table_file)
}




void free_ANEOS_table_memory(material *mat)
// Frees memory for ANEOS lookup-table for material 'mat'.
{
    int i;
    int n_rho = mat->aneos.n_rho;
    
    free(mat->aneos.rho);
    free(mat->aneos.e);
    for(i=0; i<n_rho; i++)
        free(mat->aneos.p[i]);
    free(mat->aneos.p);
    for(i=0; i<n_rho; i++)
        free(mat->aneos.T[i]);
    free(mat->aneos.T);
    for(i=0; i<n_rho; i++)
        free(mat->aneos.cs[i]);
    free(mat->aneos.cs);
    for(i=0; i<n_rho; i++)
        free(mat->aneos.entropy[i]);
    free(mat->aneos.entropy);
    for(i=0; i<n_rho; i++)
        free(mat->aneos.phase_flag[i]);
    free(mat->aneos.phase_flag);
}




void load_ANEOS_table(material *mat)
// Loads ANEOS lookup-table for one material from file. Memory has to be allocated and freed externally!
{
    int i,j;
    FILE *f;
    int n_rho = mat->aneos.n_rho;
    int n_e = mat->aneos.n_e;
    
    // open file containing ANEOS lookup table
    if ( (f = fopen(mat->aneos.table_file, "r")) == NULL )
        ERRORVAR("FILE ERROR! Cannot open '%s' for reading!\n", mat->aneos.table_file)
    
    // read rho and e (vectors) and p, T, cs, entropy and the phase-flag (matrices) from file
    for(i=0; i<3; i++)
        fscanf(f, "%*[^\n]\n"); //ignore first 3 lines
    if ( fscanf(f, "%le %le %le %le %le %le %d%*[^\n]\n", mat->aneos.rho, mat->aneos.e, &(mat->aneos.p[0][0]), &(mat->aneos.T[0][0]), &(mat->aneos.cs[0][0]), &(mat->aneos.entropy[0][0]), &(mat->aneos.phase_flag[0][0]) ) != 7 )
        ERRORVAR("ERROR! Something's wrong with the ANEOS lookup table in '%s'\n",mat->aneos.table_file)
    for(j=1; j<n_e; j++)
        fscanf(f, "%*le %le %le %le %le %le %d%*[^\n]\n", &(mat->aneos.e[j]), &(mat->aneos.p[0][j]), &(mat->aneos.T[0][j]), &(mat->aneos.cs[0][j]), &(mat->aneos.entropy[0][j]), &(mat->aneos.phase_flag[0][j]) );
    for(i=1; i<n_rho; i++)
    {
        fscanf(f, "%le %*le %le %le %le %le %d%*[^\n]\n", &(mat->aneos.rho[i]), &(mat->aneos.p[i][0]), &(mat->aneos.T[i][0]), &(mat->aneos.cs[i][0]), &(mat->aneos.entropy[i][0]), &(mat->aneos.phase_flag[i][0]) );
        for(j=1; j<n_e; j++)
            fscanf(f, "%*le %*le %le %le %le %le %d%*[^\n]\n", &(mat->aneos.p[i][j]), &(mat->aneos.T[i][j]), &(mat->aneos.cs[i][j]), &(mat->aneos.entropy[i][j]), &(mat->aneos.phase_flag[i][j]) );
    }
    
    fclose(f);
}

