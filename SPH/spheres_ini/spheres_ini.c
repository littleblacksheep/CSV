/* For generation of the initial particle distribution consisting of two colliding spheres for the SPH codes miluph and miluphcuda.
 * 
 * Both spheres can in general consist of a core, a mantle and a shell of different materials.
 * Alternatively they can also be set up following some given radial profiles.
 * 
 * A relaxation technique, which calculates the pyhsically correct hydrostatic structure (adiabatic compression) and sets 
 * the particle's characteristics accordingly, is available (along with the Tillotson/ANEOS/ideal-gas eos).
 * 
 * The particles can be set up either in an equally-spaced lattice (simple-cubic or hexagonally close-packed),
 * or in spherical shells (produced via an interface to SEAGen).
 * 
 * All materials can be optionally weibulled (distribute flaws following
 * the Weibull distribution for use in the Grady-Kipp fragmentation model).
 * 
 * Furthermore it is possible to include additional bodies/pointmasses (which then act gravitationally during the SPH simulation).
 * 
 * All units are SI.
 * 
 * Christoph Burger 07/Nov/2020
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <Python.h>

#include "spheres_ini.h"
#include "io.h"
#include "geometry.h"
#include "hydrostruct.h"

#ifdef MILUPHCUDA
#include <libconfig.h>
#endif


int main(int argc, char* argv[])
{
    int i,j,k;
    double tmp;
    // set default filenames:
    char infile[PATHLENGTH] = "spheres_ini.input";
    char outfile[PATHLENGTH] = "impact.0000";
    char source_dir[PATHLENGTH] = "../spheres_ini/";
#ifdef MILUPH
    char matfile[PATHLENGTH] = "materialconstants.data";
    char scenfile[PATHLENGTH] = "materialscenario.data";
    FILE *sfl, *mfl;
#endif
#ifdef MILUPHCUDA
    char matfile[PATHLENGTH] = "material.cfg";
#endif
    char coordfile[PATHLENGTH];
    char pointmassesfile[PATHLENGTH];
    FILE *ofl, *cfl, *tfl, *pfl;
    char t_structfile[PATHLENGTH] = "target.structure";
    char p_structfile[PATHLENGTH] = "projectile.structure";
    double sml_factor = 2.1;    // default value for sml factor
    int N_des, N;    // desired (input parameter file) and actual/final (after creating the particles in the spheres) total particle numbers
    int N_p_des, N_t_des;   // based on volume ratio of proj/targ (all SPH particles have ~equal volumes)
    int N_p, N_t, N_p_c, N_t_c, N_p_m, N_t_m;    // actual/final particle numbers of proj/targ and their respective cores and mantles
    double M_des, M_p_des, M_t_des, M_p_c_des, M_p_m_des, M_p_s_des, M_t_c_des, M_t_m_des, M_t_s_des;    // initially desired masses, p/t for proj/targ, c/m/s for core/mantle/shell
    double C_p_m_des, C_t_m_des, C_p_s_des, C_t_s_des;    // desired mantle/shell mass fractions of proj/targ
    double R_p_uncomp, R_t_uncomp, R_p_c_uncomp, R_t_c_uncomp, R_p_m_uncomp, R_t_m_uncomp;    // uncompressed radii, p/t for proj/targ, c/m for core/mantle
    double R_p_des, R_t_des, R_p_c_des, R_t_c_des, R_p_m_des, R_t_m_des;    // desired radii (i.e. before actually building the spheres' particles), p/t for proj/targ, c/m for core/mantle
    double M, M_p, M_t, M_p_c, M_t_c, M_p_m, M_t_m, R_p, R_t, R_p_c, R_t_c, R_p_m, R_t_m;    // actual/final values (after building spheres / setting hydrostatic structure)
    // hydrostatic internal structure (p/t for proj/targ, c/m/s for core/mantle/shell), where r increases with the array index:
    int_struct_point int_struct_p_c[NSTEPS+1], int_struct_p_m[NSTEPS+1], int_struct_p_s[NSTEPS+1], int_struct_t_c[NSTEPS+1], int_struct_t_m[NSTEPS+1], int_struct_t_s[NSTEPS+1];
    const double eps6 = 1.0e-6;
    material mat[NMAT];
    material_fp mat_fp[NMAT];
    double M_particle_p[NMAT], M_particle_t[NMAT];    // SPH particle masses for the different materials (core/mantle/shell) and proj/targ in the case of homogeneous densities (no hydrostatic structures)
    // either one of the following 2 pairs will be used for setting the impact geometry (but not both):
    double ini_vel = -1.0, impact_par = -1.0;    // initial velocity (in y-direction) and impact parameter
    double vel_vesc = -1.0, impact_angle = -1.0;    // v/v_esc and impact angle (in deg!), both at "touching ball" distance
    int vel_vesc_angle = FALSE; // indicates which of the above pairs is used for setting the impact geometry
    double ini_pos_p[DIM];    // initial position of projectile (where the target is at the origin)
    double ini_dist_fact = -1.0, ini_dist, des_ini_dist;
    double impact_vel_abs;   // relative speed at touching ball distance
    particle* p;    // main particle array
    int weibull_core, weibull_mantle, weibull_shell;    // stores choice from input parameter file whether weibulling of core/mantle/shell material is desired (0/1)
    double mpd;    // mean particle distance: for SC and HCP = distance to direct neighbours in the lattice (constant throughout all materials and bodies); for SEAGen setup = max(mpd_p,mpd_t)
    double mpd_p, mpd_t;    // mpd of proj/targ for SEAGen setups, computed based on the volume per particle (the sphere's volume divided by N) and the (hypothetical) particle distance in a HCP lattice, which should provide a reasonable representative (rather higher than lower) value, computed as mpd_p/_t = cbrt( sqrt(2)*V_particle_p/_t )
    double V_particle_p, V_particle_t;  // volume per particle in proj/targ; they are equal for SC and HCP, but may differ to some degree for spherical shell setups
    double r2;    // squared distance to the origin
    double baryc_x[DIM], baryc_v[DIM];    // pos and vel of the barycenter in the lab frame (where the bodies are initially placed, with the target at the origin and at rest)
    double proj_pos_final[DIM], proj_vel_final[DIM], targ_pos_final[DIM], targ_vel_final[DIM];
    double V_p_c_uncomp, V_p_m_uncomp, V_p_s_uncomp, V_t_c_uncomp, V_t_m_uncomp, V_t_s_uncomp;    // uncompressed volumes, p/t for projectile/target, c/m/s for core/mantle/shell
    int N_input = FALSE;
    int M_output = FALSE;
    int b_flag = FALSE;
    N_input_coord* N_input_data;
    int N_bodies = 2;
    double ini_vel_vec[DIM];
    double e_temp;
    int ParticleGeometry = 1;   // defaults to HCP
    int OutputMode = 0;     // defaults to HYDRO
    int hydrostructFlag = FALSE;
    double N_input_impact_angle, N_input_impact_vel_vesc, N_input_impact_vel_abs;   // values computed from relative two-body orbit of proj+targ alone when in N_input mode ('N_input_impact_angle' in rad!)
    double p_rot_period = -1.0, t_rot_period = -1.0;    // rotation periods of proj/targ - negative values mean no rotation
    double p_rot_axis[DIM], t_rot_axis[DIM];    // rotation axes of proj/targ
    double p_omega[DIM], t_omega[DIM];  // angular velocity vectors of proj/targ
    int aneos_i_rho, aneos_i_e;
    double aneos_T, aneos_cs, aneos_entropy;
    int aneos_phase_flag;
    int allocated_ANEOS_mem_core_flag = FALSE, allocated_ANEOS_mem_mantle_flag = FALSE, allocated_ANEOS_mem_shell_flag = FALSE;
    int useProfilesFlag = FALSE;
    radial_profile_data *profile_target, *profile_projectile;
    FILE *p_prof_file, *t_prof_file;
    int p_profile_no_points = 0, t_profile_no_points = 0;
    double struct_r_low = 0.1;
    double struct_r_high = 1.1;
    
    
// seed random number generator (once for whole program)
    srand( (unsigned)time(NULL) );
    
    
// process command line options
    while ( (i = getopt(argc, argv, "?hG:S:O:HL:U:Rf:o:m:s:x:N:M:b")) != -1 ) // int-representations of command line options are successively saved in i
        switch((char)i)
        {
            case '?':
                help(*argv);
                exit(0);
            case 'h':
                help(*argv);
                exit(0);
            case 'G':
                ParticleGeometry = atoi(optarg);
                if( ParticleGeometry != 0  &&  ParticleGeometry != 1  &&  ParticleGeometry != 2 )
                    ERRORTEXT("ERROR. Invalid choice of particle geometry ('-G' flag). Choose either 0, 1, or 2.\n")
                break;
            case 'S':
                strncpy(source_dir, optarg, PATHLENGTH);
                break;
            case 'O':
                OutputMode = atoi(optarg);
                if( OutputMode != 0  &&  OutputMode != 1  &&  OutputMode != 2  &&  OutputMode != 3 )
                    ERRORTEXT("ERROR. Invalid choice of output mode ('-O' flag). Choose either 0, 1, 2, or 3.\n")
                break;
            case 'H':
                hydrostructFlag = TRUE;
                break;
            case 'L':
                struct_r_low = atof(optarg);
                break;
            case 'U':
                struct_r_high = atof(optarg);
                break;
            case 'R':
                useProfilesFlag = TRUE;
                break;
            case 'f':
                strncpy(infile, optarg, PATHLENGTH);
                break;
            case 'o':
                strncpy(outfile, optarg, PATHLENGTH);
                break;
            case 'm':
                strncpy(matfile, optarg, PATHLENGTH);
                break;
#ifdef MILUPH
            case 's':
                strncpy(scenfile, optarg, PATHLENGTH);
                break;
#endif
            case 'x':
                sml_factor = atof(optarg);
                break;
            case 'N':
                N_input = TRUE;
                strncpy(coordfile, optarg, PATHLENGTH);
                break;
            case 'M':
                M_output = TRUE;
                strncpy(pointmassesfile, optarg, PATHLENGTH);
                break;
            case 'b':
                b_flag = TRUE;
                break;
            default:
                help(*argv);
                exit(1);
        }
    // checks on cmd-line choices
    if( hydrostructFlag && useProfilesFlag )
        ERRORTEXT("ERROR. Cmd-line flags '-H' and '-R' were both set. You can't use them both at the same time ... try again!\n")
    if( N_input == FALSE  &&  M_output == TRUE )
        ERRORTEXT("ERROR. Cmd-line flag '-M' was set but not '-N' ... try again!\n")
    if( M_output && b_flag )
        ERRORTEXT("ERROR. Cmd-line flag '-M' is incompatible with '-b' ... try again!\n")
    
    
// read values from input file
    fprintf(stdout, "--------------------------------\n");
    fprintf(stdout, "Reading input parameter file '%s' ... ", infile);
    read_inputfile(infile, &N_des, &M_des, &M_p_des, &C_p_m_des, &C_p_s_des, &C_t_m_des, &C_t_s_des, &ini_vel, &impact_par, &vel_vesc, 
                   &impact_angle, &ini_dist_fact, &weibull_core, &weibull_mantle, &weibull_shell, &(mat[CORE].eos), &(mat[MANTLE].eos), &(mat[SHELL].eos), 
                   mat[CORE].mat_name, mat[MANTLE].mat_name, mat[SHELL].mat_name, &p_rot_period, &t_rot_period, p_rot_axis, t_rot_axis);
    fprintf(stdout, "Done.\n");
    
    if( vel_vesc >= 0.0  &&  impact_angle >= 0.0 )
        vel_vesc_angle = TRUE;
    
    // run some consistency checks on read data
    if( M_des < M_p_des  &&  !useProfilesFlag )
        ERRORVAR("ERROR. Found 'M_tot' < 'M_proj' in input parameter file '%s'. That's not possible.\n", infile)
    if( ini_dist_fact < 1.0 )
        ERRORVAR("ERROR. 'ini_dist_fact' was found to be < 1.0 ... that's not a good idea. Check '%s'!\n", infile)
    if( hydrostructFlag )
        if ( (mat[CORE].eos != 'T' && mat[CORE].eos != 'A' && mat[CORE].eos != 'I') || (mat[MANTLE].eos != 'T' && mat[MANTLE].eos != 'A' && mat[MANTLE].eos != 'I') || (mat[SHELL].eos != 'T' && mat[SHELL].eos != 'A' && mat[SHELL].eos != 'I') )
            ERRORVAR("ERROR. Computation of the hydrostatic structure is implemented only along with Tillotson/ANEOS/ideal gas EoS. Check '%s' ...\n", infile)
    if( p_rot_period == 0.0  ||  t_rot_period == 0.0 )
        ERRORVAR("ERROR. Check rotation periods in '%s'. Zero means infinitely fast rotation. Not implemented yet.\n", infile)
    
    // mark quantities that are not used any further
    if( useProfilesFlag )
        M_des = M_p_des = C_p_m_des = C_p_s_des = C_t_m_des = C_t_s_des = -1.0;
    
    
// read coordinates file if in N_input mode
    if( N_input == TRUE )
    {
        fprintf(stdout, "--------------------------------\n");
        fprintf(stdout, "Reading coordinates file '%s' ... ", coordfile);
        if ( (cfl = fopen(coordfile, "r")) == NULL )
            ERRORVAR("FILE ERROR! Cannot open '%s' for reading!\n", coordfile)
        if( ( N_input_data = (N_input_coord*)malloc(sizeof(N_input_coord)) ) == NULL )
            ERRORTEXT("ERROR during memory allocation!\n")
        fscanf(cfl, "%*[^\n]\n");   // ignore first line
        i = N_bodies = 0;
        while( fscanf(cfl, "%le %le %le %le %le %le %le%*[^\n]\n", &(N_input_data[i].x[0]), &(N_input_data[i].x[1]), &(N_input_data[i].x[2]), &(N_input_data[i].v[0]), &(N_input_data[i].v[1]), &(N_input_data[i].v[2]), &(N_input_data[i].mass) ) == 7 )
        {
            N_bodies++;
            i++;
            if( ( N_input_data = (N_input_coord*)realloc(N_input_data,(N_bodies+1)*sizeof(N_input_coord)) ) == NULL )
                ERRORTEXT("ERROR during memory allocation!\n")
        }
        fclose(cfl);
        if( N_bodies < 2 )
            ERRORVAR("ERROR! Too little bodies in coordinates file '%s'!\n", coordfile)
        else
            fprintf(stdout, "found %d bodies.\n", N_bodies);
        
        // overwrite respective values from input parameter file
        M_des = N_input_data[0].mass + N_input_data[1].mass;
        M_p_des = N_input_data[0].mass;
        ini_vel = impact_par = vel_vesc = impact_angle = -1.0;
        ini_dist_fact = -1.0;
        
        // overwrite total and projectile mass if profiles are read from file
        if( useProfilesFlag )
            M_des = M_p_des = -1.0;
    }
    
    
// read radial profile(s) from files if desired
    if( useProfilesFlag )
    {
        fprintf(stdout, "--------------------------------\n");
        fprintf(stdout, "Reading file(s) containing radial profiles ...\n");
        
        if( PROFILE_FILE_PROJ != 0 )    // there is a projectile
        {
            if ( (p_prof_file = fopen(PROFILE_FILE_PROJ,"r")) == NULL )
                ERRORVAR("FILE ERROR! Cannot open '%s' for reading!\n", PROFILE_FILE_PROJ)
            
            if( ( profile_projectile = (radial_profile_data*)malloc(sizeof(radial_profile_data)) ) == NULL )
                ERRORTEXT("ERROR during memory allocation!\n")
            fscanf(p_prof_file, "%*[^\n]\n");   // ignore first line
            i = 0;
            while( fscanf(p_prof_file, "%le %le %le%*[^\n]\n", &(profile_projectile[i].r), &(profile_projectile[i].rho), &(profile_projectile[i].e) ) == 3 )
            {
                i++;
                if( ( profile_projectile = (radial_profile_data*)realloc(profile_projectile,(i+1)*sizeof(radial_profile_data)) ) == NULL )
                    ERRORTEXT("ERROR during memory allocation!\n")
            }
            
            p_profile_no_points = i;
            fprintf(stdout, "Found %d datapoints for the projectile profile in '%s'.\n", p_profile_no_points, PROFILE_FILE_PROJ);
            fclose(p_prof_file);
        }
        if( PROFILE_FILE_TARG != 0 )    // there is a target
        {
            if ( (t_prof_file = fopen(PROFILE_FILE_TARG,"r")) == NULL )
                ERRORVAR("FILE ERROR! Cannot open '%s' for reading!\n", PROFILE_FILE_TARG)
            
            if( ( profile_target = (radial_profile_data*)malloc(sizeof(radial_profile_data)) ) == NULL )
                ERRORTEXT("ERROR during memory allocation!\n")
            fscanf(t_prof_file, "%*[^\n]\n");   // ignore first line
            i = 0;
            while( fscanf(t_prof_file, "%le %le %le%*[^\n]\n", &(profile_target[i].r), &(profile_target[i].rho), &(profile_target[i].e) ) == 3 )
            {
                i++;
                if( ( profile_target = (radial_profile_data*)realloc(profile_target,(i+1)*sizeof(radial_profile_data)) ) == NULL )
                    ERRORTEXT("ERROR during memory allocation!\n")
            }
            
            t_profile_no_points = i;
            fprintf(stdout, "Found %d datapoints for the target profile in '%s'.\n", t_profile_no_points, PROFILE_FILE_TARG);
            fclose(t_prof_file);
        }
    }
    
    
// read (uncompressed) densities, weibull parameters (if required), bulk modulus (for calculating sound speed),
// Tillotson EoS parameters (if EoS is Tillotson), and ANEOS parameters (if EoS is ANEOS) for all relevant materials from material file,
// or assign them for ideal gas for miluph, or also read them for ideal gas from the material file for miluphcuda
    mat[CORE].mat_type = MATTYPECORE;
    mat[MANTLE].mat_type = MATTYPEMANTLE;
    mat[SHELL].mat_type = MATTYPESHELL;
#ifdef MILUPH
    if ( (mfl = fopen(matfile,"r")) == NULL )   // in case of miluphcuda file opening, etc. is handled by libconfig
        ERRORVAR("FILE ERROR! Cannot open '%s' for reading!\n", matfile)
    if( mat[CORE].eos == 'I' )
    {
        fprintf(stdout, "--------------------------------\n");
        fprintf(stdout, "Assigning values for material \"%s\", material type \"%d\", and compute sound speed from rho_0 and p_0 ...\n", mat[CORE].mat_name, mat[CORE].mat_type);
        mat[CORE].rho_0 = mat[CORE].ideal_gas.rho_0 = I_RHO0;
        mat[CORE].ideal_gas.p_0 = I_P0;
        mat[CORE].ideal_gas.gamma = I_GAMMA;
        mat[CORE].ideal_gas.polytropic_K = mat[CORE].ideal_gas.p_0 / pow(mat[CORE].ideal_gas.rho_0,mat[CORE].ideal_gas.gamma);
        mat[CORE].cs = sqrt( mat[CORE].ideal_gas.gamma * mat[CORE].ideal_gas.p_0 / mat[CORE].ideal_gas.rho_0 ); // compute cs with rho_0 and p_0 for now ...
    }
    else
    {
        readMaterialConstants(mfl, &mat[CORE], weibull_core);
        if( mat[CORE].eos == 'A' && hydrostructFlag )
        {
            allocate_ANEOS_table_memory(&mat[CORE]);
            load_ANEOS_table(&mat[CORE]);
        }
    }
    if( mat[MANTLE].eos == 'I' )
    {
        fprintf(stdout, "--------------------------------\n");
        fprintf(stdout, "Assigning values for material \"%s\", material type \"%d\", and compute sound speed from rho_0 and p_0 ...\n", mat[MANTLE].mat_name, mat[MANTLE].mat_type);
        mat[MANTLE].rho_0 = mat[MANTLE].ideal_gas.rho_0 = I_RHO0;
        mat[MANTLE].ideal_gas.p_0 = I_P0;
        mat[MANTLE].ideal_gas.gamma = I_GAMMA;
        mat[MANTLE].ideal_gas.polytropic_K = mat[MANTLE].ideal_gas.p_0 / pow(mat[MANTLE].ideal_gas.rho_0,mat[MANTLE].ideal_gas.gamma);
        mat[MANTLE].cs = sqrt( mat[MANTLE].ideal_gas.gamma * mat[MANTLE].ideal_gas.p_0 / mat[MANTLE].ideal_gas.rho_0 ); // compute cs with rho_0 and p_0 for now ...
    }
    else
    {
        readMaterialConstants(mfl, &mat[MANTLE], weibull_mantle);
        if( mat[MANTLE].eos == 'A' && hydrostructFlag )
        {
            allocate_ANEOS_table_memory(&mat[MANTLE]);
            load_ANEOS_table(&mat[MANTLE]);
        }
    }
    if( mat[SHELL].eos == 'I' )
    {
        fprintf(stdout, "--------------------------------\n");
        fprintf(stdout, "Assigning values for material \"%s\", material type \"%d\", and compute sound speed from rho_0 and p_0 ...\n", mat[SHELL].mat_name, mat[SHELL].mat_type);
        mat[SHELL].rho_0 = mat[SHELL].ideal_gas.rho_0 = I_RHO0;
        mat[SHELL].ideal_gas.p_0 = I_P0;
        mat[SHELL].ideal_gas.gamma = I_GAMMA;
        mat[SHELL].ideal_gas.polytropic_K = mat[SHELL].ideal_gas.p_0 / pow(mat[SHELL].ideal_gas.rho_0,mat[SHELL].ideal_gas.gamma);
        mat[SHELL].cs = sqrt( mat[SHELL].ideal_gas.gamma * mat[SHELL].ideal_gas.p_0 / mat[SHELL].ideal_gas.rho_0 ); // compute cs with rho_0 and p_0 for now ...
    }
    else
    {
        readMaterialConstants(mfl, &mat[SHELL], weibull_shell);
        if( mat[SHELL].eos == 'A' && hydrostructFlag )
        {
            allocate_ANEOS_table_memory(&mat[SHELL]);
            load_ANEOS_table(&mat[SHELL]);
        }
    }
    fclose(mfl);
#endif  // MILUPH
#ifdef MILUPHCUDA
    if( useProfilesFlag )   // some given radial profiles are read from files
    {
        if( (PROFILE_FILE_PROJ != 0 && PROFILE_R_P_C/PROFILE_R_P > eps6) || (PROFILE_FILE_TARG != 0 && PROFILE_R_T_C/PROFILE_R_T > eps6) )  // if there is any core in proj/targ
            readMaterialConfiguration(matfile, &mat[CORE], weibull_core);
        if( (PROFILE_FILE_PROJ != 0 && (PROFILE_R_P_M-PROFILE_R_P_C)/PROFILE_R_P > eps6) || (PROFILE_FILE_TARG != 0 && (PROFILE_R_T_M-PROFILE_R_T_C)/PROFILE_R_T > eps6) )  // if there is any mantle in proj/targ
        {
            readMaterialConfiguration(matfile, &mat[MANTLE], weibull_mantle);
        }
        else
        {
            memcpy(&mat[MANTLE], &mat[CORE], sizeof(material)); // if there is no actual mantle copy the core material to mat[MANTLE] as dummy
            mat[MANTLE].mat_type = MATTYPEMANTLE;
        }
        if( (PROFILE_FILE_PROJ != 0 && (PROFILE_R_P-PROFILE_R_P_M)/PROFILE_R_P > eps6) || (PROFILE_FILE_TARG != 0 && (PROFILE_R_T-PROFILE_R_T_M)/PROFILE_R_T > eps6) )  // if there is any shell in proj/targ
        {
            readMaterialConfiguration(matfile, &mat[SHELL], weibull_shell);
        }
        else
        {
            memcpy(&mat[SHELL], &mat[CORE], sizeof(material)); // if there is no actual shell copy the core material to mat[SHELL] as dummy
            mat[SHELL].mat_type = MATTYPESHELL;
        }
    }
    else
    {
        if( ((1.0-C_p_m_des-C_p_s_des) > eps6*eps6) || ((1.0-C_t_m_des-C_t_s_des) > eps6*eps6) )    // if there is any core in proj/targ
        {
            readMaterialConfiguration(matfile, &mat[CORE], weibull_core);
            if( mat[CORE].eos == 'A' && hydrostructFlag )
            {
                allocate_ANEOS_table_memory(&mat[CORE]);
                allocated_ANEOS_mem_core_flag = TRUE;
                load_ANEOS_table(&mat[CORE]);
            }
        }
        if( (C_p_m_des > eps6*eps6) || (C_t_m_des > eps6*eps6) )    // if there is any mantle in proj/targ
        {
            readMaterialConfiguration(matfile, &mat[MANTLE], weibull_mantle);
            if( mat[MANTLE].eos == 'A' && hydrostructFlag )
            {
                allocate_ANEOS_table_memory(&mat[MANTLE]);
                allocated_ANEOS_mem_mantle_flag = TRUE;
                load_ANEOS_table(&mat[MANTLE]);
            }
        }
        else
        {
            memcpy(&mat[MANTLE], &mat[CORE], sizeof(material)); // if there is no actual mantle copy the core material to mat[MANTLE] as dummy
            mat[MANTLE].mat_type = MATTYPEMANTLE;
        }
        if( (C_p_s_des > eps6*eps6) || (C_t_s_des > eps6*eps6) )    // if there is any shell in proj/targ
        {
            readMaterialConfiguration(matfile, &mat[SHELL], weibull_shell);
            if( mat[SHELL].eos == 'A' && hydrostructFlag )
            {
                allocate_ANEOS_table_memory(&mat[SHELL]);
                allocated_ANEOS_mem_shell_flag = TRUE;
                load_ANEOS_table(&mat[SHELL]);
            }
        }
        else
        {
            memcpy(&mat[SHELL], &mat[CORE], sizeof(material)); // if there is no actual shell copy the core material to mat[SHELL] as dummy
            mat[SHELL].mat_type = MATTYPESHELL;
        }
    }
#endif  // MILUPHCUDA
    
    
// set rho_limit (for Tillotson EoS) in case of miluph (for miluphcuda it is read from the materialconfiguration file)
#ifdef MILUPH
    mat[CORE].till.rho_limit = mat[MANTLE].till.rho_limit = mat[SHELL].till.rho_limit = TILL_RHO_LIMIT;
#endif
    
    
// assign function pointer to eos-related functions for all materials (only relevant for materials suitable for relaxation via hydrostatic structure)
    for(i=0; i<NMAT; i++)
    {
        if( mat[i].eos == 'T' )
        {
            mat_fp[i].eos_pressure = Tillotson_pressure;
            mat_fp[i].eos_density = Tillotson_density;
            mat_fp[i].eos_rho_self_consistent = rho_self_consistent_Tillotson;
            mat_fp[i].eos_e_compression = e_compression_Tillotson;
        }
        else if( mat[i].eos == 'A' )
        {
            mat_fp[i].eos_pressure = ANEOS_pressure;
            mat_fp[i].eos_density = ANEOS_density;
            mat_fp[i].eos_rho_self_consistent = rho_self_consistent_ANEOS;
            mat_fp[i].eos_e_compression = e_compression_ANEOS;
        }
        else if( mat[i].eos == 'I' )
        {
            mat_fp[i].eos_pressure = ideal_gas_pressure;
            mat_fp[i].eos_density = ideal_gas_density;
            mat_fp[i].eos_rho_self_consistent = rho_self_consistent_ideal_gas;
            mat_fp[i].eos_e_compression = e_compression_ideal_gas;
        }
    }
    
    
// calculate desired masses from input values
    if( useProfilesFlag )
    {
        M_t_des = M_p_c_des = M_p_m_des = M_p_s_des = M_t_c_des = M_t_m_des = M_t_s_des = -1.0;
    }
    else
    {
        M_t_des = M_des-M_p_des;
        M_p_c_des = M_p_des*(1-C_p_m_des-C_p_s_des);
        M_p_m_des = M_p_des*C_p_m_des;
        M_p_s_des = M_p_des*C_p_s_des;
        M_t_c_des = M_t_des*(1-C_t_m_des-C_t_s_des);
        M_t_m_des = M_t_des*C_t_m_des;
        M_t_s_des = M_t_des*C_t_s_des;
    }
    
// calculate radii assuming uncompressed material (from desired masses)
    if( useProfilesFlag )
    {
        R_p_c_uncomp = R_t_c_uncomp = R_p_m_uncomp = R_t_m_uncomp = R_p_uncomp = R_t_uncomp = -1.0;
    }
    else
    {
        R_p_c_uncomp = cbrt( THROVER4PI*M_p_c_des/mat[CORE].rho_0 );
        R_t_c_uncomp = cbrt( THROVER4PI*M_t_c_des/mat[CORE].rho_0 );
        R_p_m_uncomp = cbrt( THROVER4PI*M_p_m_des/mat[MANTLE].rho_0 + pow(R_p_c_uncomp,3) );
        R_t_m_uncomp = cbrt( THROVER4PI*M_t_m_des/mat[MANTLE].rho_0 + pow(R_t_c_uncomp,3) );
        R_p_uncomp = cbrt( THROVER4PI*M_p_s_des/mat[SHELL].rho_0 + pow(R_p_m_uncomp,3) );
        R_t_uncomp = cbrt( THROVER4PI*M_t_s_des/mat[SHELL].rho_0 + pow(R_t_m_uncomp,3) );
    }
    
    
// calculate hydrostatic internal structure and assign found radii of proj/targ
    if( hydrostructFlag )
    {
        #pragma omp parallel
        {
            #pragma omp sections
            {
                #pragma omp section
                if( M_p_des/M_des > eps6 )  // if there is a projectile
                    calc_internal_structure(M_p_des, M_p_c_des, M_p_m_des, struct_r_low*R_p_uncomp, struct_r_high*R_p_uncomp, mat, mat_fp, int_struct_p_c, int_struct_p_m, int_struct_p_s, NSTEPS);
                #pragma omp section
                if( M_t_des/M_des > eps6 )  // if there is a target
                    calc_internal_structure(M_t_des, M_t_c_des, M_t_m_des, struct_r_low*R_t_uncomp, struct_r_high*R_t_uncomp, mat, mat_fp, int_struct_t_c, int_struct_t_m, int_struct_t_s, NSTEPS);
            }   // end of omp sections region
        }   // end of omp parallel region
        
        // assign hydrostatically calculated radii of the projectile (depending on what it actually consists of) and write structure file
        if (M_p_des/M_des > eps6)
        {
            if( M_p_s_des/M_des > eps6 )    // if there is a shell
                R_p_des = int_struct_p_s[NSTEPS].r;
            else if( M_p_m_des/M_des > eps6 )    // if there is no shell, but a mantle
                R_p_des = int_struct_p_m[NSTEPS].r;
            else    // if there is no shell or mantle, but a core
                R_p_des = int_struct_p_c[NSTEPS].r;
#ifdef HYDROSTRUCT_DEBUG
            fprintf(stdout, "FOUND HYDROSTATIC STRUCTURE of the PROJECTILE with R_p_des = %.15le\n", R_p_des);
#endif
            if ( (pfl = fopen(p_structfile,"w")) == NULL )
                ERRORVAR("FILE ERROR! Cannot open '%s' for writing!\n", p_structfile)
            fprintf(pfl, "# r (m)\t\tm (kg)\t\trho (kg/m^3)\te (J/kg)\tp (Pa)");
            if( mat[CORE].eos == 'I' || mat[MANTLE].eos == 'I' || mat[SHELL].eos == 'I' || mat[CORE].eos == 'A' || mat[MANTLE].eos == 'A' || mat[SHELL].eos == 'A' )
                fprintf(pfl, "\t\tT (K)");
            if( mat[CORE].eos == 'A' || mat[MANTLE].eos == 'A' || mat[SHELL].eos == 'A' )
                fprintf(pfl, "\t\tsound-speed (m/s)\t\tentropy (J/kg/K)\t\tphase-flag");
            fprintf(pfl, "\n");
            
            if( M_p_c_des/M_des > eps6 )    // if there is a core
            {
                R_p_c_des = int_struct_p_c[NSTEPS].r;
                for(i=0; i<=NSTEPS; i++)
                {
                    e_temp = (*(mat_fp[CORE].eos_e_compression))(int_struct_p_c[i].rho, &mat[CORE]);
                    fprintf(pfl, "%.16le\t%.16le\t%.16le\t%.16le\t%.16le", int_struct_p_c[i].r, int_struct_p_c[i].m, int_struct_p_c[i].rho, e_temp, int_struct_p_c[i].p);
                    if( mat[CORE].eos == 'I' )
                    {
#ifdef MILUPH
                        fprintf(pfl, "\t%.16le", e_temp*CONVERSION_E_TO_T);
#endif
#ifdef MILUPHCUDA
                        fprintf(pfl, "\t%.16le", e_temp*(mat[CORE].ideal_gas.conv_e_to_T));
#endif
                    }
                    if( mat[CORE].eos == 'A' )
                    {
                        // find array-indices just below the actual values of rho and e
                        aneos_i_rho = array_index(int_struct_p_c[i].rho, mat[CORE].aneos.rho, mat[CORE].aneos.n_rho);
                        aneos_i_e = array_index(e_temp, mat[CORE].aneos.e, mat[CORE].aneos.n_e);
                        // interpolate (bi)linearly
                        aneos_T = bilinear_interpolation(int_struct_p_c[i].rho, e_temp, mat[CORE].aneos.T, mat[CORE].aneos.rho, mat[CORE].aneos.e, aneos_i_rho, aneos_i_e, mat[CORE].aneos.n_rho, mat[CORE].aneos.n_e);
                        aneos_cs = bilinear_interpolation(int_struct_p_c[i].rho, e_temp, mat[CORE].aneos.cs, mat[CORE].aneos.rho, mat[CORE].aneos.e, aneos_i_rho, aneos_i_e, mat[CORE].aneos.n_rho, mat[CORE].aneos.n_e);
                        aneos_entropy = bilinear_interpolation(int_struct_p_c[i].rho, e_temp, mat[CORE].aneos.entropy, mat[CORE].aneos.rho, mat[CORE].aneos.e, aneos_i_rho, aneos_i_e, mat[CORE].aneos.n_rho, mat[CORE].aneos.n_e);
                        aneos_phase_flag = discrete_value_table_lookup(int_struct_p_c[i].rho, e_temp, mat[CORE].aneos.phase_flag, mat[CORE].aneos.rho, mat[CORE].aneos.e, aneos_i_rho, aneos_i_e, mat[CORE].aneos.n_rho, mat[CORE].aneos.n_e);
                        fprintf(pfl, "\t%.16le\t%.16le\t%.16le\t%d", aneos_T, aneos_cs, aneos_entropy, aneos_phase_flag);
                    }
                    fprintf(pfl, "\n");
                }
            }
            else
                R_p_c_des = 0.0;
            
            if( M_p_m_des/M_des > eps6 )    // if there is a mantle
            {
                R_p_m_des = int_struct_p_m[NSTEPS].r;
                for(i=0; i<=NSTEPS; i++)
                {
                    e_temp = (*(mat_fp[MANTLE].eos_e_compression))(int_struct_p_m[i].rho, &mat[MANTLE]);
                    fprintf(pfl, "%.16le\t%.16le\t%.16le\t%.16le\t%.16le", int_struct_p_m[i].r, int_struct_p_m[i].m, int_struct_p_m[i].rho, e_temp, int_struct_p_m[i].p);
                    if( mat[MANTLE].eos == 'I' )
                    {
#ifdef MILUPH
                        fprintf(pfl, "\t%.16le", e_temp*CONVERSION_E_TO_T);
#endif
#ifdef MILUPHCUDA
                        fprintf(pfl, "\t%.16le", e_temp*(mat[MANTLE].ideal_gas.conv_e_to_T));
#endif
                    }
                    if( mat[MANTLE].eos == 'A' )
                    {
                        // find array-indices just below the actual values of rho and e
                        aneos_i_rho = array_index(int_struct_p_m[i].rho, mat[MANTLE].aneos.rho, mat[MANTLE].aneos.n_rho);
                        aneos_i_e = array_index(e_temp, mat[MANTLE].aneos.e, mat[MANTLE].aneos.n_e);
                        // interpolate (bi)linearly
                        aneos_T = bilinear_interpolation(int_struct_p_m[i].rho, e_temp, mat[MANTLE].aneos.T, mat[MANTLE].aneos.rho, mat[MANTLE].aneos.e, aneos_i_rho, aneos_i_e, mat[MANTLE].aneos.n_rho, mat[MANTLE].aneos.n_e);
                        aneos_cs = bilinear_interpolation(int_struct_p_m[i].rho, e_temp, mat[MANTLE].aneos.cs, mat[MANTLE].aneos.rho, mat[MANTLE].aneos.e, aneos_i_rho, aneos_i_e, mat[MANTLE].aneos.n_rho, mat[MANTLE].aneos.n_e);
                        aneos_entropy = bilinear_interpolation(int_struct_p_m[i].rho, e_temp, mat[MANTLE].aneos.entropy, mat[MANTLE].aneos.rho, mat[MANTLE].aneos.e, aneos_i_rho, aneos_i_e, mat[MANTLE].aneos.n_rho, mat[MANTLE].aneos.n_e);
                        aneos_phase_flag = discrete_value_table_lookup(int_struct_p_m[i].rho, e_temp, mat[MANTLE].aneos.phase_flag, mat[MANTLE].aneos.rho, mat[MANTLE].aneos.e, aneos_i_rho, aneos_i_e, mat[MANTLE].aneos.n_rho, mat[MANTLE].aneos.n_e);
                        fprintf(pfl, "\t%.16le\t%.16le\t%.16le\t%d", aneos_T, aneos_cs, aneos_entropy, aneos_phase_flag);
                    }
                    fprintf(pfl, "\n");
                }
            }
            else
                R_p_m_des = R_p_c_des;
            
            if( M_p_s_des/M_des > eps6 )    // if there is a shell
                for(i=0; i<=NSTEPS; i++)
                {
                    e_temp = (*(mat_fp[SHELL].eos_e_compression))(int_struct_p_s[i].rho, &mat[SHELL]);
                    fprintf(pfl, "%.16le\t%.16le\t%.16le\t%.16le\t%.16le", int_struct_p_s[i].r, int_struct_p_s[i].m, int_struct_p_s[i].rho, e_temp, int_struct_p_s[i].p);
                    if( mat[SHELL].eos == 'I' )
                    {
#ifdef MILUPH
                        fprintf(pfl, "\t%.16le", e_temp*CONVERSION_E_TO_T);
#endif
#ifdef MILUPHCUDA
                        fprintf(pfl, "\t%.16le", e_temp*(mat[SHELL].ideal_gas.conv_e_to_T));
#endif
                    }
                    if( mat[SHELL].eos == 'A' )
                    {
                        // find array-indices just below the actual values of rho and e
                        aneos_i_rho = array_index(int_struct_p_s[i].rho, mat[SHELL].aneos.rho, mat[SHELL].aneos.n_rho);
                        aneos_i_e = array_index(e_temp, mat[SHELL].aneos.e, mat[SHELL].aneos.n_e);
                        // interpolate (bi)linearly
                        aneos_T = bilinear_interpolation(int_struct_p_s[i].rho, e_temp, mat[SHELL].aneos.T, mat[SHELL].aneos.rho, mat[SHELL].aneos.e, aneos_i_rho, aneos_i_e, mat[SHELL].aneos.n_rho, mat[SHELL].aneos.n_e);
                        aneos_cs = bilinear_interpolation(int_struct_p_s[i].rho, e_temp, mat[SHELL].aneos.cs, mat[SHELL].aneos.rho, mat[SHELL].aneos.e, aneos_i_rho, aneos_i_e, mat[SHELL].aneos.n_rho, mat[SHELL].aneos.n_e);
                        aneos_entropy = bilinear_interpolation(int_struct_p_s[i].rho, e_temp, mat[SHELL].aneos.entropy, mat[SHELL].aneos.rho, mat[SHELL].aneos.e, aneos_i_rho, aneos_i_e, mat[SHELL].aneos.n_rho, mat[SHELL].aneos.n_e);
                        aneos_phase_flag = discrete_value_table_lookup(int_struct_p_s[i].rho, e_temp, mat[SHELL].aneos.phase_flag, mat[SHELL].aneos.rho, mat[SHELL].aneos.e, aneos_i_rho, aneos_i_e, mat[SHELL].aneos.n_rho, mat[SHELL].aneos.n_e);
                        fprintf(pfl, "\t%.16le\t%.16le\t%.16le\t%d", aneos_T, aneos_cs, aneos_entropy, aneos_phase_flag);
                    }
                    fprintf(pfl, "\n");
                }
            fclose(pfl);
        }
        else
            R_p_des = R_p_c_des = R_p_m_des = 0.0;
        
        // assign hydrostatically calculated radii of the target (depending on what it actually consists of) and write structure file
        if (M_t_des/M_des > eps6)
        {
            if( M_t_s_des/M_des > eps6 )    // if there is a shell
                R_t_des = int_struct_t_s[NSTEPS].r;
            else if( M_t_m_des/M_des > eps6 )    // if there is no shell, but a mantle
                R_t_des = int_struct_t_m[NSTEPS].r;
            else    // if there is no shell or mantle, but a core
                R_t_des = int_struct_t_c[NSTEPS].r;
#ifdef HYDROSTRUCT_DEBUG
            fprintf(stdout, "FOUND HYDROSTATIC STRUCTURE of the TARGET with R_t_des = %.16le\n", R_t_des);
#endif
            if ( (tfl = fopen(t_structfile,"w")) == NULL )
                ERRORVAR("FILE ERROR! Cannot open '%s' for writing!\n", t_structfile)
            fprintf(tfl, "# r (m)\t\tm (kg)\t\trho (kg/m^3)\te (J/kg)\tp (Pa)");
            if( mat[CORE].eos == 'I' || mat[MANTLE].eos == 'I' || mat[SHELL].eos == 'I' || mat[CORE].eos == 'A' || mat[MANTLE].eos == 'A' || mat[SHELL].eos == 'A' )
                fprintf(tfl, "\t\tT (K)");
            if( mat[CORE].eos == 'A' || mat[MANTLE].eos == 'A' || mat[SHELL].eos == 'A' )
                fprintf(tfl, "\t\tsound-speed (m/s)\t\tentropy (J/kg/K)\t\tphase-flag");
            fprintf(tfl, "\n");
            
            if( M_t_c_des/M_des > eps6 )    // if there is a core
            {
                R_t_c_des = int_struct_t_c[NSTEPS].r;
                for(i=0; i<=NSTEPS; i++)
                {
                    e_temp = (*(mat_fp[CORE].eos_e_compression))(int_struct_t_c[i].rho, &mat[CORE]);
                    fprintf(tfl, "%.16le\t%.16le\t%.16le\t%.16le\t%.16le", int_struct_t_c[i].r, int_struct_t_c[i].m, int_struct_t_c[i].rho, e_temp, int_struct_t_c[i].p);
                    if( mat[CORE].eos == 'I' )
                    {
#ifdef MILUPH
                        fprintf(tfl, "\t%.16le", e_temp*CONVERSION_E_TO_T);
#endif
#ifdef MILUPHCUDA
                        fprintf(tfl, "\t%.16le", e_temp*(mat[CORE].ideal_gas.conv_e_to_T));
#endif
                    }
                    if( mat[CORE].eos == 'A' )
                    {
                        // find array-indices just below the actual values of rho and e
                        aneos_i_rho = array_index(int_struct_t_c[i].rho, mat[CORE].aneos.rho, mat[CORE].aneos.n_rho);
                        aneos_i_e = array_index(e_temp, mat[CORE].aneos.e, mat[CORE].aneos.n_e);
                        // interpolate (bi)linearly
                        aneos_T = bilinear_interpolation(int_struct_t_c[i].rho, e_temp, mat[CORE].aneos.T, mat[CORE].aneos.rho, mat[CORE].aneos.e, aneos_i_rho, aneos_i_e, mat[CORE].aneos.n_rho, mat[CORE].aneos.n_e);
                        aneos_cs = bilinear_interpolation(int_struct_t_c[i].rho, e_temp, mat[CORE].aneos.cs, mat[CORE].aneos.rho, mat[CORE].aneos.e, aneos_i_rho, aneos_i_e, mat[CORE].aneos.n_rho, mat[CORE].aneos.n_e);
                        aneos_entropy = bilinear_interpolation(int_struct_t_c[i].rho, e_temp, mat[CORE].aneos.entropy, mat[CORE].aneos.rho, mat[CORE].aneos.e, aneos_i_rho, aneos_i_e, mat[CORE].aneos.n_rho, mat[CORE].aneos.n_e);
                        aneos_phase_flag = discrete_value_table_lookup(int_struct_t_c[i].rho, e_temp, mat[CORE].aneos.phase_flag, mat[CORE].aneos.rho, mat[CORE].aneos.e, aneos_i_rho, aneos_i_e, mat[CORE].aneos.n_rho, mat[CORE].aneos.n_e);
                        fprintf(tfl, "\t%.16le\t%.16le\t%.16le\t%d", aneos_T, aneos_cs, aneos_entropy, aneos_phase_flag);
                    }
                    fprintf(tfl, "\n");
                }
            }
            else
                R_t_c_des = 0.0;
            
            if( M_t_m_des/M_des > eps6 )    // if there is a mantle
            {
                R_t_m_des = int_struct_t_m[NSTEPS].r;
                for(i=0; i<=NSTEPS; i++)
                {
                    e_temp = (*(mat_fp[MANTLE].eos_e_compression))(int_struct_t_m[i].rho, &mat[MANTLE]);
                    fprintf(tfl, "%.16le\t%.16le\t%.16le\t%.16le\t%.16le", int_struct_t_m[i].r, int_struct_t_m[i].m, int_struct_t_m[i].rho, e_temp, int_struct_t_m[i].p);
                    if( mat[MANTLE].eos == 'I' )
                    {
#ifdef MILUPH
                        fprintf(tfl, "\t%.16le", e_temp*CONVERSION_E_TO_T);
#endif
#ifdef MILUPHCUDA
                        fprintf(tfl, "\t%.16le", e_temp*(mat[MANTLE].ideal_gas.conv_e_to_T));
#endif
                    }
                    if( mat[MANTLE].eos == 'A' )
                    {
                        // find array-indices just below the actual values of rho and e
                        aneos_i_rho = array_index(int_struct_t_m[i].rho, mat[MANTLE].aneos.rho, mat[MANTLE].aneos.n_rho);
                        aneos_i_e = array_index(e_temp, mat[MANTLE].aneos.e, mat[MANTLE].aneos.n_e);
                        // interpolate (bi)linearly
                        aneos_T = bilinear_interpolation(int_struct_t_m[i].rho, e_temp, mat[MANTLE].aneos.T, mat[MANTLE].aneos.rho, mat[MANTLE].aneos.e, aneos_i_rho, aneos_i_e, mat[MANTLE].aneos.n_rho, mat[MANTLE].aneos.n_e);
                        aneos_cs = bilinear_interpolation(int_struct_t_m[i].rho, e_temp, mat[MANTLE].aneos.cs, mat[MANTLE].aneos.rho, mat[MANTLE].aneos.e, aneos_i_rho, aneos_i_e, mat[MANTLE].aneos.n_rho, mat[MANTLE].aneos.n_e);
                        aneos_entropy = bilinear_interpolation(int_struct_t_m[i].rho, e_temp, mat[MANTLE].aneos.entropy, mat[MANTLE].aneos.rho, mat[MANTLE].aneos.e, aneos_i_rho, aneos_i_e, mat[MANTLE].aneos.n_rho, mat[MANTLE].aneos.n_e);
                        aneos_phase_flag = discrete_value_table_lookup(int_struct_t_m[i].rho, e_temp, mat[MANTLE].aneos.phase_flag, mat[MANTLE].aneos.rho, mat[MANTLE].aneos.e, aneos_i_rho, aneos_i_e, mat[MANTLE].aneos.n_rho, mat[MANTLE].aneos.n_e);
                        fprintf(tfl, "\t%.16le\t%.16le\t%.16le\t%d", aneos_T, aneos_cs, aneos_entropy, aneos_phase_flag);
                    }
                    fprintf(tfl, "\n");
                }
            }
            else
                R_t_m_des = R_t_c_des;
            
            if( M_t_s_des/M_des > eps6 )    // if there is a shell
                for(i=0; i<=NSTEPS; i++)
                {
                    e_temp = (*(mat_fp[SHELL].eos_e_compression))(int_struct_t_s[i].rho, &mat[SHELL]);
                    fprintf(tfl, "%.16le\t%.16le\t%.16le\t%.16le\t%.16le", int_struct_t_s[i].r, int_struct_t_s[i].m, int_struct_t_s[i].rho, e_temp, int_struct_t_s[i].p);
                    if( mat[SHELL].eos == 'I' )
                    {
#ifdef MILUPH
                        fprintf(tfl, "\t%.16le", e_temp*CONVERSION_E_TO_T);
#endif
#ifdef MILUPHCUDA
                        fprintf(tfl, "\t%.16le", e_temp*(mat[SHELL].ideal_gas.conv_e_to_T));
#endif
                    }
                    if( mat[SHELL].eos == 'A' )
                    {
                        // find array-indices just below the actual values of rho and e
                        aneos_i_rho = array_index(int_struct_t_s[i].rho, mat[SHELL].aneos.rho, mat[SHELL].aneos.n_rho);
                        aneos_i_e = array_index(e_temp, mat[SHELL].aneos.e, mat[SHELL].aneos.n_e);
                        // interpolate (bi)linearly
                        aneos_T = bilinear_interpolation(int_struct_t_s[i].rho, e_temp, mat[SHELL].aneos.T, mat[SHELL].aneos.rho, mat[SHELL].aneos.e, aneos_i_rho, aneos_i_e, mat[SHELL].aneos.n_rho, mat[SHELL].aneos.n_e);
                        aneos_cs = bilinear_interpolation(int_struct_t_s[i].rho, e_temp, mat[SHELL].aneos.cs, mat[SHELL].aneos.rho, mat[SHELL].aneos.e, aneos_i_rho, aneos_i_e, mat[SHELL].aneos.n_rho, mat[SHELL].aneos.n_e);
                        aneos_entropy = bilinear_interpolation(int_struct_t_s[i].rho, e_temp, mat[SHELL].aneos.entropy, mat[SHELL].aneos.rho, mat[SHELL].aneos.e, aneos_i_rho, aneos_i_e, mat[SHELL].aneos.n_rho, mat[SHELL].aneos.n_e);
                        aneos_phase_flag = discrete_value_table_lookup(int_struct_t_s[i].rho, e_temp, mat[SHELL].aneos.phase_flag, mat[SHELL].aneos.rho, mat[SHELL].aneos.e, aneos_i_rho, aneos_i_e, mat[SHELL].aneos.n_rho, mat[SHELL].aneos.n_e);
                        fprintf(tfl, "\t%.16le\t%.16le\t%.16le\t%d", aneos_T, aneos_cs, aneos_entropy, aneos_phase_flag);
                    }
                    fprintf(tfl, "\n");
                }
            fclose(tfl);
        }
        else
            R_t_des = R_t_c_des = R_t_m_des = 0.0;
    }
    else if( useProfilesFlag )  // some given profiles are used for setting the bodies' structure
    {
        R_p_des = PROFILE_R_P;
        R_p_c_des = PROFILE_R_P_C;
        R_p_m_des = PROFILE_R_P_M;
        R_t_des = PROFILE_R_T;
        R_t_c_des = PROFILE_R_T_C;
        R_t_m_des = PROFILE_R_T_M;
    }
    else    // a homogeneous density is assumed throughout each material
    {
        R_p_des = R_p_uncomp;
        R_p_c_des = R_p_c_uncomp;
        R_p_m_des = R_p_m_uncomp;
        R_t_des = R_t_uncomp;
        R_t_c_des = R_t_c_uncomp;
        R_t_m_des = R_t_m_uncomp;
    }   // end 'if( hydrostructFlag )'
    
    
// compute desired particle numbers of proj/targ weighted by the volume of proj/targ
    N_p_des = N_des * pow(R_p_des,3)/( pow(R_p_des,3) + pow(R_t_des,3) );
    N_t_des = N_des * pow(R_t_des,3)/( pow(R_p_des,3) + pow(R_t_des,3) );
    
    
// allocate memory for all SPH particles in 'p'
    if ( (p = (particle*)malloc(MEMALLOCFACT*N_des*sizeof(particle))) == NULL )
        ERRORTEXT("ERROR during memory allocation for SPH particles!\n")
    
// add projectile and target spheres around the origin - positions, all velocity components = 0, and material types are added
    N_p_c = N_t_c = N_p_m = N_t_m = 0;
    fprintf(stdout, "--------------------------------\n");
    if( ParticleGeometry == 0 )     // SC lattice
    {
        mpd = cbrt( (pow(R_p_des,3)+pow(R_t_des,3))/(THROVER4PI*(double)N_des) );
        V_particle_p = V_particle_t = pow(mpd,3);
        fprintf(stdout, "Building sphere(s) based on a simple-cubic lattice ... ");
        N_p = add_sphere_particles_SC(p, mat, mpd, R_p_des, R_p_c_des, R_p_m_des, &N_p_c, &N_p_m);
        N_t = add_sphere_particles_SC(p+N_p, mat, mpd, R_t_des, R_t_c_des, R_t_m_des, &N_t_c, &N_t_m);
        fprintf(stdout, "Done.\n");
    }
    else if( ParticleGeometry == 1 )    // HCP lattice
    {
        mpd = cbrt( sqrt(2.0)*(pow(R_p_des,3)+pow(R_t_des,3))/(THROVER4PI*(double)N_des) );
        V_particle_p = V_particle_t = pow(mpd,3)/sqrt(2.0);
        fprintf(stdout, "Building sphere(s) based on a hexagonally close-packed lattice ... ");
        N_p = add_sphere_particles_HCP(p, mat, mpd, R_p_des, R_p_c_des, R_p_m_des, &N_p_c, &N_p_m);
        N_t = add_sphere_particles_HCP(p+N_p, mat, mpd, R_t_des, R_t_c_des, R_t_m_des, &N_t_c, &N_t_m);
        fprintf(stdout, "Done.\n");
    }
    else if( ParticleGeometry == 2 )    // spherical shell setup with SEAGen
    {
        fprintf(stdout, "Building sphere(s) based on a spherical shell setup generated with SEAGen ...\n");
        Py_Initialize();    // it is important to initialize (and finalize) Python only once (module stuff might get messed up otherwise)
        if( !Py_IsInitialized() )
            ERRORTEXT("ERROR! Unable to initialize Python interpreter.\n")
        if( M_p_des/M_des > eps6 )  // if there is a projectile
            N_p = add_sphere_particles_SS(p, N_p_des, R_p_des, R_p_c_des, R_p_m_des, source_dir, "projectile.SEAGen", &N_p_c, &N_p_m, &mpd_p);
        if( M_t_des/M_des > eps6 )  // if there is a target
            N_t = add_sphere_particles_SS(p+N_p, N_t_des, R_t_des, R_t_c_des, R_t_m_des, source_dir, "target.SEAGen", &N_t_c, &N_t_m, &mpd_t);
        Py_Finalize();
        V_particle_p = pow(mpd_p,3) / sqrt(2.0);
        V_particle_t = pow(mpd_t,3) / sqrt(2.0);
        mpd = MAX(mpd_p, mpd_t);    // to be on the safe side (for setting the sml) ...
    }
    N = N_p+N_t;
    
    
// compute sml based on mean particle distance
    mat[CORE].sml = mat[MANTLE].sml = mat[SHELL].sml = mpd * sml_factor;
    
// write sml to all relevant materials in materialconfiguration file for miluphcuda
#ifdef MILUPHCUDA
    pasteSml(matfile, mat);
#endif
    
// set artificial viscosity parameters and write scenario file for miluph
#ifdef MILUPH
    if ( (sfl = fopen(scenfile,"w")) == NULL )
        ERRORVAR("FILE ERROR! Cannot open '%s' for writing!\n", scenfile)
    mat[CORE].alpha = mat[MANTLE].alpha = mat[SHELL].alpha = ART_VISC_ALPHA;
    mat[CORE].beta = mat[MANTLE].beta = mat[SHELL].beta = ART_VISC_BETA;
    write_scenario_data(sfl, mat, NMAT);
    fclose(sfl);
#endif    
    
    
// if in N_input mode, either create SPH particles for the additional bodies/pointmasses,
// or write their properties to a separate output file (pointmasses-file)
    if( N_input == TRUE  &&  M_output == FALSE )   // create SPH particles for additional pointmasses
    {
        for(i=2; i<N_bodies; i++)
        {
            for(j=0; j<DIM; j++)
            {
                p[N].x[j] = N_input_data[i].x[j];
                p[N].v[j] = N_input_data[i].v[j];
            }
            p[N].mat_type = N_BODIES_MATTYPE;
            p[N].rho = N_BODIES_RHO;
            p[N].mass = N_input_data[i].mass;
            p[N].e = 0.0;
            N++;
        }
        if( (N_bodies-2 + N_p + N_t) != N )
            ERRORTEXT("ERROR! Strange particle number mismatch in N_bodies mode.\n")
    }
    else if( N_input == TRUE  &&  M_output == TRUE )   // write additional bodies/pointmasses to pointmasses-file
    {
        FILE *tmp_f;
        
        if ( (tmp_f = fopen(pointmassesfile,"w")) == NULL )
            ERRORVAR("FILE ERROR! Cannot open '%s' for writing!\n", pointmassesfile)
        
        for(i=2; i<N_bodies; i++)
        {
            for(j=0; j<DIM; j++)
                fprintf(tmp_f, "%.16le\t", N_input_data[i].x[j]);
            for(j=0; j<DIM; j++)
                fprintf(tmp_f, "%.16le\t", N_input_data[i].v[j]);
            fprintf(tmp_f, "%.16le\t", N_input_data[i].mass);
            fprintf(tmp_f, "%.16le\t", N_BODIES_RMIN);
            fprintf(tmp_f, "%.16le\t", N_BODIES_RMAX);
            fprintf(tmp_f, "%d\n", N_BODIES_FEELING_FLAG);
        }
        fclose(tmp_f);
    }
    
    
// make sure that the allocated memory (at p) for the particles was sufficient
    if ( MEMALLOCFACT*N_des < N )
        ERRORTEXT("ERROR! Too little memory has been originally allocated for the actual number of particles.\n")
    
    
// calculate and assign densities, masses and internal energies to the proj/targ particles
    if( hydrostructFlag )   // the density distribution follows the correct hydrostatic structure
    {
        // assign densities, masses and internal energies to the projectile's particles, assuming the hydrostatic structure calculated above
        for(i=0; i<N_p; i++)
        {
            r2 = p[i].x[0]*p[i].x[0] + p[i].x[1]*p[i].x[1] + p[i].x[2]*p[i].x[2];
            if( (r2 < R_p_c_des*R_p_c_des) && (M_p_c_des/M_des > eps6) && (p[i].mat_type == mat[CORE].mat_type) )    // if particle is in the projectile's core
            {
                set_hydrostruct_density(p, i, r2, int_struct_p_c);
                p[i].e = (*(mat_fp[CORE].eos_e_compression))(p[i].rho, &mat[CORE]);
            }
            else if( (r2 < R_p_m_des*R_p_m_des) && (M_p_m_des/M_des > eps6) && (p[i].mat_type == mat[MANTLE].mat_type) )    // if particle is in the projectile's mantle
            {
                set_hydrostruct_density(p, i, r2, int_struct_p_m);
                p[i].e = (*(mat_fp[MANTLE].eos_e_compression))(p[i].rho, &mat[MANTLE]);
            }
            else if( (r2 < R_p_des*R_p_des) && (M_p_s_des/M_des > eps6) && (p[i].mat_type == mat[SHELL].mat_type) )    // if particle is in the projectile's shell
            {
                set_hydrostruct_density(p, i, r2, int_struct_p_s);
                p[i].e = (*(mat_fp[SHELL].eos_e_compression))(p[i].rho, &mat[SHELL]);
            }
            else
                ERRORTEXT("ERROR when assigning densities, masses and internal energies (calculated via hydrostatic structure) to the projectile's particles! Particle characteristics seem to match neither the projectile's core nor mantle nor shell perfectly ...\n")
            
            p[i].mass = p[i].rho * V_particle_p;
        }
        // assign densities, masses and internal energies to the target's particles, assuming the hydrostatic structure calculated above
        for( i=N_p; i<(N_p+N_t); i++)
        {
            r2 = p[i].x[0]*p[i].x[0] + p[i].x[1]*p[i].x[1] + p[i].x[2]*p[i].x[2];
            if( (r2 < R_t_c_des*R_t_c_des) && (M_t_c_des/M_des > eps6) && (p[i].mat_type == mat[CORE].mat_type) )    // if particle is in the target's core
            {
                set_hydrostruct_density(p, i, r2, int_struct_t_c);
                p[i].e = (*(mat_fp[CORE].eos_e_compression))(p[i].rho, &mat[CORE]);
            }
            else if( (r2 < R_t_m_des*R_t_m_des) && (M_t_m_des/M_des > eps6) && (p[i].mat_type == mat[MANTLE].mat_type) )    // if particle is in the target's mantle
            {
                set_hydrostruct_density(p, i, r2, int_struct_t_m);
                p[i].e = (*(mat_fp[MANTLE].eos_e_compression))(p[i].rho, &mat[MANTLE]);
            }
            else if( (r2 < R_t_des*R_t_des) && (M_t_s_des/M_des > eps6) && (p[i].mat_type == mat[SHELL].mat_type) )    // if particle is in the target's shell
            {
                set_hydrostruct_density(p, i, r2, int_struct_t_s);
                p[i].e = (*(mat_fp[SHELL].eos_e_compression))(p[i].rho, &mat[SHELL]);
            }
            else
                ERRORTEXT("ERROR when assigning densities, masses and internal energies (calculated via hydrostatic structure) to the target's particles! Particle characteristics seem to match neither the target's core nor mantle nor shell perfectly ...\n")
            
            p[i].mass = p[i].rho * V_particle_t;
        }
    }
    else if( useProfilesFlag )  // given profiles are used for setting the bodies structure
    {
        // assign densities, masses and energies to the projectile's particles, following the given profile
        for(i=0; i<N_p; i++)
        {
            r2 = p[i].x[0]*p[i].x[0] + p[i].x[1]*p[i].x[1] + p[i].x[2]*p[i].x[2];
            set_profile_rho_e(p, i, r2, profile_projectile, p_profile_no_points);
            p[i].mass = p[i].rho * V_particle_p;
        }
        // assign densities, masses, and energies to the target's particles, following the given profile
        for( i=N_p; i<(N_p+N_t); i++)
        {
            r2 = p[i].x[0]*p[i].x[0] + p[i].x[1]*p[i].x[1] + p[i].x[2]*p[i].x[2];
            set_profile_rho_e(p, i, r2, profile_target, t_profile_no_points);
            p[i].mass = p[i].rho * V_particle_t;
        }
    }
    else    // homogeneous density is assumed throughout each material
    {
        M_particle_p[CORE] = V_particle_p*mat[CORE].rho_0;
        M_particle_p[MANTLE] = V_particle_p*mat[MANTLE].rho_0;
        M_particle_p[SHELL] = V_particle_p*mat[SHELL].rho_0;
        M_particle_t[CORE] = V_particle_t*mat[CORE].rho_0;
        M_particle_t[MANTLE] = V_particle_t*mat[MANTLE].rho_0;
        M_particle_t[SHELL] = V_particle_t*mat[SHELL].rho_0;
        
        for( i=0; i<(N_p+N_t); i++)
        {
            if( p[i].mat_type == mat[CORE].mat_type )
            {
                if( i < N_p )     // projectile
                    p[i].mass = M_particle_p[CORE];
                if( i >= N_p )  // target
                    p[i].mass = M_particle_t[CORE];
                p[i].rho = mat[CORE].rho_0;
            }
            else if( p[i].mat_type == mat[MANTLE].mat_type )
            {
                if( i < N_p )     // projectile
                    p[i].mass = M_particle_p[MANTLE];
                if( i >= N_p )  // target
                    p[i].mass = M_particle_t[MANTLE];
                p[i].rho = mat[MANTLE].rho_0;
            }
            else if( p[i].mat_type == mat[SHELL].mat_type )
            {
                if( i < N_p )     // projectile
                    p[i].mass = M_particle_p[SHELL];
                if( i >= N_p )  // target
                    p[i].mass = M_particle_t[SHELL];
                p[i].rho = mat[SHELL].rho_0;
            }
            // assign internal energies depending on eos
            if( (p[i].mat_type==mat[CORE].mat_type) && (mat[CORE].eos=='A') )
                p[i].e = mat[CORE].aneos.e_norm;
            else if( (p[i].mat_type==mat[MANTLE].mat_type) && (mat[MANTLE].eos == 'A') )
                p[i].e = mat[MANTLE].aneos.e_norm;
            else if( (p[i].mat_type==mat[SHELL].mat_type) && (mat[SHELL].eos=='A') )
                p[i].e = mat[SHELL].aneos.e_norm;
            else if( (p[i].mat_type==mat[CORE].mat_type) && (mat[CORE].eos=='I') )
                p[i].e = e_compression_ideal_gas(p[i].rho, &mat[CORE]);
            else if( (p[i].mat_type==mat[MANTLE].mat_type) && (mat[MANTLE].eos=='I') )
                p[i].e = e_compression_ideal_gas(p[i].rho, &mat[MANTLE]);
            else if( (p[i].mat_type==mat[SHELL].mat_type) && (mat[SHELL].eos=='I') )
                p[i].e = e_compression_ideal_gas(p[i].rho, &mat[SHELL]);
            else
                p[i].e = 0.0;
        }
    }   // end 'if( hydrostructFlag )'
    
    
// rotate projectile and target about fixed angles if defined
#ifdef ROTATED_CONFIGURATION
    rotate_sphere(p, N_p, P_Z_ANGLE*M_PI/180.0, P_Y_ANGLE*M_PI/180.0, P_X_ANGLE*M_PI/180.0);    // rotate projectile
    rotate_sphere(p+N_p, N_t, T_Z_ANGLE*M_PI/180.0, T_Y_ANGLE*M_PI/180.0, T_X_ANGLE*M_PI/180.0);    // rotate target
#endif
    
    
// generate projectile/target rotation if set
    if( p_rot_period > 0.0 )
    {
        for(i=0; i<DIM; i++)
            p_omega[i] = p_rot_axis[i];
        
        // give 'p_omega' the proper length to make it the angular velocity vector
        tmp = sqrt( p_omega[0]*p_omega[0] + p_omega[1]*p_omega[1] + p_omega[2]*p_omega[2] );
        for(i=0; i<DIM; i++)
            p_omega[i] *= 2.0 * M_PI / p_rot_period / tmp;
        
        // compute particle velocities via cross-product (v = omega x r)
        for(i=0; i<N_p; i++)
        {
            p[i].v[0] = p_omega[1]*p[i].x[2] - p_omega[2]*p[i].x[1];
            p[i].v[1] = p_omega[2]*p[i].x[0] - p_omega[0]*p[i].x[2];
            p[i].v[2] = p_omega[0]*p[i].x[1] - p_omega[1]*p[i].x[0];
        }
    }
    if( t_rot_period > 0.0 )
    {
        for(i=0; i<DIM; i++)
            t_omega[i] = t_rot_axis[i];
        
        // give 't_omega' the proper length to make it the angular velocity vector
        tmp = sqrt( t_omega[0]*t_omega[0] + t_omega[1]*t_omega[1] + t_omega[2]*t_omega[2] );
        for(i=0; i<DIM; i++)
            t_omega[i] *= 2.0 * M_PI / t_rot_period / tmp;
        
        // compute particle velocities via cross-product (v = omega x r)
        for(i=N_p; i<(N_p+N_t); i++)
        {
            p[i].v[0] = t_omega[1]*p[i].x[2] - t_omega[2]*p[i].x[1];
            p[i].v[1] = t_omega[2]*p[i].x[0] - t_omega[0]*p[i].x[2];
            p[i].v[2] = t_omega[0]*p[i].x[1] - t_omega[1]*p[i].x[0];
        }
    }
    
    
// calculate corrected masses and radii of proj/targ due to the actual particle numbers (which in general deviate from the desired ones)
    if( hydrostructFlag || useProfilesFlag )
    {
        M_p = M_p_c = M_p_m = 0.0;
        for(i=0; i<N_p; i++)
        {
            M_p += p[i].mass;
            if( p[i].mat_type == mat[CORE].mat_type )
                M_p_c += p[i].mass;
            if( p[i].mat_type == mat[MANTLE].mat_type )
                M_p_m += p[i].mass;
        }
        M_t = M_t_c = M_t_m = 0.0;
        for(i=N_p; i<(N_p+N_t); i++)
        {
            M_t += p[i].mass;
            if( p[i].mat_type == mat[CORE].mat_type )
                M_t_c += p[i].mass;
            if( p[i].mat_type == mat[MANTLE].mat_type )
                M_t_m += p[i].mass;
        }
    }
    else
    {
        M_p_c = N_p_c * M_particle_p[CORE];
        M_t_c = N_t_c * M_particle_t[CORE];
        M_p_m = N_p_m * M_particle_p[MANTLE];
        M_t_m = N_t_m * M_particle_t[MANTLE];
        M_p = M_p_c + M_p_m + (N_p-N_p_c-N_p_m)*M_particle_p[SHELL];
        M_t = M_t_c + M_t_m + (N_t-N_t_c-N_t_m)*M_particle_t[SHELL];
    }
    M = M_p + M_t;
    
    if( ParticleGeometry == 0 )     // SC lattice
    {
        // Calculation of the actual radii is identical with and without HYDROSTRUCT (or given radial profiles) since "ideal" lattices are implemented in both cases - the mpd is simply a bit smaller with HYDROSTRUCT
        R_p_c = mpd * cbrt( THROVER4PI*N_p_c );
        R_t_c = mpd * cbrt( THROVER4PI*N_t_c );
        R_p_m = mpd * cbrt( THROVER4PI*(N_p_c+N_p_m) );    // because core/mantle/shell particles all take the same volume
        R_t_m = mpd * cbrt( THROVER4PI*(N_t_c+N_t_m) );
        R_p = mpd * cbrt( THROVER4PI*N_p );
        R_t = mpd * cbrt( THROVER4PI*N_t );
    }
    else if( ParticleGeometry == 1 )    // HCP lattice
    {
        R_p_c = mpd * cbrt( THROVER4PI*N_p_c/sqrt(2.0) );
        R_t_c = mpd * cbrt( THROVER4PI*N_t_c/sqrt(2.0) );
        R_p_m = mpd * cbrt( THROVER4PI*(N_p_c+N_p_m)/sqrt(2.0) );    // because core/mantle/shell particles all take the same volume
        R_t_m = mpd * cbrt( THROVER4PI*(N_t_c+N_t_m)/sqrt(2.0) );
        R_p = mpd * cbrt( THROVER4PI*N_p/sqrt(2.0) );
        R_t = mpd * cbrt( THROVER4PI*N_t/sqrt(2.0) );
    }
    else if( ParticleGeometry == 2 )    // spherical shell setup
    {
        R_p_c = cbrt( N_p_c*V_particle_p*THROVER4PI );
        R_t_c = cbrt( N_t_c*V_particle_t*THROVER4PI );
        R_p_m = cbrt( (N_p_c+N_p_m)*V_particle_p*THROVER4PI );
        R_t_m = cbrt( (N_t_c+N_t_m)*V_particle_t*THROVER4PI );
        R_p = cbrt( N_p*V_particle_p*THROVER4PI );
        R_t = cbrt( N_t*V_particle_t*THROVER4PI );
    }
    
    
// if not in N_input mode, move proj and targ to the initial positions and velocities (where the targ is at the origin at rest)
    if( !N_input )
    {
        // calculate initial proj position (where the targ is at the origin)
        // NOTE: There's a minimum possible distance given by their radii + an additional sml distance.
        ini_dist = R_p + R_t + 1.5*MAX(MAX(mat[CORE].sml,mat[SHELL].sml),mat[MANTLE].sml);   // minimum possible initial distance
        des_ini_dist = (R_p + R_t)*ini_dist_fact;
        if ( des_ini_dist > ini_dist )
            ini_dist = des_ini_dist;
        if ( vel_vesc_angle )   // compute initial geometry in case of v/v_esc and impact angle as input parameters
        {
            collision_geometry(M_p, M_t, R_p, R_t, ini_dist, vel_vesc, impact_angle, &impact_par, &ini_vel, &impact_vel_abs);
            ini_vel = -1.0*ini_vel;
        }
        if ( ini_dist < impact_par )    // make sure that the actual initial distance is not smaller than the impact parameter
            ERRORTEXT("ERROR. The spheres' initial distance is smaller than the impact parameter. Geometrically impossible!\n")
        ini_pos_p[0] = impact_par;
        ini_pos_p[1] = sqrt(ini_dist*ini_dist - impact_par*impact_par);
        ini_pos_p[2] = 0.0;
        
        // now move the projectile to its initial position and set the initial velocity (ini_vel) in y-direction
        for(i=0; i<N_p; i++)
        {
            for(j=0; j<DIM; j++)
                p[i].x[j] += ini_pos_p[j];
            p[i].v[1] += ini_vel;
        }
        
        // set like that for now, but will be corrected below along with barycentric correction
        for(i=0; i<DIM; i++) {
            proj_pos_final[i] = ini_pos_p[i];
            targ_pos_final[i] = 0.0;
            targ_vel_final[i] = 0.0;
        }
        proj_vel_final[0] = proj_vel_final[2] = 0.0;
        proj_vel_final[1] = ini_vel;
    }
    
    
// if in N_input mode, move projectile and target to the correct positions and velocities (from the coordinates-file)
    if( N_input )
    {
        for(i=0; i<N_p; i++)
            for(j=0; j<DIM; j++) {
                p[i].x[j] += N_input_data[0].x[j];
                p[i].v[j] += N_input_data[0].v[j];
            }
        for( i=N_p; i<(N_p+N_t); i++)
            for( j=0; j<DIM; j++) {
                p[i].x[j] += N_input_data[1].x[j];
                p[i].v[j] += N_input_data[1].v[j];
            }
        
        for( i=0; i<DIM; i++) {
            proj_pos_final[i] = N_input_data[0].x[i];
            proj_vel_final[i] = N_input_data[0].v[i];
            targ_pos_final[i] = N_input_data[1].x[i];
            targ_vel_final[i] = N_input_data[1].v[i];
        }
    }
    
    
// apply a barycentric correction, i.e. transform pos and vel of all particles to a frame barycentric w.r.t. proj + target
    // first calculate the (proj + target) center of mass' postition and velocity in the initial frame
    for( i=0; i<DIM; i++)
        baryc_x[i] = baryc_v[i] = 0.0;
    for( i=0; i<(N_p+N_t); i++)
        for( j=0; j<DIM; j++) {
            baryc_x[j] += p[i].mass * p[i].x[j];
            baryc_v[j] += p[i].mass * p[i].v[j];
        }
    for( i=0; i<DIM; i++) {
        baryc_x[i] /= M;
        baryc_v[i] /= M;
    }
    
    // now perform a Galilean transformation (here: x'=x-x_b and v'=v-v_b) to the center-of-mass frame (x' and v') for all particles
    if( !N_input || (N_input && b_flag) )
    {
        for( i=0; i<N; i++)
            for( j=0; j<DIM; j++) {
                p[i].x[j] -= baryc_x[j];
                p[i].v[j] -= baryc_v[j];
            }
        
        for( i=0; i<DIM; i++) {
            proj_pos_final[i] -= baryc_x[i];
            proj_vel_final[i] -= baryc_v[i];
            targ_pos_final[i] -= baryc_x[i];
            targ_vel_final[i] -= baryc_v[i];
        }
    }
    
    
// initialize yet missing components of the particle array
    for( i=0; i<N; i++)
    {
        p[i].damage = 0.0;
#ifdef MILUPH
        p[i].plastic_strain = 0.0;
        p[i].temp = TEMP;
#endif
        for( j=0; j<DIM; j++)
            for( k=0; k<DIM; k++)
                p[i].S[j][k] = 0.0;
        p[i].flaws.n_flaws = 0;
    }
    
    
// print scenario information
    fprintf(stdout, "----------------------------------------------------------------\n");
    fprintf(stdout, "Particle numbers:\n");
    if( !N_input )
        fprintf(stdout, "    desired total N = %d\t actual/final total N = %d\n", N_des, N);
    else if( N_input  &&  !M_output )
        fprintf(stdout, "    desired N_p+N_t = %d\t actual/final N_p+N_t = %d\t N_other_bodies (added as SPH particles) = %d\t actual/final total N = %d\n", N_des, N_p+N_t, N_bodies-2, N);
    else if( N_input  &&  M_output )
        fprintf(stdout, "    desired N_p+N_t = %d\t actual/final N_p+N_t = %d\t N_other_bodies (written to pointmasses-file) = %d\n", N_des, N, N_bodies-2);
    fprintf(stdout, "    projectile: N_des = %d\t N = %d\t N_core = %d\t N_mantle = %d\t N_shell = %d\n", N_p_des, N_p, N_p_c, N_p_m, N_p-N_p_c-N_p_m);
    fprintf(stdout, "    target:     N_des = %d\t N = %d\t N_core = %d\t N_mantle = %d\t N_shell = %d\n", N_t_des, N_t, N_t_c, N_t_m, N_t-N_t_c-N_t_m);
    fprintf(stdout, "----------------------------------------------------------------\n");
    fprintf(stdout, "Materials:\n");
    fprintf(stdout, "    core/mantle/shell:  \"%s\"/\"%s\"/\"%s\"\n", mat[CORE].mat_name, mat[MANTLE].mat_name, mat[SHELL].mat_name);
    fprintf(stdout, "    core:   mat. type = %d\t rho_0 = %g\t cs = %e\t eos = %c\n", mat[CORE].mat_type, mat[CORE].rho_0, mat[CORE].cs, mat[CORE].eos);
    if( mat[CORE].eos == 'A' && hydrostructFlag )
        fprintf(stdout, "          table-file = %s\t n_rho = %d\t n_e = %d\n", mat[CORE].aneos.table_file, mat[CORE].aneos.n_rho, mat[CORE].aneos.n_e);
    if( mat[CORE].eos == 'I' )
        fprintf(stdout, "          p_0 = %g\t gamma = %g\t polytropic_K = %g\n", mat[CORE].ideal_gas.p_0, mat[CORE].ideal_gas.gamma, mat[CORE].ideal_gas.polytropic_K);
    fprintf(stdout, "    mantle: mat. type = %d\t rho_0 = %g\t cs = %e\t eos = %c\n", mat[MANTLE].mat_type, mat[MANTLE].rho_0, mat[MANTLE].cs, mat[MANTLE].eos);
    if( mat[MANTLE].eos == 'A' && hydrostructFlag )
        fprintf(stdout, "          table-file = %s\t n_rho = %d\t n_e = %d\n", mat[MANTLE].aneos.table_file, mat[MANTLE].aneos.n_rho, mat[MANTLE].aneos.n_e);
    if( mat[MANTLE].eos == 'I' )
        fprintf(stdout, "          p_0 = %g\t gamma = %g\t polytropic_K = %g\n", mat[MANTLE].ideal_gas.p_0, mat[MANTLE].ideal_gas.gamma, mat[MANTLE].ideal_gas.polytropic_K);
    fprintf(stdout, "    shell:  mat. type = %d\t rho_0 = %g\t cs = %e\t eos = %c\n", mat[SHELL].mat_type, mat[SHELL].rho_0, mat[SHELL].cs, mat[SHELL].eos);
    if( mat[SHELL].eos == 'A' && hydrostructFlag )
        fprintf(stdout, "          table-file = %s\t n_rho = %d\t n_e = %d\n", mat[SHELL].aneos.table_file, mat[SHELL].aneos.n_rho, mat[SHELL].aneos.n_e);
    if( mat[SHELL].eos == 'I' )
        fprintf(stdout, "          p_0 = %g\t gamma = %g\t polytropic_K = %g\n", mat[SHELL].ideal_gas.p_0, mat[SHELL].ideal_gas.gamma, mat[SHELL].ideal_gas.polytropic_K);
#ifdef MILUPH
    fprintf(stdout, "    all:  artificial viscosity: alpha = %g\n", mat->alpha);
    fprintf(stdout, "                                 beta = %g\n", mat->beta);
#endif
    fprintf(stdout, "----------------------------------------------------------------\n");
    fprintf(stdout, "Masses:\n");
    fprintf(stdout, "    total: desired:      M = %e\n", M_des);
    fprintf(stdout, "           actual/final: M = %e\n", M);
    fprintf(stdout, "    projectile: desired:      M = %e\t M_core = %e\t M_mantle = %e\t M_shell = %e\n", M_p_des, M_p_c_des, M_p_m_des, M_p_s_des);
    fprintf(stdout, "                actual/final: M = %e\t M_core = %e\t M_mantle = %e\t M_shell = %e\n", M_p, M_p_c, M_p_m, M_p-M_p_c-M_p_m);
    fprintf(stdout, "    target: desired:      M = %e\t M_core = %e\t M_mantle = %e\t M_shell = %e\n", M_t_des, M_t_c_des, M_t_m_des, M_t_s_des);
    fprintf(stdout, "            actual/final: M = %e\t M_core = %e\t M_mantle = %e\t M_shell = %e\n", M_t, M_t_c, M_t_m, M_t-M_t_c-M_t_m);
    if( !hydrostructFlag && !useProfilesFlag )
    {
        fprintf(stdout, "    single particle masses:  projectile:  core = %e\t mantle = %e\t shell = %e\n", M_particle_p[CORE], M_particle_p[MANTLE], M_particle_p[SHELL]);
        fprintf(stdout, "                                 target:  core = %e\t mantle = %e\t shell = %e\n", M_particle_t[CORE], M_particle_t[MANTLE], M_particle_t[SHELL]);
    }
    fprintf(stdout, "Mantle/shell mass fractions:\n");
    fprintf(stdout, "    projectile: mantle: desired = %g\t actual/final = %g\n", C_p_m_des, M_p_m/M_p);
    fprintf(stdout, "                 shell: desired = %g\t actual/final = %g\n", C_p_s_des, (M_p-M_p_c-M_p_m)/M_p);
    fprintf(stdout, "    target: mantle: desired = %g\t actual/final = %g\n", C_t_m_des, M_t_m/M_t);
    fprintf(stdout, "             shell: desired = %g\t actual/final = %g\n", C_t_s_des, (M_t-M_t_c-M_t_m)/M_t);
    fprintf(stdout, "----------------------------------------------------------------\n");
    fprintf(stdout, "Radii:\n");
    fprintf(stdout, "    projectile: desired:      R = %e\t R_core = %e\t R_mantle = %e\n", R_p_des, R_p_c_des, R_p_m_des);
    fprintf(stdout, "                actual/final: R = %e\t R_core = %e\t R_mantle = %e\n", R_p, R_p_c, R_p_m);
    fprintf(stdout, "    target: desired:      R = %e\t R_core = %e\t R_mantle = %e\n", R_t_des, R_t_c_des, R_t_m_des);
    fprintf(stdout, "            actual/final: R = %e\t R_core = %e\t R_mantle = %e\n", R_t, R_t_c, R_t_m);
    fprintf(stdout, "    sum of actual/final radii = %e\n", R_p+R_t);
    
    fprintf(stdout, "----------------------------------------------------------------\n");
    fprintf(stdout, "Geometry:\n");
    
    if( N_input )
    {
        fprintf(stdout, "    Initial positions and velocities of proj + target + additional pointmasses are given in cartesian coordinates via file '%s'.\n", coordfile);
        if( b_flag )
            fprintf(stdout, "    This whole arrangement is set up in a frame that is (initially) barycentric w.r.t. proj + target alone.\n");
        if( M_output )
            fprintf(stdout, "    Data on the %d additional pointmasses is written to the file '%s' for processing by miluphcuda.\n", N_bodies-2, pointmassesfile);
        else
            fprintf(stdout, "    The %d additional pointmasses are included as single SPH particles each.\n", N_bodies-2);
        fprintf(stdout, "\n    The relative two-body orbit of proj + targ (i.e., neglecting all additional pointmasses) is used to estimate collision parameters ...\n");
        collision_parameters_from_cartesian(M_p, M_t, R_p, R_t, N_input_data[0].x, N_input_data[1].x, N_input_data[0].v, N_input_data[1].v, &N_input_impact_angle, &N_input_impact_vel_vesc, &N_input_impact_vel_abs);
        if( N_input_impact_angle >= 0.0 ) {
            fprintf(stdout, "    It's a physical collision (pericenter distance < R_p+R_t) with parameters:\n");
            fprintf(stdout, "    impact angle = %e deg\n    impact velocity = %e\n    v/v_esc = %e\n", N_input_impact_angle*180.0/M_PI, N_input_impact_vel_abs, N_input_impact_vel_vesc);
            fprintf(stdout, "    collision timescale (R_p+R_t)/|v_imp| = %g sec\n\n", (R_p+R_t)/fabs(N_input_impact_vel_abs) );
        }
        else {
            fprintf(stdout, "    It's NOT a physical collision (pericenter distance > R_p+R_t = %e) with parameters at pericenter:\n", R_p+R_t);
            fprintf(stdout, "    relative velocity = %e\n    v/v_esc = %e\n", N_input_impact_vel_abs, N_input_impact_vel_vesc);
            fprintf(stdout, "    collision timescale (R_p+R_t)/|v_pericenter| = %g sec\n\n", (R_p+R_t)/fabs(N_input_impact_vel_abs) );
        }
    }
    else
    {
        if( vel_vesc_angle )
            fprintf(stdout, "    At \"touching ball\" distance (R_p+R_t = %e):\n        v_imp = %e\n        v_imp/v_esc = %e\n        impact angle = %e deg\n", R_p+R_t, impact_vel_abs, vel_vesc, impact_angle);
        fprintf(stdout, "    At initial distance (ini_dist = %e):\n        ini_vel = %e\n        impact parameter = %e\n", ini_dist, ini_vel, impact_par);
        fprintf(stdout, "        collision timescale (R_p+R_t)/|v_imp| = %g sec\n", (R_p+R_t)/fabs(impact_vel_abs) );
    }
    
    if( !N_input )
        fprintf(stdout, "    projectile position before barycentric correction =  %24.16le %24.16le %24.16le\n", ini_pos_p[0], ini_pos_p[1], ini_pos_p[2]);
    if( !N_input || (N_input && b_flag) )
    {
        fprintf(stdout, "    Barycentric correction applied (w.r.t. proj + target). Barycenter initially at:\n");
        fprintf(stdout, "           x/y/z = %24.16le %24.16le %24.16le\n", baryc_x[0], baryc_x[1], baryc_x[2]);
        fprintf(stdout, "        vx/vy/vz = %24.16le %24.16le %24.16le\n", baryc_v[0], baryc_v[1], baryc_v[2]);
    }
    
    fprintf(stdout, "    Final positions and velocities:\n");
    fprintf(stdout, "        projectile:  x/y/z = %24.16le %24.16le %24.16le    vx/vy/vz = %24.16le %24.16le %24.16le\n",
            proj_pos_final[0], proj_pos_final[1], proj_pos_final[2], proj_vel_final[0], proj_vel_final[1], proj_vel_final[2] );
    fprintf(stdout, "            target:  x/y/z = %24.16le %24.16le %24.16le    vx/vy/vz = %24.16le %24.16le %24.16le\n",
            targ_pos_final[0], targ_pos_final[1], targ_pos_final[2], targ_vel_final[0], targ_vel_final[1], targ_vel_final[2] );
    
    fprintf(stdout, "----------------------------------------------------------------\n");
    if( ParticleGeometry == 0 )
        fprintf(stdout, "Initial lattice structure:\n    SIMPLE CUBIC\n");
    if( ParticleGeometry == 1 )
        fprintf(stdout, "Initial lattice structure:\n    HEXAGONALLY CLOSE-PACKED\n");
    if( ParticleGeometry == 2 )
        fprintf(stdout, "Initial particle geometry:\n    SPHERICAL SHELL SETUP with SEAGen\n");
    fprintf(stdout, "    mean particle dist. mpd = %e\t sml = %e ( = mpd * %e )\n", mpd, mat->sml, sml_factor);
    if( ParticleGeometry == 2 )     // spherical shell setup
        fprintf(stdout, "                      ( mpd = MAX(mpd-proj,mpd-targ) = MAX(%e,%e) )\n", mpd_p, mpd_t);
    
#ifdef ROTATED_CONFIGURATION
    fprintf(stdout, "----------------------------------------------------------------\n");
    fprintf(stdout, "Rotated (by a fixed angle) initial configuration used. Angles (deg):\n");
    fprintf(stdout, "    target:     z/y/x =  %g  %g  %g\n", T_Z_ANGLE, T_Y_ANGLE, T_X_ANGLE);
    fprintf(stdout, "    projectile: z/y/x =  %g  %g  %g\n", P_Z_ANGLE, P_Y_ANGLE, P_X_ANGLE);
#endif
    
    fprintf(stdout, "----------------------------------------------------------------\n");
    fprintf(stdout, "Initial rotation:\n");
    if( p_rot_period > 0.0 ) {
        fprintf(stdout, "    projectile: period = %g sec\n", p_rot_period);
        fprintf(stdout, "                rotation-axis =  %g  %g  %g\n", p_rot_axis[0], p_rot_axis[1], p_rot_axis[2] );
    } else {
        fprintf(stdout, "    None for projectile.\n");
    }
    if( t_rot_period > 0.0 ) {
        fprintf(stdout, "    target: period = %g sec\n", t_rot_period);
        fprintf(stdout, "            rotation-axis =  %g  %g  %g\n", t_rot_axis[0], t_rot_axis[1], t_rot_axis[2] );
    } else {
        fprintf(stdout, "    None for target.\n");
    }
    
    fprintf(stdout, "----------------------------------------------------------------\n");
    fprintf(stdout, "Relaxation technique:\n");
    if( hydrostructFlag ) {
        fprintf(stdout, "    Calculate hydrostatic structure and set particle densities/masses accordingly.\n");
        fprintf(stdout, "    Calculate and set internal energies following adiabatic compression.\n");
    } else if( useProfilesFlag ) {
        fprintf(stdout, "    Use given radial profiles to set densities, masses, and internal energies.\n");
    } else {
        fprintf(stdout, "    None.\n");
    }
    
    fprintf(stdout, "----------------------------------------------------------------\n");
    fprintf(stdout, "Damage model:\n");
    fprintf(stdout, "    weibulling core material:  ");
    if (weibull_core == 1)
        fprintf(stdout, "yes\t k = %g\t m = %g\n", mat[CORE].k, mat[CORE].m);
    else
        fprintf(stdout, "no\n");
    fprintf(stdout, "    weibulling mantle material:  ");
    if (weibull_mantle == 1)
        fprintf(stdout, "yes\t k = %g\t m = %g\n", mat[MANTLE].k, mat[MANTLE].m);
    else
        fprintf(stdout, "no\n");
    fprintf(stdout, "    weibulling shell material:  ");
    if (weibull_shell == 1)
        fprintf(stdout, "yes\t k = %g\t m = %g\n", mat[SHELL].k, mat[SHELL].m);
    else
        fprintf(stdout, "no\n");
    fprintf(stdout, "----------------------------------------------------------------\n");
    if( N_input == TRUE )   // compute ini_vel for courant-like criterion
    {
        for(i=0; i<DIM; i++)
            ini_vel_vec[i] = N_input_data[0].v[i] - N_input_data[1].v[i];
        ini_vel = sqrt(ini_vel_vec[0]*ini_vel_vec[0] + ini_vel_vec[1]*ini_vel_vec[1] + ini_vel_vec[2]*ini_vel_vec[2]);
    }
    fprintf(stdout, "A courant-like criterion suggests:\t Delta_t < %e\n", mpd/MAX( MAX(mat[CORE].cs, MAX(mat[MANTLE].cs,mat[SHELL].cs) ),fabs(ini_vel) ));
    
    
// weibull particles if desired and write (weibulled or not) data to output file
    V_p_c_uncomp = M_p_c/mat[CORE].rho_0;
    V_p_m_uncomp = M_p_m/mat[MANTLE].rho_0;
    V_p_s_uncomp = (M_p-M_p_c-M_p_m)/mat[SHELL].rho_0;
    V_t_c_uncomp = M_t_c/mat[CORE].rho_0;
    V_t_m_uncomp = M_t_m/mat[MANTLE].rho_0;
    V_t_s_uncomp = (M_t-M_t_c-M_t_m)/mat[SHELL].rho_0;
    if( weibull_core == 1 )
    {
        weibull_particles(p, &mat[CORE], V_p_c_uncomp, N_p, N_p_c, " the projectile's core,");    // weibull projectile's core
        weibull_particles(p+N_p, &mat[CORE], V_t_c_uncomp, N_t, N_t_c, " the target's core,");    // weibull target's core
    }
    if( weibull_mantle == 1 )
    {
        weibull_particles(p, &mat[MANTLE], V_p_m_uncomp, N_p, N_p_m, " the projectile's mantle,");    // weibull projectile's mantle
        weibull_particles(p+N_p, &mat[MANTLE], V_t_m_uncomp, N_t, N_t_m, " the target's mantle,");    // weibull target's mantle
    }
    if( weibull_shell == 1 )
    {
        weibull_particles(p, &mat[SHELL], V_p_s_uncomp, N_p, N_p-N_p_c-N_p_m, " the projectile's shell,");    // weibull projectile's shell
        weibull_particles(p+N_p, &mat[SHELL], V_t_s_uncomp, N_t, N_t-N_t_c-N_t_m, " the target's shell,");    // weibull target's shell
    }
    if ( (ofl = fopen(outfile,"w")) == NULL )
        ERRORVAR("FILE ERROR! Cannot open '%s' for writing!\n", outfile)
    write_outfile(ofl, p, N, OutputMode);    // write whole p (all information on all particles) to the output file
    fclose(ofl);
    
    
// clean up
#ifdef MILUPH
    if( mat[CORE].eos == 'A' && hydrostructFlag )
        free_ANEOS_table_memory(&mat[CORE]);
    if( mat[MANTLE].eos == 'A' && hydrostructFlag )
        free_ANEOS_table_memory(&mat[MANTLE]);
    if( mat[SHELL].eos == 'A' && hydrostructFlag )
        free_ANEOS_table_memory(&mat[SHELL]);
#endif
#ifdef MILUPHCUDA
    if( allocated_ANEOS_mem_core_flag )
        free_ANEOS_table_memory(&mat[CORE]);
    if( allocated_ANEOS_mem_mantle_flag )
        free_ANEOS_table_memory(&mat[MANTLE]);
    if( allocated_ANEOS_mem_shell_flag )
        free_ANEOS_table_memory(&mat[SHELL]);
#endif
    free(p);
    if( N_input == TRUE )
        free(N_input_data);
    if( useProfilesFlag )
    {
        if( PROFILE_FILE_PROJ != 0 )    // there is a projectile
            free(profile_projectile);
        if( PROFILE_FILE_TARG != 0 )    // there is a target
            free(profile_target);
    }
    return(0);
    
}   // end 'main()'




void weibull_particles(particle* p, material* mat, double volume, int n_all_p, int n_mat_p, const char* message)
// Distributes flaws following the Weibull distribution to particles starting at address p of a certain material mat->mat_type until every particle of that material (mat) has at least one flaw.
// n_all_p is the overall particle number to consider (starting at address p), n_mat_p is the number of particles of the material type that is expected to/shall receive flaws.
// volume contains the total volume of the material that shall be weibulled, message optionally contains text to insert in the output message below.
{
    int i,j;
    int n_miss = n_mat_p;    // n_miss = number of particles left without any flaws
    double act_thr;
    
    // checking whether the compiler-dependent resolution of the rand() generator poses a threat to the statistical quality of the random numbers:
    // see below: n_all_p/(RAND_MAX+1) produces a constant<1 (otherwise it would fail entirely). Multiplied by rand() and then truncated some indices become more likely than others!
    // e.g. drand48 would be an alternative ...
    if (n_all_p > 0.1*RAND_MAX)
    {
        fprintf(stderr, "WARNING! The random number generator for weibulling produces int values only in the range [0,%d]. ", RAND_MAX);
        fprintf(stderr, "This limited resolution could threaten the statistical quality. Program stopped, think about it.\n");
        exit(1);
    }
    
    // checking whether the number of particles in p with desired material equals n_mat_p:
    for(i=j=0; i<n_all_p; i++)
        if ( p[i].mat_type == mat->mat_type )
            j++;
    if ( j != n_mat_p )
        ERRORTEXT("ERROR! Strange particle number mismatch during weibulling.\n")
    
    fprintf(stdout, "--------------------------------\n");
    fprintf(stdout, "Now weibulling%s material '%s', material type %d ... ", message, mat->mat_name, mat->mat_type);
    
    j=0;
    while(n_miss)    // distribute flaws until every particle has at least one flaw, i.e. until n_miss==0
    {
        i = (int) ( (double)n_all_p * rand()/(RAND_MAX+1.0) );    // generating random particle index in p, (int) truncates
        if ( p[i].mat_type == mat->mat_type )
        {
            if (p[i].flaws.n_flaws == 0)
            {
                n_miss--;
                if ( (p[i].flaws.act_thr = (double*)malloc( sizeof(double) )) == NULL )
                    ERRORTEXT("ERROR during memory allocation for flaw activation thresholds!\n")
            }
            else
                if ( (p[i].flaws.act_thr = realloc( p[i].flaws.act_thr, sizeof(double)*(p[i].flaws.n_flaws+1) )) == NULL )
                    ERRORTEXT("ERROR during memory allocation for flaw activation thresholds!\n")
            j++;
            act_thr = pow(j/(mat->k)/volume,1.0/(mat->m));    // activation threshold according to Weibull distribution (for flaw j)
            
            p[i].flaws.act_thr[p[i].flaws.n_flaws] = act_thr;
            p[i].flaws.n_flaws++;
        }
    }
    fprintf(stdout, "Done.\n");
    fprintf(stdout, "Distributed %d flaws for %d particles.\n", j, n_mat_p);
    fprintf(stdout, "Mean number of flaws per particle: %g\n", (double)j/n_mat_p);
}




void set_profile_rho_e(particle* p, int i, double r2, radial_profile_data *profile, int n)
// This function sets (linearly interpolates) rho and e for one particle 'p[i]' following the radial profile in 'profile' (length 'n'). 'p' is the absolute (first) address
// of the particle vector, 'i' is the index/element of this vector for which rho and e should be found, 'r2' is the squared distance to the origin.
{
    int j;
    
    j=0;
    while( j<n )
    {
        if ( r2 < pow(profile[j].r,2) )
            break;
        j++;
    }
    
    if( j==0 )    // particle inside the innermost radius in 'profile'
    {
        p[i].rho = profile[0].rho;
        p[i].e = profile[0].e;
        fprintf(stderr, "WARNING: Particle with index %d has r = %.16le which is inside the innermost datapoint (at r = %.16le) in the radial profile. Assigned rho = %e and e = %e to it.\n", 
                i, sqrt(r2), profile[0].r, p[i].rho, p[i].e);
    }
    else if( j<n )    //particle somewhere between two radii in 'profile'
    {
        p[i].rho = profile[j-1].rho + (profile[j].rho-profile[j-1].rho)/(profile[j].r-profile[j-1].r) * (sqrt(r2)-profile[j-1].r);
        p[i].e = profile[j-1].e + (profile[j].e-profile[j-1].e)/(profile[j].r-profile[j-1].r) * (sqrt(r2)-profile[j-1].r);
    }
    else    // particle outside the outermost radius in 'profile'
    {
        p[i].rho = profile[n-1].rho;
        p[i].e = profile[n-1].e;
        fprintf(stderr, "WARNING: Particle with index %d has r = %.16le which is outside the outermost datapoint (at r = %.16le) in the radial profile. Assigned rho = %e and e = %e to it.\n", 
                i, sqrt(r2), profile[n-1].r, p[i].rho, p[i].e);
    }
}

