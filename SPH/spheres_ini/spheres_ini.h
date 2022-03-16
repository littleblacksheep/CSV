#ifndef _SPHERES_INI_H
#define _SPHERES_INI_H


#include <stdio.h>
#include <stdlib.h>



// choose output file format - miluph vs. miluphcuda
#define MILUPHCUDA
//#define MILUPH
#if defined MILUPHCUDA && defined MILUPH
    #error "Choose either MILUPH or MILUPHCUDA as output file format in 'spheres_ini.h', not both!"
#endif
#if !defined MILUPHCUDA && !defined MILUPH
    #error "Choose either MILUPH or MILUPHCUDA as output file format in 'spheres_ini.h'!"
#endif


// additional parameters for hydrostruct computation
#define MAXITERATIONS 100    // max number of iterations during the hydrostatic structure calculation before the program is stopped
#define NSTEPS 1000    // number of steps for hydrostatic structure calculation (for core/mantle/shell separately - i.e. each with NSTEPS)


// rotate the generated spheres (extrinsically) about the subsequently defined angles
// NOTE: This is a rotation about a fixed angle to reduce symmetry effects (from particles arranged in lattices), and not related to actually rotating bodies!
#define ROTATED_CONFIGURATION
// rotation angles (T/P for target/projectile), angles in degrees:
#define T_Z_ANGLE 20.0
#define T_Y_ANGLE 20.0
#define T_X_ANGLE 20.0
#define P_Z_ANGLE -20.0
#define P_Y_ANGLE -20.0
#define P_X_ANGLE -20.0


// density and material type used for additional bodies/pointmasses - used if the additional pointmasses are represented by single SPH particles
#define N_BODIES_RHO 7800.0
#define N_BODIES_MATTYPE 3  // best use the highest not already used mattype (material ID), and copy e.g. Iron there in 'material.cfg'. Also set sml by hand!
// r_min and r_max used for additional bodies/pointmasses - used if the additional pointmasses are written to a separate output file (pointmasses-file)
#define N_BODIES_RMIN 1.0e6
#define N_BODIES_RMAX 1.0e9
#define N_BODIES_FEELING_FLAG 1  // last column in pointmasses-file; specifies whether the point-masses also feel gravity by the SPH particles in miluphcuda or not (1/0)


// settings for reading profiles from files - set file path and radii if body exists, and 0 for file path and 0.0 for radii if it doesn't - the format is: first line ignored, then columns r-rho-e
// NOTE: make sure that at material boundaries there is a double entry - one for the end of one material and another one for the start of the next one
#define PROFILE_FILE_PROJ "projectile.profile"
#define PROFILE_FILE_TARG "target.profile"
#define PROFILE_R_P 4.141247e+05     // radius of projectile
#define PROFILE_R_P_C 0.0   // radius of projectile core
#define PROFILE_R_P_M 3.619571e+05   // radius of projectile mantle
#define PROFILE_R_T 3.985e+06
#define PROFILE_R_T_C 2.285e6
#define PROFILE_R_T_M 3.895e6


// options relevant only for miluph (for miluphcuda they are read from the materialconfiguration file!)
#define TILL_RHO_LIMIT 0.9    // limit for low-density pressure cutoff in Tillotson eos
#define TEMP 0.0    // temperature value in output file
#define ART_VISC_ALPHA 1.0  // artificial viscosity parameters (same for all materials)
#define ART_VISC_BETA 2.0
// ideal gas eos parameters (also relevant only for miluph, because read from materialconfiguration file for miluphcuda!)
#define I_GAMMA 1.4
#define I_RHO0 0.2  // density used at outer boundary, or for homogeneous density bodies, respectively
#define I_P0 1.0e5   // pressure at outer boundary, or for homogeneous density bodies, respectively
#define CONVERSION_E_TO_T 0.00009698    // conversion factor from e to T for an ideal H2 gas (m_H2=3.347e-27, f=5)


// options relevant only for miluphcuda
#define EOS_TYPE_MURNAGHAN 1
#define EOS_TYPE_TILLOTSON 2
#define EOS_TYPE_ANEOS 7
#define EOS_TYPE_IDEAL_GAS 9


// miscellaneous
#define DIM 3   // not reliable for values other than 3
#define NMAT 3    // max. number of different materials
#define GRAV_CONST_SI 6.6741e-11
#define MATTYPECORE 0
#define MATTYPEMANTLE 1
#define MATTYPESHELL 2
#define CORE 0
#define MANTLE 1
#define SHELL 2
#define MEMALLOCFACT 2.5    // determines how much more memory for the particles is allocated than theoretically necessary for the desired N
#define TRUE 1
#define FALSE 0
#define PATHLENGTH 256


#define THROVER4PI (3.0/(4.0*M_PI))
#define MAX(x,y) ((x)>(y) ? (x):(y))
#define ERRORTEXT(x) {fprintf(stderr,x); exit(1);}
#define ERRORVAR(x,y) {fprintf(stderr,x,y); exit(1);}
#define ERRORVAR2(x,y,z) {fprintf(stderr,x,y,z); exit(1);}
#define ERRORVAR3(x,y,z,a) {fprintf(stderr,x,y,z,a); exit(1);}
#define ERRORVAR4(x,y,z,a,b) {fprintf(stderr,x,y,z,a,b); exit(1);}


// debugging flags
//#define HYDROSTRUCT_DEBUG   // prints a lot information about the hydrostatic structure computation
//#define MORE_HYDROSTRUCT_DEBUG    // prints even more information about the hydrostatic structure computation


typedef struct tillotson_eos_data
{
    double rho_0;
    double e_0, e_iv, e_cv;
    double a, b, A, B;
    double alpha, beta;
    double rho_limit;
} tillotson_data;

typedef struct aneos_eos_data
{
    double rho_0;   // equal to rho at norm condition
    double e_norm;  // spec. int. energy at norm condition
    double bulk_cs;
    char table_file[PATHLENGTH];
    int n_rho;
    int n_e;
    double *rho;
    double *e;
    double **p;
    double **T;
    double **cs;
    double **entropy;
    int **phase_flag;
} aneos_data;

typedef struct ideal_gas_eos_data
{
    double gamma;
    double rho_0;   // density used at outer boundary, or for homogeneous density bodies, respectively
    double p_0;   // pressure at outer boundary, or for homogeneous density bodies, respectively
    double polytropic_K;    // proportionality constant in p = K*rho^gamma, used for adiabatic compression in calculation of e
    double conv_e_to_T; // conversion factor from e to T
} ideal_gas_data;

typedef struct material_data
{
    int mat_type;
    char mat_name[PATHLENGTH];
    char eos;   // one character identifying eos: A/M/T/I (ANEOS/Murnaghan/Tillotson/Ideal-gas)
    int eos_type;   // number identifying eos (see #defines); to check consistency with materialconfiguration file in case of miluphcuda
    double sml;
    double alpha,beta;  // artificial viscosity parameters
    double rho_0;
    double k,m; // weibull parameters
    double cs;
    tillotson_data till;
    aneos_data aneos;
    ideal_gas_data ideal_gas;
} material;

typedef struct material_function_pointer    // contains function pointer to various eos-related functions for one material
{
    double (*eos_pressure)(double,double,material*);
    double (*eos_density)(double,double,material*);
    double (*eos_rho_self_consistent)(double,double,material*);
    double (*eos_e_compression)(double,material*);
} material_fp;

typedef struct sph_particle
{
    double x[DIM];
    double v[DIM];
    double mass;
    double rho;
    double e;
    int mat_type;
    double damage;
#ifdef MILUPH
    double plastic_strain;
    double temp;
#endif
    double S[DIM][DIM];
    struct
    {
        int n_flaws;
        double* act_thr;    // activation thresholds
    } flaws;
} particle;

typedef struct internal_structure_point        // used for storing the results of the hydrostatic structure calculation
{
    double r, m;
    double rho, p;
} int_struct_point;

typedef struct N_input_coordinates
{
    double x[DIM];
    double v[DIM];
    double mass;
} N_input_coord;

typedef struct _radial_profile_data
{
    double r;
    double rho;
    double e;
} radial_profile_data;



void weibull_particles(particle* p, material* mat, double volume, int n_all_p, int n_mat_p, const char* message);
void set_profile_rho_e(particle* p, int i, double r2, radial_profile_data *profile, int n);


#endif
