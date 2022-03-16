#ifndef _IO_H
#define _IO_H

#include "spheres_ini.h"


void help(char* programname);
void read_inputfile(char* infile, int* N, double* M, double* M_p, double* c_p_m, double* c_p_s, double* c_t_m, double* c_t_s, double* ini_vel, 
    double* impact_par, double* vel_vesc, double* impact_angle, double *ini_dist_fact, int* weibull_core, int* weibull_mantle, int* weibull_shell, char* core_eos, 
    char* mantle_eos, char* shell_eos, char* mat_core, char* mat_mantle, char* mat_shell,
    double *rot_period_p, double *rot_period_t, double *rot_axis_p, double *rot_axis_t);
#ifdef MILUPH
void readMaterialConstants(FILE* f, material* mat, int weibull_it);
void write_scenario_data(FILE* f, material* mat, int nmat);
#endif
#ifdef MILUPHCUDA
void readMaterialConfiguration(char *matfile, material *mat, int weibull_it);
void pasteSml(char *matfile, material *mat);
#endif
void write_outfile(FILE* f, particle* p, int n, int OutputMode);
void allocate_ANEOS_table_memory(material *mat);
void free_ANEOS_table_memory(material *mat);
void load_ANEOS_table(material *mat);


#endif
