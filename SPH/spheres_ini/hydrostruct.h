#ifndef _HYDROSTRUCT_H
#define _HYDROSTRUCT_H

double Tillotson_pressure(double e, double rho, material* mat);
double Tillotson_density(double e, double p, material* mat);
double ANEOS_pressure(double e, double rho, material* mat);
double ANEOS_density(double e, double p, material* mat);
double ideal_gas_pressure(double e, double rho, material* mat);
double ideal_gas_density(double e, double p, material* mat);
void rhs_lagrange(double m, double r, double p, double* f, material* mat, material_fp* mat_fp, double rho_approx);
void RK4_step_lagrange(double h, int_struct_point* s, material* mat, material_fp* mat_fp);
void calc_internal_structure(double M, double M_c, double M_m, double R_lb, double R_ub, material* mat, material_fp* mat_fp, int_struct_point* int_struct_c, int_struct_point* int_struct_m, int_struct_point* int_struct_s, int n_steps);
void set_hydrostruct_density(particle* p, int i, double r2, int_struct_point* int_struct);
double e_compression_Tillotson(double rho_des, material* mat);
double e_compression_ANEOS(double rho_des, material* mat);
double e_compression_ideal_gas(double rho_des, material* mat);
double rho_self_consistent_Tillotson(double p, double rho_start, material* mat);
double rho_self_consistent_ANEOS(double p, double rho_start, material* mat);
double rho_self_consistent_ideal_gas(double p, double rho_start, material* mat);
int array_index(double x, double* array, int n);
double bilinear_interpolation(double x, double y, double** table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y);
int discrete_value_table_lookup(double x, double y, int** table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y);


#endif

