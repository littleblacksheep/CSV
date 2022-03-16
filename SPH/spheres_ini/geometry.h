#ifndef _GEOMETRY_H
#define _GEOMETRY_H

int add_sphere_particles_SC(particle* p, material* mat, double mpd, double R, double R_core, double R_mantle, int* N_core, int* N_mantle);
int add_sphere_particles_HCP(particle* p, material* mat, double mpd, double R, double R_core, double R_mantle, int* N_core, int* N_mantle);
int add_sphere_particles_SS(particle* p, int N_des, double R, double R_core, double R_mantle, char* source_dir, const char* outfile, int* N_core, int* N_mantle, double* mpd);
void matrix_times_vector(int n, double m[][n], double* v);
void rotate_sphere(particle* p, int n, double z_angle, double y_angle, double x_angle);
void collision_geometry(double m_p, double m_t, double R_p, double R_t, double ini_dist, double vel_vesc, double alpha, double* impact_par, double* ini_vel, double *impact_vel_abs);
void collision_parameters_from_cartesian(double m_p, double m_t, double R_p, double R_t, double* x_p, double* x_t, double* v_p, double* v_t, double* impact_angle, double* impact_vel_vesc, double* impact_vel_abs);


#endif

