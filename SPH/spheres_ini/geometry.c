// spheres_ini functions related to geometry.

// Christoph Burger 20/Feb/2020


#include <stdio.h>
#include <math.h>
#include <Python.h>
#include <wchar.h>

#include "spheres_ini.h"




int add_sphere_particles_SC(particle* p, material* mat, double mpd, double R, double R_core, double R_mantle, int* N_core, int* N_mantle)
// Adds SPH particles for a sphere (at the origin) to the array of particles, starting at address p.
// Particles are arranged in a simple-cubic equally-spaced lattice, symmetric around the center.
// The sphere (radius R) consists in general of a core (R_core, N_core), a mantle (R_mantle, N_mantle) and a shell. 
// Returns the number of overall added particles.
// Adds positions, all velocity components = 0, and material types.
// Enough allocated memory has to be provided externally - p is just incremented (locally) inside this function.
{
    double x[DIM];
    int i, n=0;
    double start, radius2;        // squared distance to center
    const double eps = 1.0e-5;
    
    if (R > eps)    // makes sure that limited machine precision doesn't produce particles when R is actually zero
    {
        start = 0.5*(double)ceil(2.0*R/mpd)*mpd;    // determines the start corner of the grid: (x/y/z)=(-start/-start/-start); this start corner makes the grid symmetric related to the sphere
        
        for( x[2]=-start; x[2]<(start+mpd*eps); x[2]+=mpd )
            for( x[1]=-start; x[1]<(start+mpd*eps); x[1]+=mpd )
                for( x[0]=-start; x[0]<(start+mpd*eps); x[0]+=mpd )
                {
                    radius2 = x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
                    if ( radius2 < R*R )    // particle inside the sphere if true
                    {
                        for(i=0; i<DIM; i++)
                        {
                            p->x[i] = x[i];
                            p->v[i] = 0.0;
                        }
                        if ( radius2 < R_core*R_core )    // particle in the core if true
                        {
                            (*N_core)++;
                            i=CORE;
                        }
                        else if ( radius2 < R_mantle*R_mantle )    // particle in the mantle if true
                        {
                            (*N_mantle)++;
                            i=MANTLE;
                        }
                        else    // particle in the shell if true
                            i=SHELL;
                        
                        p->mat_type = mat[i].mat_type;
                        n++;
                        p++;
                    }
                }
    }
    return(n);
}




int add_sphere_particles_HCP(particle* p, material* mat, double mpd, double R, double R_core, double R_mantle, int* N_core, int* N_mantle)
// Adds SPH particles for a sphere (at the origin) to the array of particles, starting at address p.
// Particles are arranged in a hexagonally close-packed equally-spaced lattice, symmetric around the center, with a particle at the very center.
// The sphere (radius R) consists in general of a core (R_core, N_core), a mantle (R_mantle, N_mantle) and a shell. 
// Returns the number of overall added particles.
// Adds positions, all velocity components = 0, and material types.
// Enough allocated memory has to be provided externally - p is just incremented (locally) inside this function.
{
    double x[DIM];
    int i, n=0;
    double radius2;        // squared distance to center
    double z_start, y_start, y_end, x_start, z_step, y_step, x_step;
    int z_count, y_count;    // counter for indicating currently processed z and y layer
    const double eps = 1.0e-5;
    
    if (R > eps)    // makes sure that limited machine precision doesn't produce particles when R is actually zero
    {
        z_step = sqrt(2.0/3.0)*mpd;
        y_step = sqrt(3.0)*mpd/2.0;
        x_step = mpd;
        
        z_count = -(int)floor(R/z_step);    // set the current z-layer
        z_start = -z_step*(double)z_count;    // starting z-level as positive value
        for( x[2]=-z_start; x[2]<(z_start+z_step*eps); x[2]+=z_step )
        {
            y_count = -(int)floor(R/y_step);    // set the current y-layer
            if ( z_count%2 )
            {
                y_start = -y_step*(double)y_count + mpd/sqrt(12.0);
                y_end = -y_step*(double)(y_count-1) - mpd/sqrt(12.0);
            }
            else
                y_start = y_end = -y_step*(double)y_count;
            
            for( x[1]=-y_start; x[1]<(y_end+y_step*eps); x[1]+=y_step )
            {
                if ( (z_count%2==0 && y_count%2==0) || (z_count%2!=0 && y_count%2!=0) )
                    x_start = mpd*floor(R/mpd);
                else
                    x_start = mpd/2.0 + mpd*floor(R/mpd);
                
                for( x[0]=-x_start; x[0]<(x_start+x_step*eps); x[0]+=x_step)
                {
                    radius2 = x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
                    if ( radius2 < R*R )    // particle inside the sphere if true
                    {
                        for(i=0; i<DIM; i++)
                        {
                            p->x[i] = x[i];
                            p->v[i] = 0.0;
                        }
                        if ( radius2 < R_core*R_core )    // particle in the core if true
                        {
                            (*N_core)++;
                            i=CORE;
                        }
                        else if ( radius2 < R_mantle*R_mantle )    // particle in the mantle if true
                        {
                            (*N_mantle)++;
                            i=MANTLE;
                        }
                        else    // particle in the shell if true
                            i=SHELL;
                        
                        p->mat_type = mat[i].mat_type;
                        n++;
                        p++;
                    }
                }
                y_count++;    // increment the current y-layer
            }
            z_count++;    // increment the current z-layer
        }
    }
    return(n);
}




int add_sphere_particles_SS(particle* p, int N_des, double R, double R_core, double R_mantle, char* source_dir, const char* outfile, int* N_core, int* N_mantle, double* mpd)
// Adds SPH particles for a sphere (at the origin) to the array of particles, starting at address 'p'.
// Particles are arranged in spherical shells computed externally via SEAGen. The particle arrangement is additionally written to 'outfile'.
// 'source_dir' specifies the spheres_ini directory containing 'run_SEAGen.py' and the directory 'SEAGen'.
// The sphere (radius 'R') consists in general of a core (R_core, N_core), a mantle (R_mantle, N_mantle) and a shell. 
// Returns the number of overall added particles.
// Adds positions, all velocity components = 0, and material types.
// The mean particle distance 'mpd' is computed based on the volume per particle (the sphere's volume divided by N) and the (hypothetical) particle
// distance in a HCP lattice, which should provide a reasonable representative (rather higher than lower) value. It is mpd = cbrt( sqrt(2)*V_particle ).
// Enough allocated memory has to be provided externally - 'p' is just incremented (locally) inside this function.
{
    int i;
    char handlerfile[PATHLENGTH];
    FILE *hfl, *f;
    wchar_t **py_wargv;     // Python is Unicode ...
    int py_argc = 14;
    char tmpstring[PATHLENGTH];
    int py_rc;
    int N_particles, N_read;
    double radius;
    double V_particle;
    
    
    // build path to Python handler script
    strcpy(handlerfile, source_dir);
    strcat(handlerfile, "/run_SEAGen.py");
    
    // open Python handler script
    if( (hfl = fopen(handlerfile, "r")) == NULL )
        ERRORVAR("FILE ERROR! Cannot open '%s' for reading!\n", handlerfile)
    
    // prepare cmd-line arguments for Python handler script
    py_wargv = (wchar_t**)malloc(py_argc*sizeof(wchar_t*));
    for (i=0; i<py_argc; i++)
        py_wargv[i] = (wchar_t*)malloc(sizeof(wchar_t)*PATHLENGTH);
    mbstowcs(py_wargv[0], "", PATHLENGTH);  // has to be set somehow ...
    mbstowcs(py_wargv[1], "--SEAGen_dir", PATHLENGTH);
    sprintf(tmpstring, "%s/SEAGen/", source_dir);
    mbstowcs(py_wargv[2], tmpstring, PATHLENGTH);
    mbstowcs(py_wargv[3], "--N_des", PATHLENGTH);
    sprintf(tmpstring, "%d", N_des);
    mbstowcs(py_wargv[4], tmpstring, PATHLENGTH);
    mbstowcs(py_wargv[5], "--R_total", PATHLENGTH);
    sprintf(tmpstring, "%.16le", R);
    mbstowcs(py_wargv[6], tmpstring, PATHLENGTH);
    mbstowcs(py_wargv[7], "--R_core", PATHLENGTH);
    sprintf(tmpstring, "%.16le", R_core);
    mbstowcs(py_wargv[8], tmpstring, PATHLENGTH);
    mbstowcs(py_wargv[9], "--R_mantle", PATHLENGTH);
    sprintf(tmpstring, "%.16le", R_mantle);
    mbstowcs(py_wargv[10], tmpstring, PATHLENGTH);
    mbstowcs(py_wargv[11], "--outfile", PATHLENGTH);
    mbstowcs(py_wargv[12], outfile, PATHLENGTH);
    mbstowcs(py_wargv[13], "-v", PATHLENGTH);
    
    // run the Python handler script
    fprintf(stdout, "\nCalling Python interface to invoke SEAGen, with cmd-line:   %s", handlerfile);
    for(i=0; i<py_argc; i++)
        fprintf(stdout, " %ls", py_wargv[i]);
    fprintf(stdout, "\n\n");
    fflush(stdout);
    fflush(stderr);
    PySys_SetArgv(py_argc, (wchar_t**)py_wargv);
    py_rc = PyRun_SimpleFile(hfl, handlerfile);     // the filename is required additionally for identification in error messages only
    fflush(stdout);
    fflush(stderr);
    if( py_rc != 0 )
        ERRORVAR2("ERROR! Python handler script '%s' returned exit status %d ...\n", handlerfile, py_rc)
    
    // clean up
    fclose(hfl);
    for (i=0; i<py_argc; i++)
        free(py_wargv[i]);
    free(py_wargv);
    
    // read file created by Python handler script containing particle data
    if( (f = fopen(outfile, "r")) == NULL )
        ERRORVAR("FILE ERROR! Cannot open '%s' for reading!\n", outfile)
    fscanf(f, "# %d\n", &N_particles);   // read actual number of particles from first line
    fscanf(f, "%*[^\n]\n%*[^\n]\n");   // ignore next 2 lines (comments)
    N_read = *N_core = *N_mantle = 0;
    
    while( fscanf(f, "%le %le %le %d %le%*[^\n]\n", &(p->x[0]), &(p->x[1]), &(p->x[2]), &(p->mat_type), &radius ) == 5 )
    {
        N_read++;
        for(i=0; i<DIM; i++)
            p->v[i] = 0.0;
        if( p->mat_type == MATTYPECORE )
            (*N_core)++;
        if( p->mat_type == MATTYPEMANTLE )
            (*N_mantle)++;
        
        // consistency check
        if( (p->mat_type==0 && radius>R_core) || (p->mat_type==1 && radius<R_core) || (p->mat_type==1 && radius>R_mantle) || (p->mat_type==2 && radius<R_mantle) || (p->mat_type==2 && radius>R) )
            ERRORVAR2("ERROR. Inconsistency in SEAGen particle setup. Found particle with radius = %e, but mat-type = %d ...\n", radius, p->mat_type)
        
        p++;
    }
    fclose(f);
    
    // consistency check
    if( N_read != N_particles )
        ERRORVAR3("ERROR when reading particles from '%s'. File header indicates %d particles, but found only %d particles ...\n", outfile, N_particles, N_read)
    
    // compute mpd
    V_particle = 4.0/3.0*M_PI*R*R*R / N_particles;
    *mpd = cbrt( sqrt(2.0)*V_particle );
    
    return(N_particles);
}




void matrix_times_vector(int n, double m[][n], double* v)
// Multiplies the nxn matrix m with the n-dimensional column vector v. The result is stored in v again.
// The parameter definition "int n, double m[][n]" probably works only with C99 compatible compilers.
{
    int i,j;
    double res[n];    // for intermediate storage of the result
    
    for (i=0; i<n; i++)
        res[i] = 0.0;
    
    // calculate matrix product with i,j being the row and column
    for (i=0; i<n; i++)
        for (j=0; j<n; j++)
            res[i] += m[i][j]*v[j];
    
    // assign stored values to v
    for (i=0; i<n; i++)
        v[i] = res[i];
}




void rotate_sphere(particle* p, int n, double z_angle, double y_angle, double x_angle)
// Performs three subsequent extrinsic (i.e all three about always fixed (initial) coord. axes) rotations, first about the z-axis (z_angle),
// then about the y -and x-axis. For a right-handed coord. system and a positive angle the respective rotation follows the "right-hand rule",
// with the thumb pointing toward the coord. axis. All angles passed to the function in rad!
// The function acts successively on 'n' particles, starting at address 'p'. The particles' positions are changed in place in 'p'.
// For saving unnecessary calculations merely the necessary two-dimensional parts of the rotation matrices are used.
{
    int i;
    double help[2];    // for intermediate storage during the y-rotation
    // build two-dimensional parts of rotation matrices:
    double z_rot[2][2] = { {cos(z_angle), -sin(z_angle)},
                           {sin(z_angle), cos(z_angle)} };
    double y_rot[2][2] = { {cos(y_angle), sin(y_angle)},
                           {-sin(y_angle), cos(y_angle)} };
    double x_rot[2][2] = { {cos(x_angle), -sin(x_angle)},
                           {sin(x_angle), cos(x_angle)} };
    
    for (i=0; i<n; i++)
    {
        matrix_times_vector(2, z_rot, &(p[i].x[0]));
        help[0] = p[i].x[0];
        help[1] = p[i].x[2];
        matrix_times_vector(2, y_rot, help);
        p[i].x[0] = help[0];
        p[i].x[2] = help[1];
        matrix_times_vector(2, x_rot, &(p[i].x[1]));
    }
}




void collision_geometry(double m_p, double m_t, double R_p, double R_t, double ini_dist, double vel_vesc, double alpha, double* impact_par, double* ini_vel, double *impact_vel_abs)
// Computes the initial position and relative velocity by tracing back the analytical orbit from a given v/v_esc (vel_vesc) and 
// impact angle (alpha - passed in deg!) at "touching ball" to a distance ini_dist. m_p, m_t, R_p, R_t are masses and radii of projectile and target. 
// Returns the impact parameter (impact_par) and relative velocity (ini_vel) at the bodies' initial position (i.e. at ini_dist).
// Additionally returns the impact speed at touching-ball distance impact_vel_abs.
{
    const double G = GRAV_CONST_SI;    // gravitational constant
    double mu, r, v;  // gravitational parameter = G*(m_p+m_t), and the bodies' distance and relative velocity
    double v_esc;   // escape velocity
    double a, e;    // orbital parameters, in case of a parabolic orbit p is saved to a
    int orbit_shape = -1;    // 1 for parabolic, 2 for hyperbolic, 3 for elliptic
    const double eps = 1.0e-6;
    
    mu = G*(m_p+m_t);
    r = R_p+R_t;
    alpha = alpha * M_PI / 180.0;   // convert alpha to rad
    v_esc = sqrt( 2.0*mu/r );
    v = vel_vesc*v_esc;
    *impact_vel_abs = v;
    fprintf(stdout, "--------------------------------\n");
    fprintf(stdout, "Compute initial position via tracing back the analytical orbit from a given v/v_esc and impact angle at \"touching ball\" distance:\n");
    fprintf(stdout, "  The mutual v_esc at \"touching ball\" distance (%e m) is %e m/s, the relative velocity (%e m/s) is %e times this value.\n", r, v_esc, v, v/v_esc);
    
    // calculate orbit shape for the respective conic section
    if( (vel_vesc > 1.0-eps) && (vel_vesc < 1.0+eps) )  // the orbit is parabolic
    {
        orbit_shape = 1;
        a = 2*r*sin(alpha)*sin(alpha);
        fprintf(stdout, "  This is treated as parabolic orbit with p = %e m (parabolic orbits are just a limiting case, make sure it is indeed (sufficiently close to) parabolic!\n", a);
    }
    else if( vel_vesc > 1.0 )    // the orbit is hyperbolic
    {
        orbit_shape = 2;
        a = 1.0 / ( v*v/mu - 2.0/r );
        e = sqrt( 1.0 + r/a/a*(2*a+r)*sin(alpha)*sin(alpha) );
        fprintf(stdout, "  This is a hyperbolic orbit with a = %e m and e = %e.\n", a, e);
    }
    else if( (vel_vesc < 1.0) && (vel_vesc > 0.0) )   // the orbit is elliptic
    {
        orbit_shape = 3;
        a = 1.0 / ( 2.0/r - v*v/mu );
        e = sqrt( 1.0 - r/a/a*(2*a-r)*sin(alpha)*sin(alpha) );
        fprintf(stdout, "  This is an elliptic orbit with a = %e m and e = %e.\n", a, e);
    }
    else
        ERRORTEXT("ERROR! Invalid result for v/v_esc during orbit calculation!\n")
    
    // calculate parameters at desired position on the orbit (i.e. at ini_dist)
    r = ini_dist;
    v_esc = sqrt(2.0*mu/r);
    
    if( orbit_shape == 1 )  // the orbit is parabolic
    {
        v = v_esc;
        alpha = asin( sqrt(a/2.0/r) );
    }
    else if( orbit_shape == 2 ) // the orbit is hyperbolic
    {
        v = sqrt( mu*( 2.0/r + 1.0/a ) );
        alpha = asin( sqrt( a*a*(e*e-1.0)/r/(2.0*a+r) ) );
    }
    else    // the orbit is elliptic
    {
        v = sqrt( mu*( 2.0/r - 1.0/a ) );
        alpha = asin( sqrt( a*a*(1.0-e*e)/r/(2.0*a-r) ) );
    }
    
    *ini_vel = v;
    *impact_par = r*sin(alpha);
    fprintf(stdout, "  At the desired initial distance (%e m) the mutual v_esc is %e m/s, the relative velocity (%e m/s) is %e times this value.\n", r, v_esc, v, v/v_esc);
    fprintf(stdout, "  (impact angle at this distance = %e deg)\n", alpha*180.0/M_PI);
    
}   // end function 'collision_geometry()'




void collision_parameters_from_cartesian(double m_p, double m_t, double R_p, double R_t, double* x_p, double* x_t, double* v_p, double* v_t, double* impact_angle, double* impact_vel_vesc, double* impact_vel_abs)
// Takes masses, radii, pos. and vel. vectors of projectile and target, and returns (at touching-ball distance R_p+R_t) the impact angle (in rad!), v/vesc, and the absolute value of the relative velocity.
// NOTE: If it is not a physical collision but only a close encounter then it returns 'impact_angle' = -1.0, and for 'impact_vel_vesc' and 'impact_vel_abs' the respective values at the relative orbit's pericenter!
{
    int i;
    double mu, r_ini, v_ini, alpha_ini, vesc_ini;
    int orbit_shape = -1;    // 1 for parabolic, 2 for hyperbolic, 3 for elliptic
    double vec1[DIM], vec2[DIM];
    const double eps = 1.0e-6;
    double a, e, p_parabola;    // orbital parameters
    double pericenter_dist;
    double vesc_col, r_col, v_col, alpha_col; // values at the moment of collision (at touching-ball distance)
    
    
    // compute initial distance, relative velocity, and alpha (in rad!)
    mu = GRAV_CONST_SI*(m_p+m_t);
    for(i=0; i<DIM; i++)
        vec1[i] = x_t[i] - x_p[i];
    for(i=0; i<DIM; i++)
        vec2[i] = v_p[i] - v_t[i];
    r_ini=0.0;
    v_ini=0.0;
    for(i=0; i<DIM; i++)
        r_ini += vec1[i] * vec1[i];
    r_ini = sqrt(r_ini);
    for(i=0; i<DIM; i++)
        v_ini += vec2[i] * vec2[i];
    v_ini = sqrt(v_ini);
    alpha_ini = 0.0;
    for(i=0; i<DIM; i++)
        alpha_ini += vec1[i] * vec2[i];
    alpha_ini = acos( alpha_ini/(r_ini*v_ini) );    //'alpha_ini' in rad!
    
    
    // calculate v/vesc and orbit shape for the respective conic section
    vesc_ini = sqrt( 2.0*mu/r_ini );
    if( (v_ini/vesc_ini > 1.0-eps) && (v_ini/vesc_ini < 1.0+eps) )  //the orbit is parabolic, with pericenter distance p/2
    {
        orbit_shape = 1;
        p_parabola = 2*r_ini*sin(alpha_ini)*sin(alpha_ini);
        fprintf(stdout, "  The relative orbit is treated as parabolic with p = %e ...\n", p_parabola);
        pericenter_dist = p_parabola/2.0;
    }
    else if( v_ini/vesc_ini > 1.0 )    //the orbit is hyperbolic, with pericenter distance a(e-1)
    {
        orbit_shape = 2;
        a = 1.0 / ( v_ini*v_ini/mu - 2.0/r_ini );
        e = sqrt( 1.0 + r_ini/a/a*(2*a+r_ini)*sin(alpha_ini)*sin(alpha_ini) );
        fprintf(stdout, "  The relative orbit is hyperbolic with a = %e and e = %e.\n", a, e);
        pericenter_dist = a*(e-1.0);
    }
    else if( (v_ini/vesc_ini < 1.0) && (v_ini/vesc_ini > 0.0) )   //the orbit is elliptic, with pericenter distance a(1-e)
    {
        orbit_shape = 3;
        a = 1.0 / ( 2.0/r_ini - v_ini*v_ini/mu );
        e = sqrt( 1.0 - r_ini/a/a*(2*a-r_ini)*sin(alpha_ini)*sin(alpha_ini) );
        fprintf(stdout, "  The relative orbit is elliptic with a = %e and e = %e.\n", a, e);
        pericenter_dist = a*(1.0-e);
    }
    else
        ERRORVAR("ERROR! Invalid result for velocity over escape velocity = %e in function 'collision_parameters_from_cartesian()' ...\n", v_ini/vesc_ini)
    
    
    // compute collision parameters at touching-ball distance R_p+R_t if it's a physical collision, and at the pericenter of the relative orbit if not
    if( pericenter_dist < R_p+R_t ) // it's a pyhsical collision
    {
        r_col = R_p+R_t;
        vesc_col = sqrt(2.0*mu/r_col);
        
        if( orbit_shape == 1 ) // parabolic
        {
            v_col = vesc_col;
            alpha_col = asin( sqrt(p_parabola/2.0/r_col) );
        }
        else if( orbit_shape == 2 ) // hyperbolic
        {
            v_col = sqrt( mu*( 2.0/r_col + 1.0/a ) );
            alpha_col = asin( sqrt( a*a*(e*e-1.0)/r_col/(2.0*a+r_col) ) );
        }
        else    // elliptic
        {
            v_col = sqrt( mu*( 2.0/r_col - 1.0/a ) );
            alpha_col = asin( sqrt( a*a*(1.0-e*e)/r_col/(2.0*a-r_col) ) );
        }
        
        *impact_angle = alpha_col;  // in rad!
        *impact_vel_abs = v_col;
        *impact_vel_vesc = v_col/vesc_col;
    }
    else    // it's no physical collision
    {
        r_col = pericenter_dist;
        vesc_col = sqrt(2.0*mu/r_col);
        
        if( orbit_shape == 1 ) // parabolic
            v_col = vesc_col;
        else if( orbit_shape == 2 ) // hyperbolic
            v_col = sqrt( mu/a*(e+1.0)/(e-1.0) );
        else    // elliptic
            v_col = sqrt( mu/a*(1.0+e)/(1.0-e) );
        
        *impact_angle = -1.0;
        *impact_vel_abs = v_col;
        *impact_vel_vesc = v_col/vesc_col;
    }
}   // end function 'collision_parameters_from_cartesian()'

