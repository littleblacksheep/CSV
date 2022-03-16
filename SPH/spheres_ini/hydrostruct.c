// spheres_ini functions related to hydrostatic structure calculation.

// Christoph Burger 02/Mar/2019


#include <stdio.h>
#include <math.h>

#include "spheres_ini.h"
#include "hydrostruct.h"



double Tillotson_pressure(double e, double rho, material* mat)
// Calculates the pressure from the internal energy e and the density rho, following the Tillotson eos, for material mat.
{
    double rho_0, e_0, e_iv, e_cv, a, b, A, B, alpha, beta;
    double eta, mu, p, p_c, p_e;    // p_c/p_e are pressures calculated via the commpressed/expanded form of the Tillotson eos (for a given e and rho) - used for weighted average for intermediate states
    
    rho_0 = mat->till.rho_0;
    e_0 = mat->till.e_0;
    e_iv = mat->till.e_iv;
    e_cv = mat->till.e_cv;
    a = mat->till.a;
    b = mat->till.b;
    A = mat->till.A;
    B = mat->till.B;
    alpha = mat->till.alpha;
    beta = mat->till.beta;
    
    eta = rho/rho_0;
    mu = eta-1.0;
    
    if ( eta<(mat->till.rho_limit) && e<e_cv )    // low-density pressure cutoff
        p = 0.0;
    else if ( e<=e_iv || eta>=1.0 )    // compressed/solid region
        p = (a + b/(e/(e_0*eta*eta) + 1.0))*rho*e + A*mu + B*mu*mu;
    else if ( e>=e_cv && eta>=0.0 )    // expanded/vapor region
        p = a*rho*e + (b*rho*e/(e/(e_0*eta*eta)+1.0) + A*mu*exp(-beta*(rho_0/rho-1.0))) * exp(-alpha*(pow(rho_0/rho-1.0,2)));
    else if ( e>e_iv && e<e_cv )    // intermediate states: weighted average of pressures calculated by expanded and compressed versions of Tillotson (both evaluated at e)
    {
        p_c = (a + b/(e/(e_0*eta*eta) + 1.0))*rho*e + A*mu + B*mu*mu;
        p_e = a*rho*e + (b*rho*e/(e/(e_0*eta*eta)+1.0) + A*mu*exp(-beta*(rho_0/rho-1.0))) * exp(-alpha*(pow(rho_0/rho-1.0,2)));
        p = ( p_c*(e_cv-e) + p_e*(e-e_iv) ) / (e_cv-e_iv);
    }
    else
    {
        fprintf(stderr, "ERROR! Deep trouble in pressure for rho = %.16le, e = %.16le. Pressure set to zero!\n", rho, e);
        p = 0.0;
    }
    return(p);
}




double Tillotson_density(double e, double p, material* mat)
// Calculates the density from the internal energy e (only values >=0) and the pressure p (only values >=0), following the Tillotson eos, for material mat.
// The result is found by iterating the Tillotson eos (~bisection method) until the relative error becomes sufficiently small.
{
    const int i_limit = 100;        // maximum number of iterations before the algorithm is stopped
    int i=0;
    double lb, ub;    // lower/upper density boundaries for (density-)iteration
    double rho;
    const double eps = 1.0e-9;    // (more or less) carefully chosen value adapted to the purposes of this function
    double rho_0 = mat->till.rho_0;
#ifdef HYDROSTRUCT_DEBUG
    int j;
    double debug_lb[i_limit];
    double debug_ub[i_limit];
    double debug_rho[i_limit];
    double debug_rel_error[i_limit];
    double debug_p[i_limit];
#endif
    
    
    // find lower/upper boundary of subsequent (density-)iteration range (implemented that way due to non-monotone behaviour of Tillotson for (very) high densities??)
    // lb = (mat->till.rho_limit) * (mat->till.rho_0);
    lb = 0.0;
    ub = rho_0;
    while ( Tillotson_pressure(e,ub,mat) <= p )
    {
        lb = ub;
        ub *= 1.1;
    }
    
    // now start (density-)iteration
    rho = (lb+ub)/2.0;
    do
    {
#ifdef HYDROSTRUCT_DEBUG
        debug_lb[i] = lb;
        debug_ub[i] = ub;
        debug_rho[i] = rho;
        debug_rel_error[i] = (ub-lb)/rho;
        debug_p[i] = Tillotson_pressure(e,rho,mat);
#endif
        if ( Tillotson_pressure(e,rho,mat) >= p )
            ub = rho;
        else
            lb = rho;
        rho = (lb+ub)/2.0;
        i++;
    }
    while( ((ub-lb)/rho >= eps) && (i < i_limit) );
//    while( ((ub-lb)/rho_0 >= eps) && (i < i_limit) );
    
    if ( i >= i_limit )
    {
        fprintf(stderr, "ERROR during Tillotson density calculation for material '%s' with e = %.16le and p = %.16le! Iteration limit exceeded, probably no convergence! Maybe try slightly different initial conditions or to increase eps a bit ...\n", mat->mat_name, e, p);
#ifdef HYDROSTRUCT_DEBUG
        for(j=0; j<i_limit; j++)
            fprintf(stderr, "    iteration no. %d:  lb = %.16le    ub = %.16le    rho = %.16le    (ub-lb)/rho = %.16le    p = %.16le\n", j, debug_lb[j], debug_ub[j], debug_rho[j], debug_rel_error[j], debug_p[j] );
#endif
        exit(1);
    }
    
    return(rho);
}




double ANEOS_pressure(double e, double rho, material* mat)
// Calculates the pressure from the internal energy e and the density rho, following the ANEOS eos, for material mat.
{
    double p;
    int i_rho, i_e;
    
    // find array-indices just below the actual values of rho and e
    i_rho = array_index(rho, mat->aneos.rho, mat->aneos.n_rho);
    i_e = array_index(e, mat->aneos.e, mat->aneos.n_e);
    
    // interpolate (bi)linearly to obtain the pressure
    p = bilinear_interpolation(rho, e, mat->aneos.p, mat->aneos.rho, mat->aneos.e, i_rho, i_e, mat->aneos.n_rho, mat->aneos.n_e);
    
    return(p);
}




double ANEOS_density(double e, double p, material* mat)
// Calculates the density from the internal energy 'e' and the pressure 'p', with ANEOS, for material 'mat'.
// The result is found by iterating the ANEOS eos (~bisection method) until the relative error becomes sufficiently small.
{
    const int i_limit = 100;        // maximum number of iterations before the algorithm is stopped
    int i=0;
    double lb, ub;    // lower/upper density boundaries for (density-)iteration
    double rho;
    const double eps = 1.0e-9;    // (more or less) carefully chosen value adapted to the purposes of this function
#ifdef HYDROSTRUCT_DEBUG
    int j;
    double debug_lb[i_limit];
    double debug_ub[i_limit];
    double debug_rho[i_limit];
    double debug_rel_error[i_limit];
    double debug_p[i_limit];
#endif
    
    
    // find lower/upper boundary of subsequent (density-)iteration range
    lb = 0.0;
    ub = mat->aneos.rho_0;
    while ( ANEOS_pressure(e,ub,mat) <= p )
    {
        lb = ub;
        ub *= 1.1;
    }
    
    // now start (density-)iteration
    rho = (lb+ub)/2.0;
    do
    {
#ifdef HYDROSTRUCT_DEBUG
        debug_lb[i] = lb;
        debug_ub[i] = ub;
        debug_rho[i] = rho;
        debug_rel_error[i] = (ub-lb)/rho;
        debug_p[i] = ANEOS_pressure(e,rho,mat);
#endif
        if ( ANEOS_pressure(e,rho,mat) >= p )
            ub = rho;
        else
            lb = rho;
        rho = (lb+ub)/2.0;
        i++;
    }
    while( ((ub-lb)/rho >= eps) && (i < i_limit) );
    
    if ( i >= i_limit )
    {
        fprintf(stderr, "ERROR during ANEOS density calculation for material '%s' with e = %.16le and p = %.16le! Iteration limit exceeded, probably no convergence! Maybe try slightly different initial conditions or to increase eps a bit ...\n", mat->mat_name, e, p);
#ifdef HYDROSTRUCT_DEBUG
        for(j=0; j<i_limit; j++)
            fprintf(stderr, "    iteration no. %d:  lb = %.16le    ub = %.16le    rho = %.16le    (ub-lb)/rho = %.16le    p = %.16le\n", j, debug_lb[j], debug_ub[j], debug_rho[j], debug_rel_error[j], debug_p[j] );
#endif
        exit(1);
    }
    
    return(rho);
}




double ideal_gas_pressure(double e, double rho, material* mat)
// Calculates the pressure from the internal energy e and the density rho, following the ideal_gas eos, for material mat.
{
    return( (mat->ideal_gas.gamma-1.0)*rho*e );
}




double ideal_gas_density(double e, double p, material* mat)
// Calculates the density from the internal energy e and the pressure p, following the ideal_gas eos, for material mat.
{
    return( p/(mat->ideal_gas.gamma-1.0)/e );
}




void rhs_lagrange(double m, double r, double p, double* f, material* mat, material_fp* mat_fp, double rho_approx)
// Calculates the rhs of the system of ODEs for the internal structure calculation in its Lagrangian form (i.e ODEs for r(m) and p(m)).
// 'm', 'r' and 'p' are passed, the result is stored in the double array at address 'f' (2 elements) and the material is specified in 'mat'.
// 'rho_approx' is required as start value for the iteration when solving the self consistency problem for the density.
{
    const double G = GRAV_CONST_SI;   // gravitational constant
    
    f[0] = 1.0/(4*M_PI*r*r*(*(mat_fp->eos_rho_self_consistent))(p, rho_approx, mat));
    f[1] = -G*m/(4*M_PI*pow(r,4));
}




void RK4_step_lagrange(double h, int_struct_point* s, material* mat, material_fp* mat_fp)
// Calculates one RungeKutta4 step following the hydrostatic structure equations (in their Lagrangian form, i.e. with m as the independent
// variable). The respective function pointer are passed in 'mat_fp'.
// The entire (i.e. r, m, rho and p) solution for the internal structure is advanced from the point at address 's' to the point at s-1.
// The material is passed in 'mat', 'h' is the stepsize (in the integrated mass m).
{
    double k1[2], k2[2], k3[2], k4[2];
    
    rhs_lagrange(s->m, s->r, s->p, k1, mat, mat_fp, s->rho);
    rhs_lagrange((s->m)+0.5*h, (s->r)+0.5*h*k1[0], (s->p)+0.5*h*k1[1], k2, mat, mat_fp, s->rho);
    rhs_lagrange((s->m)+0.5*h, (s->r)+0.5*h*k2[0], (s->p)+0.5*h*k2[1], k3, mat, mat_fp, s->rho);
    rhs_lagrange((s->m)+h, (s->r)+h*k3[0], (s->p)+h*k3[1], k4, mat, mat_fp, s->rho);
    (s-1)->r = (s->r)+1.0/6.0*h*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0]);
    (s-1)->p = (s->p)+1.0/6.0*h*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1]);
    (s-1)->m = (s->m)+h;
    (s-1)->rho = (*(mat_fp->eos_rho_self_consistent))((s-1)->p, s->rho, mat);
}




void calc_internal_structure(double M, double M_c, double M_m, double R_lb, double R_ub, material* mat, material_fp* mat_fp, int_struct_point* int_struct_c, int_struct_point* int_struct_m, int_struct_point* int_struct_s, int n_steps)
// Calculates the internal structure of a sphere, consisting in general of a core, a mantle and a shell of different materials, by solving 
// the hydrostatic equations (in their Lagrangian form, i.e. with m as the independent variable). The respective function pointer are passed in 'mat_fp'.
// 'M' is the overall mass, M_c/M_m are masses of core/mantle, R_lb/R_ub are initial values for the lower/upper boundary of the overall radius (which is varied (~ bisection method) until the correct value is found).
// The results are written to 'int_struct_c' (core), 'int_struct_m' (mantle) and 'int_struct_s' (shell). If any component is not present the corresponding array will be left untouched!
{
    int i,j;
    double R;    // overall Radius (varied (~ bisection method) until correct value is determined)
    double stepsize_c, stepsize_m, stepsize_s;    // stepsizes for core/mantle/shell
    const double eps6 = 1.0e-6;
    
#ifdef HYDROSTRUCT_DEBUG
    fprintf(stdout, "\nBEGIN HYDROSTRUCT CALCULATION FOR BODY WITH M = %g, M_c = %g, M_m = %g, M_s = %g ...\n", M, M_c, M_m, M-M_c-M_m );
    fprintf(stdout, "    R_lb = %g\n    R_ub = %g\n", R_lb, R_ub);
#endif
    
    if ( (M-M_c-M_m)/M > eps6 )    // if there is a shell ...
    {
        stepsize_s = -(M-M_c-M_m)/n_steps;
        int_struct_s[n_steps].m = M;
        if( mat[SHELL].eos == 'I' )
            int_struct_s[n_steps].p = mat[SHELL].ideal_gas.p_0;
        else if( mat[SHELL].eos == 'A' )
            int_struct_s[n_steps].p = ANEOS_pressure(mat[SHELL].aneos.e_norm, mat[SHELL].aneos.rho_0, &mat[SHELL]);     // use pressure at norm condition
        else
            int_struct_s[n_steps].p = 0.0;
        int_struct_s[n_steps].rho = mat[SHELL].rho_0;
    }
    else if ( M_m/M > eps6 )    // if there is no shell, but a mantle ...
    {
        int_struct_m[n_steps].m = M;
        if( mat[MANTLE].eos == 'I' )
            int_struct_m[n_steps].p = mat[MANTLE].ideal_gas.p_0;
        else if( mat[MANTLE].eos == 'A' )
            int_struct_m[n_steps].p = ANEOS_pressure(mat[MANTLE].aneos.e_norm, mat[MANTLE].aneos.rho_0, &mat[MANTLE]);     // use pressure at norm condition
        else
            int_struct_m[n_steps].p = 0.0;
        int_struct_m[n_steps].rho = mat[MANTLE].rho_0;
    }
    else if ( M_c/M > eps6 )    // if there is no shell or mantle, but a core ...
    {
        int_struct_c[n_steps].m = M;
        if( mat[CORE].eos == 'I' )
            int_struct_c[n_steps].p = mat[CORE].ideal_gas.p_0;
        else if( mat[CORE].eos == 'A' )
            int_struct_c[n_steps].p = ANEOS_pressure(mat[CORE].aneos.e_norm, mat[CORE].aneos.rho_0, &mat[CORE]);     // use pressure at norm condition
        else
            int_struct_c[n_steps].p = 0.0;
        int_struct_c[n_steps].rho = mat[CORE].rho_0;
    }
    else
        ERRORTEXT("ERROR! Some strange mass mismatch during hydrostatic structure calculation ...\n")
    
    if ( M_m/M > eps6 )     // if there is a mantle
        stepsize_m = -M_m/n_steps;
    if ( M_c/M > eps6 )     // if there is a core
        stepsize_c = -M_c/n_steps;
    
    j = 0;
    while(TRUE)    // repeat until the overall radius (and structure) is found ...
    {
        R = (R_lb+R_ub)/2.0;
        
#ifdef HYDROSTRUCT_DEBUG
        fprintf(stdout, "    Start iteration no. %d    from R = %.16le\n", j, R);
#endif
        
        if ( (M-M_c-M_m)/M > eps6 )    // if there is a shell ...
        {
            int_struct_s[n_steps].r = R;
            // calculate structure for shell:
            for(i=n_steps; i>=1; i--)
                RK4_step_lagrange(stepsize_s, &int_struct_s[i], &mat[SHELL], &mat_fp[SHELL]);
            if (M_m/M > eps6)    // if there is a mantle (in addition to the shell) ...
            {
                // set all values at the mantle/shell boundary (for subsequent mantle calculation):
                int_struct_m[n_steps].m = int_struct_s[0].m;
                int_struct_m[n_steps].r = int_struct_s[0].r;
                int_struct_m[n_steps].p = int_struct_s[0].p;
                int_struct_m[n_steps].rho = (*(mat_fp[MANTLE].eos_rho_self_consistent))(int_struct_m[n_steps].p, (*(mat_fp[MANTLE].eos_density))(0.0, int_struct_m[n_steps].p, &mat[MANTLE]), &mat[MANTLE]);    //using eos_density with e=0.0 as start value for self consistency iteration
            }
            else if (M_c/M > eps6)    // if there is no mantle, but a core in addition to the shell ...
            {
                // set all values at the core/shell boundary (for subsequent core calculation):
                int_struct_c[n_steps].m = int_struct_s[0].m;
                int_struct_c[n_steps].r = int_struct_s[0].r;
                int_struct_c[n_steps].p = int_struct_s[0].p;
                int_struct_c[n_steps].rho = (*(mat_fp[CORE].eos_rho_self_consistent))(int_struct_c[n_steps].p, (*(mat_fp[CORE].eos_density))(0.0, int_struct_c[n_steps].p, &mat[CORE]), &mat[CORE]);    //using eos_density with e=0.0 as start value for self consistency iteration
            }
            else    // if there is no mantle or core below the shell ...
            {
#ifdef HYDROSTRUCT_DEBUG
                fprintf(stdout, "        Found 'int_struct_s[0].r' = %.16le\n", int_struct_s[0].r);
#endif
                // check whether supposed R was just right (then break) or too large/small (and improve the considered R-range [R_lb,R_ub] in this case):
                if (int_struct_s[0].r > R*eps6)
                    R_ub = R;
                else if (int_struct_s[0].r < 0.0)
                    R_lb = R;
                else
                    break;
#ifdef HYDROSTRUCT_DEBUG
                fprintf(stdout, "        next R_lb = %.16le\n        next R_ub = %.16le\n", R_lb, R_ub);
#endif
            }
        }
        else if ( M_m/M > eps6 )    // if there is no shell, but a mantle ...
            int_struct_m[n_steps].r = R;
        else    // if there is no shell or mantle, but a core ...
            int_struct_c[n_steps].r = R;
    
        if (M_m/M > eps6)    // if there is a mantle ...
        {
            // calculate structure for mantle:
            for(i=n_steps; i>=1; i--)
                RK4_step_lagrange(stepsize_m, &int_struct_m[i], &mat[MANTLE], &mat_fp[MANTLE]);
            if (M_c/M > eps6)    // if there is a core below the mantle ...
            {
                // set all values at the core/mantle boundary (for subsequent core calculation):
                int_struct_c[n_steps].m = int_struct_m[0].m;
                int_struct_c[n_steps].r = int_struct_m[0].r;
                int_struct_c[n_steps].p = int_struct_m[0].p;
                int_struct_c[n_steps].rho = (*(mat_fp[CORE].eos_rho_self_consistent))(int_struct_c[n_steps].p, (*(mat_fp[CORE].eos_density))(0.0, int_struct_c[n_steps].p, &mat[CORE]), &mat[CORE]);    //using eos_density with e=0.0 as start value for self consistency iteration
            }
            else    // if there is no core below the mantle ...
            {
#ifdef HYDROSTRUCT_DEBUG
                fprintf(stdout, "        Found 'int_struct_m[0].r' = %.16le\n", int_struct_m[0].r);
#endif
                // check whether supposed R was just right (then break) or too large/small (and improve the considered R-range [R_lb,R_ub] in this case):
                if (int_struct_m[0].r > R*eps6)
                    R_ub = R;
                else if (int_struct_m[0].r < 0.0)
                    R_lb = R;
                else
                    break;
#ifdef HYDROSTRUCT_DEBUG
                fprintf(stdout, "        next R_lb = %.16le\n        next R_ub = %.16le\n", R_lb, R_ub);
#endif
            }
        }
        
        if (M_c/M > eps6)    // if there is a core ...
        {
            // calculate structure for core:
            for(i=n_steps; i>=1; i--)
                RK4_step_lagrange(stepsize_c, &int_struct_c[i], &mat[CORE], &mat_fp[CORE]);
#ifdef HYDROSTRUCT_DEBUG
            fprintf(stdout, "        Found 'int_struct_c[0].r' = %.16le\n", int_struct_c[0].r);
#endif
            // check whether supposed R was just right (then break) or too large/small (and improve the considered R-range [R_lb,R_ub] in this case):
            if (int_struct_c[0].r > R*eps6)
                R_ub = R;
            else if (int_struct_c[0].r < 0.0)
                R_lb = R;
            else
                break;
#ifdef HYDROSTRUCT_DEBUG
            fprintf(stdout, "        next R_lb = %.16le\n        next R_ub = %.16le\n", R_lb, R_ub);
#endif
        }
        
#ifdef MORE_HYDROSTRUCT_DEBUG
        fprintf(stdout, "\n");
        if( (M-M_c-M_m)/M > eps6 )    // if there is a shell ...
        {
            fprintf(stdout, "found shell structure (iteration no. %d):\n", j);
            for(i=n_steps; i>=0; i--)
                fprintf(stdout, "    r = %g    m = %g    rho = %g    p = %g    e = %g\n", int_struct_s[i].r, int_struct_s[i].m, int_struct_s[i].rho, int_struct_s[i].p,
                        (*(mat_fp[SHELL].eos_e_compression))(int_struct_s[i].rho, &mat[SHELL]) );
        }
        if( M_m/M > eps6 )    // if there is a mantle ...
        {
            fprintf(stdout, "found mantle structure (iteration no. %d):\n", j);
            for(i=n_steps; i>=0; i--)
                fprintf(stdout, "    r = %g    m = %g    rho = %g    p = %g    e = %g\n", int_struct_m[i].r, int_struct_m[i].m, int_struct_m[i].rho, int_struct_m[i].p,
                        (*(mat_fp[MANTLE].eos_e_compression))(int_struct_m[i].rho, &mat[MANTLE]) );
        }
        if( M_c/M > eps6 )    // if there is a core ...
        {
            fprintf(stdout, "found core structure (iteration no. %d):\n", j);
            for(i=n_steps; i>=0; i--)
                fprintf(stdout, "    r = %g    m = %g    rho = %g    p = %g    e = %g\n", int_struct_c[i].r, int_struct_c[i].m, int_struct_c[i].rho, int_struct_c[i].p,
                        (*(mat_fp[CORE].eos_e_compression))(int_struct_c[i].rho, &mat[CORE]) );
        }
        fprintf(stdout, "\n\n");
#endif
        
        if ( (j++) >= MAXITERATIONS )
        {
            fprintf(stderr, "ERROR when calculating hydrostatic structure for relaxation! Number of iterations exceeded MAXITERATIONS ...\n");
            fprintf(stderr, "Are you using ANEOS? If so, make sure that the respective lookup-tables cover also the probably very extreme conditions during the structure iteration!\n");
            fprintf(stderr, "To circumvent this issue you can also try slightly different initial conditions or a slight increase of the used eps ... ");
            fprintf(stderr, "Are you dealing with rather extreme conditions, i.e. very massive bodies leading to very high densities, etc.? ");
            fprintf(stderr, "Then try to constrain the initial radii-range via the cmd-line options as narrow as possible!\n");
            exit(1);
        }
    }
}   // end function 'calc_internal_structure()'




void set_hydrostruct_density(particle* p, int i, double r2, int_struct_point* int_struct)
// This function sets (interpolates) the density (following the correct hydrostatic structure) for one particle 'p[i]' for which it was
// (externally) predetermined where (projectile/target/core/mantle/shell) it is - the structure of this part is passed in 'int_struct'.
// 'p' is the absolute (first) address of the particle vector, 'i' is the index/element of this vector for which a density should be found.
// 'r2' is the squared distance to the origin.
{
    int j;
    
    for( j=0; j<=NSTEPS; j++)
        if ( r2 < pow(int_struct[j].r,2) )
            break;
    if (j==0)    // particle inside the innermost radius in 'int_struct' (this innermost radius is always >= 0)
        p[i].rho = int_struct[0].rho;
    else if (j<=NSTEPS)    //particle somewhere between two radii given in int_struct
        p[i].rho = int_struct[j-1].rho + (int_struct[j].rho-int_struct[j-1].rho)/(int_struct[j].r-int_struct[j-1].r) * (sqrt(r2)-int_struct[j-1].r);
    else    // particle outside the radius range of int_struct
        ERRORTEXT("ERROR when assigning densities (calculated via hydrostatic structure) to particles! Strange radius mismatch ...\n")
}




double e_compression_Tillotson(double rho_des, material* mat)
// Integrates the energy equation de/dt=-p/rho*div(v) with RK4 to obtain the internal energy e for a given material density rho_des (adiabatic compression).
// -div(v)=const=1 is assumed and the continuity equation (for obtaining rho(t) analytically) and the eos are used.
{
    const int n_steps = 100;    // number of integration steps
    int i;
    double t_end, h;        // end time for integration (start time is t=0); stepsize for integration
    double e = 0.0;     // initial value for int. energy
    double t = 0.0;     // initial value for time
    double k1, k2, k3, k4;        // Runge-Kutta auxiliary variables
    double rho_0 = mat->rho_0;
    
    t_end = log(rho_des/rho_0);
    h = t_end/n_steps;
    
    // do Runge-Kutta integration over time
    for(i=0; i<n_steps; i++)
    {
        k1 = Tillotson_pressure(e, rho_0*exp(t), mat) / rho_0 * exp(-t);
        k2 = Tillotson_pressure(e+0.5*h*k1, rho_0*exp(t+0.5*h), mat) / rho_0 * exp(-(t+0.5*h));
        k3 = Tillotson_pressure(e+0.5*h*k2, rho_0*exp(t+0.5*h), mat) / rho_0 * exp(-(t+0.5*h));
        k4 = Tillotson_pressure(e+h*k3, rho_0*exp(t+h), mat) / rho_0 * exp(-(t+h));
        e = e + 1.0/6.0*h*(k1 + 2.0*k2 + 2.0*k3 + k4);
        t += h;
    }
    return(e);
}




double e_compression_ANEOS(double rho_des, material* mat)
// Integrates the energy equation de/dt=-p/rho*div(v) with RK4 to obtain the internal energy e for a given material density rho_des (adiabatic compression).
// -div(v)=const=1 is assumed and the continuity equation (for obtaining rho(t) analytically) and the eos are used.
{
    const int n_steps = 100;    // number of integration steps
    int i;
    double t_end, h;        // end time for integration (start time is t=0); stepsize for integration
    double e = mat->aneos.e_norm;   // initial value for int. energy
    double t = 0.0;    // initial value for time
    double k1, k2, k3, k4;        // Runge-Kutta auxiliary variables
    double rho_0 = mat->rho_0;
    
    t_end = log(rho_des/rho_0);
    h = t_end/n_steps;
    
    // do Runge-Kutta integration over time
    for(i=0; i<n_steps; i++)
    {
        k1 = ANEOS_pressure(e, rho_0*exp(t), mat) / rho_0 * exp(-t);
        k2 = ANEOS_pressure(e+0.5*h*k1, rho_0*exp(t+0.5*h), mat) / rho_0 * exp(-(t+0.5*h));
        k3 = ANEOS_pressure(e+0.5*h*k2, rho_0*exp(t+0.5*h), mat) / rho_0 * exp(-(t+0.5*h));
        k4 = ANEOS_pressure(e+h*k3, rho_0*exp(t+h), mat) / rho_0 * exp(-(t+h));
        e = e + 1.0/6.0*h*(k1 + 2.0*k2 + 2.0*k3 + k4);
        t += h;
    }
    return(e);
}




double e_compression_ideal_gas(double rho_des, material* mat)
// Returns the internal energy of an ideal gas under adiabatic compression (i.e. of a polytropic gas) as a function of rho, as e=1/(gamma-1)*K*rho^(gamma-1).
{
    return(1.0/(mat->ideal_gas.gamma-1.0)*(mat->ideal_gas.polytropic_K)*pow(rho_des,mat->ideal_gas.gamma-1.0));
}




double rho_self_consistent_Tillotson(double p, double rho_start, material* mat)
// Solves the self consistency problem rho=rho(p,e(rho)), where e(rho) represents the internal energy for a given density (adiabatic compression, see the respective function).
// The problem is solved by simply iterating (kind of fixed-point iteration): rho_(n+1)=Tillotson_density(p,e(rho_pre)), where rho_pre=(rho_n+rho_(n-1))/2.
// The pressure p (only values >= 0, since Tillotson_density() requires that), a suitable start value for the density and the respective material are passed.
// NOTE: If the iteration doesn't converge then 'rho_start' is simply returned!
{
    int i=0;
    const double eps = 1.0e-7;        // carefully chosen value, especially w.r.t. the eps in Tillotson_density! If this eps is not smaller, Tillotson_density can't calculate accurate enough for the break condition here to succeed!
    const int i_limit = 200;        // maximum number of iterations before algorithm is stopped
    double rho, rho_pre = rho_start;    // current and previous density value during iteration
    double rel_error;
#ifdef MORE_HYDROSTRUCT_DEBUG
    int j;
    double debug_rho[i_limit];
    double debug_rho_pre[i_limit];
    double debug_rel_error[i_limit];
#endif
    
    
    do
    {
        rho = Tillotson_density(e_compression_Tillotson(rho_pre, mat), p, mat);
        rel_error = fabs(rho-rho_pre) / rho_pre;        // relative error - relative to the previous value of the density(-iteration)
#ifdef MORE_HYDROSTRUCT_DEBUG
        debug_rho[i] = rho;
        debug_rho_pre[i] = rho_pre;
        debug_rel_error[i] = rel_error;
#endif
        rho_pre = (rho_pre+rho) / 2.0;    // using the mean value here gives probably a better approximation towards the final result, and prohibits convergence towards 2 fixed points
        i++;
    }
    while( (rel_error > eps) && (i < i_limit) );
    
    
    if ( i >= i_limit )
    {
        fprintf(stderr, "WARNING: Probably no convergence during self-consistent density calculation in 'rho_self_consistent_Tillotson()', ");
        fprintf(stderr, "for p = %e, rho_start = %e, material = %s, e_compression(rho_start) = %e. Returned rho_start = %e ...\n", p, rho_start, mat->mat_name, e_compression_Tillotson(rho_start,mat), rho_start);
#ifdef MORE_HYDROSTRUCT_DEBUG
        for(j=0; j<i_limit; j++)
            fprintf(stderr, "    iteration no. %d:  rho_pre = %.16le    rho = %.16le    rel_error = %.16le\n", j, debug_rho_pre[j], debug_rho[j], debug_rel_error[j]);
#endif
        fflush(stderr);
        return(rho_start);    // return 'rho_start' if the iteration limit was exceeded
    }
    else
    {
        return(rho);
    }
}




double rho_self_consistent_ANEOS(double p, double rho_start, material* mat)
// Solves the self consistency problem rho=rho(p,e(rho)), where e(rho) represents the internal energy for a given density (adiabatic compression, see the respective
// function). The problem is solved by simply iterating (kind of fixed-point iteration): rho_(n+1)=ANEOS_density(p,e(rho_pre)), where rho_pre=(rho_n+rho_(n-1))/2.
// The pressure 'p', a suitable start value for the density 'rho_start', and the respective material are passed.
// NOTE: If the iteration doesn't converge then 'rho_start' is simply returned!
{
    int i=0;
    const double eps = 1.0e-7;        // (more or less) carefully chosen value, especially w.r.t. the eps in ANEOS_density! If this eps is not smaller, ANEOS_density can't calculate accurate enough for the break condition here to succeed!
    const int i_limit = 200;        // maximum number of iterations before algorithm is stopped
    double rho, rho_pre = rho_start;    // current and previous density value during iteration
    double rel_error;
#ifdef MORE_HYDROSTRUCT_DEBUG
    int j;
    double debug_rho[i_limit];
    double debug_rho_pre[i_limit];
    double debug_rel_error[i_limit];
#endif
    
    do
    {
        rho = ANEOS_density(e_compression_ANEOS(rho_pre, mat), p, mat);
        rel_error = fabs(rho-rho_pre) / rho_pre;        // relative error - relative to the previous value of the density(-iteration)
#ifdef MORE_HYDROSTRUCT_DEBUG
        debug_rho[i] = rho;
        debug_rho_pre[i] = rho_pre;
        debug_rel_error[i] = rel_error;
#endif
        rho_pre = (rho_pre+rho) / 2.0;    // using the mean value here gives probably a better approximation towards the final result, and prohibits convergence towards 2 fixed points
        i++;
    }
    while( (rel_error > eps) && (i < i_limit) );
    
/*
    // try iteration again but this time use simply rho_pre=rho_n
    if( i >= i_limit )
    {
        rho_pre = rho_start;
        i=0;
        
        do
        {
            rho = ANEOS_density(e_compression_ANEOS(rho_pre, mat), p, mat);
            rel_error = fabs(rho-rho_pre) / rho_pre;        // relative error - relative to the previous value of the density(-iteration)
#ifdef HYDROSTRUCT_DEBUG
            debug_rho[i] = rho;
            debug_rho_pre[i] = rho_pre;
            debug_rel_error[i] = rel_error;
#endif
            rho_pre = rho;
            i++;
        }
        while( (rel_error > eps) && (i < i_limit) );
    }
*/


/*
    // try iteration again but this time use rho_pre=(rho_n+rho_(n-1)+rho_(n-2))/3
    if( i >= i_limit )
    {
        double a;
        double rho_pre_pre = rho_start;
        rho_pre = ANEOS_density(e_compression_ANEOS(rho_pre_pre, mat), p, mat);
#ifdef HYDROSTRUCT_DEBUG
        debug_rho[0] = rho_pre;
        debug_rho_pre[0] = rho_pre_pre;
        debug_rel_error[0] = fabs(rho_pre-rho_pre_pre) / rho_pre_pre;
#endif
        i=1;
        
        do
        {
            rho = ANEOS_density(e_compression_ANEOS(rho_pre, mat), p, mat);
            rel_error = fabs(rho-rho_pre) / rho_pre;        // relative error - relative to the previous value of the density(-iteration)
#ifdef HYDROSTRUCT_DEBUG
            debug_rho[i] = rho;
            debug_rho_pre[i] = rho_pre;
            debug_rel_error[i] = rel_error;
#endif
            a = rho_pre;
            rho_pre = (rho_pre_pre+rho_pre+rho) / 3.0;    // using this mean value here gives probably a better approximation towards the final result, and prohibits convergence towards 3 fixed points
            rho_pre_pre = a;
            i++;
        }
        while( (rel_error > eps) && (i < i_limit) );
    }
*/
    
    if ( i >= i_limit )
    {
        fprintf(stderr, "WARNING: Probably no convergence during self-consistent density calculation in 'rho_self_consistent_ANEOS()', ");
        fprintf(stderr, "for p = %e, rho_start = %e, material = %s, e_compression(rho_start) = %e. Returned rho_start = %e ...\n", p, rho_start, mat->mat_name, e_compression_ANEOS(rho_start,mat), rho_start);
#ifdef MORE_HYDROSTRUCT_DEBUG
        for(j=0; j<i_limit; j++)
            fprintf(stderr, "    iteration no. %d:  rho_pre = %.16le    rho = %.16le    rel_error = %.16le\n", j, debug_rho_pre[j], debug_rho[j], debug_rel_error[j]);
#endif
        fflush(stderr);
        return(rho_start);  // return 'rho_start' if the iteration limit was exceeded
    }
    else
    {
        return(rho);
    }
}




double rho_self_consistent_ideal_gas(double p, double rho_start, material* mat)
// Solves the self consistency problem rho=rho(p,e(rho)), which can be done analytically for and ideal gas.
{
    return( pow(p/mat->ideal_gas.polytropic_K, 1.0/mat->ideal_gas.gamma) );
}




int array_index(double x, double* array, int n)
// Uses simple bisection to find the index 'i' in an ordered array (length 'n') that satisfies 'array[i] <= x < array[i+1]'.
// If x lies outside the array-covered values it returns -1.
{
    int i,i1,i2;    //current index and its lower and upper bound

    // return -1 if x lies outside the array-covered values
    if( x < array[0] || x >= array[n-1])
        return(-1);
    
    i1 = 0;
    i2 = n-1;
    do
    {
        i = (int)( (double)(i1+i2)/2.0 );
        if( array[i] <= x )
            i1 = i;    // i becomes new lower bound
        else
            i2 = i;    // i becomes new upper bound
    }
    while( (i2-i1)>1 );
    
    return(i1);
}




double bilinear_interpolation(double x, double y, double** table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y)
// Performs bilinear interpolation (2d lin. interp.) of values in 'table' which correspond to x- and y-values in 'xtab' and 'ytab'.
// The target values are 'x' and 'y'. 'ix' holds the index that satisfies 'xtab[ix] <= x < xtab[ix+1]' (similar for iy).
// 'n_x' holds the length of a row of x-values for a single y-value (similar for n_y).
// If (x,y) lies outside the table then ix<0 || iy<0 and the table values are (somewhat linearly) extrapolated.
{
    double normx = -1.0, normy = -1.0;
    double a, b, p = -1.0;
//    FILE *f;
    
    
    // if (x,y) lies outside table then extrapolate (somewhat linearly)
    if( ix < 0 || iy < 0 )
    {
        if( ix < 0 && iy < 0 )  // (x,y) lies in one of the 4 "corners"
        {
            if( x < xtab[0] && y < ytab[0] )
            {
                normx = (xtab[0]-x) / (xtab[1]-xtab[0]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                normy = (ytab[0]-y) / (ytab[1]-ytab[0]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = table[0][0] + normx*(table[0][0]-table[1][0]) + normy*(table[0][0]-table[0][1]);
            }
            else if( x < xtab[0] && y >= ytab[n_y-1] )
            {
                normx = (xtab[0]-x) / (xtab[1]-xtab[0]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                normy = (y-ytab[n_y-1]) / (ytab[n_y-1]-ytab[n_y-2]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = table[0][n_y-1] + normx*(table[0][n_y-1]-table[1][n_y-1]) + normy*(table[0][n_y-1]-table[0][n_y-2]);
            }
            else if( x >= xtab[n_x-1] && y < ytab[0] )
            {
                normx = (x-xtab[n_x-1]) / (xtab[n_x-1]-xtab[n_x-2]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                normy = (ytab[0]-y) / (ytab[1]-ytab[0]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = table[n_x-1][0] + normx*(table[n_x-1][0]-table[n_x-2][0]) + normy*(table[n_x-1][0]-table[n_x-1][1]);
            }
            else if( x >= xtab[n_x-1] && y >= ytab[n_y-1] )
            {
                normx = (x-xtab[n_x-1]) / (xtab[n_x-1]-xtab[n_x-2]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                normy = (y-ytab[n_y-1]) / (ytab[n_y-1]-ytab[n_y-2]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = table[n_x-1][n_y-1] + normx*(table[n_x-1][n_y-1]-table[n_x-2][n_y-1]) + normy*(table[n_x-1][n_y-1]-table[n_x-1][n_y-2]);
            }
            else
                ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table encountered for rho = %.16le and e = %.16le !\n", x, y)
        }
        else if( ix < 0 )
        {
            normy = (y-ytab[iy]) / (ytab[iy+1]-ytab[iy]);
            if( x < xtab[0] )
            {
                // linear interpolation in y-direction at xtab[0] and xtab[1]
                a = table[0][iy] + normy*(table[0][iy+1]-table[0][iy]);
                b = table[1][iy] + normy*(table[1][iy+1]-table[1][iy]);
                // linear extrapolation in x-direction from a and b
                normx = (x-xtab[0]) / (xtab[1]-xtab[0]);    // (always negative) distance from table end, normalized to x-spacing between 2 outermost table values
                p = a + normx*(b-a);
            }
            else if( x >= xtab[n_x-1] )
            {
                // linear interpolation in y-direction at xtab[n_x-1] and xtab[n_x-2]
                a = table[n_x-1][iy] + normy*(table[n_x-1][iy+1]-table[n_x-1][iy]);
                b = table[n_x-2][iy] + normy*(table[n_x-2][iy+1]-table[n_x-2][iy]);
                // linear extrapolation in x-direction from a and b
                normx = (x-xtab[n_x-1]) / (xtab[n_x-1]-xtab[n_x-2]);    // (always positive) distance from table end, normalized to x-spacing between 2 outermost table values
                p = a + normx*(a-b);
            }
            else
                ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table encountered for rho = %.16le and e = %.16le !\n", x, y)
        }
        else if( iy < 0 )
        {
            normx = (x-xtab[ix]) / (xtab[ix+1]-xtab[ix]);
            if( y < ytab[0] )
            {
                // linear interpolation in x-direction at ytab[0] and ytab[1]
                a = table[ix][0] + normx*(table[ix+1][0]-table[ix][0]);
                b = table[ix][1] + normx*(table[ix+1][1]-table[ix][1]);
                // linear extrapolation in y-direction from a and b
                normy = (y-ytab[0]) / (ytab[1]-ytab[0]);    // (always negative) distance from table end, normalized to y-spacing between 2 outermost table values
                p = a + normy*(b-a);
            }
            else if( y >= ytab[n_y-1] )
            {
                // linear interpolation in x-direction at ytab[n_y-1] and ytab[n_y-2]
                a = table[ix][n_y-1] + normx*(table[ix+1][n_y-1]-table[ix][n_y-1]);
                b = table[ix][n_y-2] + normx*(table[ix+1][n_y-2]-table[ix][n_y-2]);
                // linear extrapolation in y-direction from a and b
                normy = (y-ytab[n_y-1]) / (ytab[n_y-1]-ytab[n_y-2]);    // (always positive) distance from table end, normalized to y-spacing between 2 outermost table values
                p = a + normy*(a-b);
            }
            else
                ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table encountered for rho = %.16le and e = %.16le !\n", x, y)
        }
        else
            ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table encountered for rho = %.16le and e = %.16le !\n", x, y)
        
        // write a warning to warnings file
//        if ( (f = fopen("spheres_ini.warnings", "a")) == NULL )
//            ERRORTEXT("FILE ERROR! Cannot open 'spheres_ini.warnings' for appending!\n")
//        fprintf(f, "WARNING: At least one of rho = %e and e = %e is out of ANEOS lookup table range! Use extrapolated p(rho,e) = %e\n", x, y, p);
//        fclose(f);
        
        return(p);
    }
    
    
    // calculate normalized distances of x and y from (lower) table values
    normx = (x-xtab[ix]) / (xtab[ix+1]-xtab[ix]);
    normy = (y-ytab[iy]) / (ytab[iy+1]-ytab[iy]);
    
    // linear interpolation in x-direction at ytab[iy] and ytab[iy+1]
    a = table[ix][iy] + normx*(table[ix+1][iy]-table[ix][iy]);
    b = table[ix][iy+1] + normx*(table[ix+1][iy+1]-table[ix][iy+1]);
    
    // linear interpolation in y-direction between a and b
    return( a + normy*(b-a) );
        
}   // end function 'bilinear_interpolation()'




int discrete_value_table_lookup(double x, double y, int** table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y)
// Discrete (int) values in 'table' correspond to x- and y-values (doubles) in 'xtab' and 'ytab'.
// This function finds the closest "corner" (in the x-y-plane) of the respective cell and returns the value of 'table' in that corner.
// The target values are 'x' and 'y'. 'ix' holds the index that satisfies 'xtab[ix] <= x < xtab[ix+1]' (similar for iy).
// 'n_x' holds the length of a row of x-values for a single y-value (similar for n_y).
// If (x,y) lies outside the table then ix<0 || iy<0 and the closest (in the x-y-plane) value of 'table' is returned.
{
    int phase_flag = -1;
    double normx = -1.0, normy = -1.0;
//    FILE *f;
    
    
    // if (x,y) lies outside table then find the closest value (in the x-y-plane) of 'table'
    if( ix < 0 || iy < 0 )
    {
        if( ix < 0 && iy < 0 )  // (x,y) lies in one of the 4 "corners"
        {
            if( x < xtab[0] && y < ytab[0] )    // "lower left" corner
            {
                phase_flag = table[0][0];
            }
            else if( x < xtab[0] && y >= ytab[n_y-1] )  // "upper left" corner
            {
                phase_flag = table[0][n_y-1];
            }
            else if( x >= xtab[n_x-1] && y < ytab[0] )  // "lower right" corner
            {
                phase_flag = table[n_x-1][0];
            }
            else if( x >= xtab[n_x-1] && y >= ytab[n_y-1] ) // "upper right" corner
            {
                phase_flag = table[n_x-1][n_y-1];
            }
            else
                ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table in 'discrete_value_table_lookup()' encountered for rho = %.16le and e = %.16le !\n", x, y)
        }
        else if( ix < 0 )
        {
            normy = (y-ytab[iy]) / (ytab[iy+1]-ytab[iy]);
            if( normy >= 0.5 && normy <= 1.0 )
            {
                if( x < xtab[0] )
                {
                    phase_flag = table[0][iy+1];
                }
                else if( x >= xtab[n_x-1] )
                {
                    phase_flag = table[n_x-1][iy+1];
                }
                else
                    ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table in 'discrete_value_table_lookup()' encountered for rho = %.16le and e = %.16le !\n", x, y)
            }
            else if( normy < 0.5 && normy >= 0.0 )
            {
                if( x < xtab[0] )
                {
                    phase_flag = table[0][iy];
                }
                else if( x >= xtab[n_x-1] )
                {
                    phase_flag = table[n_x-1][iy];
                }
                else
                    ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table in 'discrete_value_table_lookup()' encountered for rho = %.16le and e = %.16le !\n", x, y)
            }
            else
                ERRORVAR("ERROR! 'normy' = %.16le (is not in [0,1]) in 'discrete_value_table_lookup()' ...\n", normy)
        }
        else if( iy < 0 )
        {
            normx = (x-xtab[ix]) / (xtab[ix+1]-xtab[ix]);
            if( normx >= 0.5 && normx <= 1.0 )
            {
                if( y < ytab[0] )
                {
                    phase_flag = table[ix+1][0];
                }
                else if( y >= ytab[n_y-1] )
                {
                    phase_flag = table[ix+1][n_y-1];
                }
                else
                    ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table in 'discrete_value_table_lookup()' encountered for rho = %.16le and e = %.16le !\n", x, y)
            }
            else if( normx < 0.5 && normx >= 0.0 )
            {
                if( y < ytab[0] )
                {
                    phase_flag = table[ix][0];
                }
                else if( y >= ytab[n_y-1] )
                {
                    phase_flag = table[ix][n_y-1];
                }
                else
                    ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table in 'discrete_value_table_lookup()' encountered for rho = %.16le and e = %.16le !\n", x, y)
            }
            else
                ERRORVAR("ERROR! 'normx' = %.16le (is not in [0,1]) in 'discrete_value_table_lookup()' ...\n", normx)
        }
        else
            ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table in 'discrete_value_table_lookup()' encountered for rho = %.16le and e = %.16le !\n", x, y)
        
        // write a warning to warnings file
//        if ( (f = fopen("spheres_ini.warnings", "a")) == NULL )
//            ERRORTEXT("FILE ERROR! Cannot open 'spheres_ini.warnings' for appending!\n")
//        fprintf(f, "WARNING: At least one of rho = %e and e = %e is out of ANEOS lookup table range! Use extrapolated phase-flag = %d\n", x, y, phase_flag);
//        fclose(f);
        
        return(phase_flag);
    }
    
    
    // calculate normalized distances of x and y from (lower) table values
    normx = (x-xtab[ix]) / (xtab[ix+1]-xtab[ix]);
    normy = (y-ytab[iy]) / (ytab[iy+1]-ytab[iy]);
    
    // find the closest "corner" (in the x-y-plane) and return respective value of 'table'
    if( normx >= 0.5 && normx <= 1.0 && normy >= 0.5 && normy <= 1.0 )  // "upper right" quadrant of cell
    {
        phase_flag = table[ix+1][iy+1];
    }
    else if( normx >= 0.5 && normx <= 1.0 && normy < 0.5 && normy >= 0.0 )  // "lower right" quadrant of cell
    {
        phase_flag = table[ix+1][iy];
    }
    else if( normx < 0.5 && normx >= 0.0 && normy >= 0.5 && normy <= 1.0 )  // "upper left" quadrant of cell
    {
        phase_flag = table[ix][iy+1];
    }
    else if( normx < 0.5 && normx >= 0.0 && normy < 0.5 && normy >= 0.0 )   // "lower left" quadrant of cell
    {
        phase_flag = table[ix][iy];
    }
    else
        ERRORVAR2("ERROR: Some odd behavior during \"discrete interpolation\" from ANEOS table in 'discrete_value_table_lookup()' encountered for rho = %.16le and e = %.16le !\n", x, y)
    
    return( phase_flag );
    
}   // end function 'discrete_value_table_lookup()'

