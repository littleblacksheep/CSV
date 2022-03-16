/* This program reads the radial profile for one body and converts it to the input format for radial profiles of 'spheres_ini'.
 * 
 * All units are SI.
 * Christoph Burger 09/Jul/2018
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <libconfig.h>


#define EOS_TYPE_ANEOS 7

#define PATHLENGTH 256
#define TRUE 1
#define FALSE 0

#define ERRORTEXT(x) {fprintf(stderr,x); exit(1);}
#define ERRORVAR(x,y) {fprintf(stderr,x,y); exit(1);}
#define ERRORVAR2(x,y,z) {fprintf(stderr,x,y,z); exit(1);}
#define ERRORVAR3(x,y,z,a) {fprintf(stderr,x,y,z,a); exit(1);}
#define ERRORVAR4(x,y,z,a,b) {fprintf(stderr,x,y,z,a,b); exit(1);}


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

typedef struct material_data
{
    int mat_type;
    char mat_name[PATHLENGTH];
    int eos_type;   // number identifying eos (see #defines)
    aneos_data aneos;
} material;

typedef struct _A_infile_line
{
    double depth;
    double T;
    double rho;
    int mat_type;
} A_infile_line;


double energy(double rho, double T, material *mat);
int array_index(double x, double* array, int n);
double bilinear_interpolation(double x, double y, double** table, double* xtab, double* ytab, int ix, int iy, int n_x, int n_y);
void allocate_ANEOS_table_memory(material *mat);
void free_ANEOS_table_memory(material *mat);
void load_ANEOS_table(material *mat);


void help(char* programname)
{
    fprintf(stdout, "\nUsage: %s [Options]\n", programname);
    fprintf(stdout, "Reads the radial profile for one body and converts it to the input format for radial profiles of 'spheres_ini' (first line ignored, then columns r-rho-e).\n\n");
    fprintf(stdout, "Options (all optional):\n");
    fprintf(stdout, "    -?                    displays this help message and exits\n");
    fprintf(stdout, "    -i input-file         specify file to read from\n");
    fprintf(stdout, "    -o output-file        specify file to write to\n");
    fprintf(stdout, "    -N                    input file format is the spheres_ini output file format for hydrostatic structures - first line ignored, then columns r-m-rho-e-p\n");
    fprintf(stdout, "    -A                    input file format is: first line ignored, then depth-T-rho-p-solidus-mattype, where depth is in km and negative!\n");
    fprintf(stdout, "                          this input file format is used together with ANEOS materials, where mattype corresponds to the entries in the 'material.cfg' file (flag '-m')\n");
    fprintf(stdout, "    -m material-config    specify 'material.cfg' file - only relevant if flag '-A' is chosen\n");
    fprintf(stdout, "\n");
}



int main(int argc, char* argv[])
{
    int i,j;
    char infile[PATHLENGTH];
    char outfile[PATHLENGTH];
    char matfile[PATHLENGTH];
    FILE *ifl, *ofl, *mfl;
    int N_flag = FALSE;
    int A_flag = FALSE;
    double r, rho, e;
    config_t config;    // configuration structure
    config_setting_t *all_materials, *one_material, *subset; // configuration settings structures
    const char *tmpstring;     // allocation of storage is done automatically by libconfig (when running 'config_setting_lookup_string')
    const char *tmp_table_file;     // allocation of storage is done automatically by libconfig (when running 'config_setting_lookup_string')
    int n_mat;  // number of materials found in 'material.cfg' file
    material *mat;
    A_infile_line *A_ifl_data;
    int N_ifl_lines = -1;
    double R_tot = -1.0;
    
    
    while ( (i = getopt(argc, argv, "?i:o:NAm:")) != -1 ) // int-representations of command line options are successively saved in i
        switch((char)i)
        {
            case '?':
                help(*argv);
                exit(0);
            case 'i':
                strncpy(infile, optarg, PATHLENGTH);
                break;
            case 'o':
                strncpy(outfile, optarg, PATHLENGTH);
                break;
            case 'N':
                N_flag = TRUE;
                break;
            case 'A':
                A_flag = TRUE;
                break;
            case 'm':
                strncpy(matfile, optarg, PATHLENGTH);
                break;
            default:
                help(*argv);
                exit(1);
        }
    if( N_flag && A_flag )
        ERRORTEXT("ERROR. You can't choose both cmd-line flags '-N' and '-A' ...\n")
    
    
    if( N_flag )
    {
        if ( (ifl = fopen(infile,"r")) == NULL )
            ERRORVAR("FILE ERROR! Cannot open '%s' for reading!\n", infile)
        if ( (ofl = fopen(outfile,"w")) == NULL )
            ERRORVAR("FILE ERROR! Cannot open '%s' for writing!\n", outfile)
        
        fprintf(ofl, "#  r (m)    rho (kg/m^3)    e (J/kg)\n");     // write first line of outfile
        fscanf(ifl, "%*[^\n]\n");   // ignore first line of infile
        
        while( fscanf(ifl, "%le %*le %le %le%*[^\n]\n", &r, &rho, &e ) == 3 )
            fprintf(ofl, "%e\t%e\t%e\n", r, rho, e);
        
        fclose(ifl);
        fclose(ofl);
    }
    
    
    if( A_flag )
    {
        // open infile and outfile
        if ( (ifl = fopen(infile,"r")) == NULL )
            ERRORVAR("FILE ERROR! Cannot open '%s' for reading!\n", infile)
        if ( (ofl = fopen(outfile,"w")) == NULL )
            ERRORVAR("FILE ERROR! Cannot open '%s' for writing!\n", outfile)
        
        
        // read all materials from 'material.cfg' file
        config_init(&config);
        if( !config_read_file(&config, matfile) )
            ERRORVAR("ERROR when reading 'material.cfg' file '%s'!\n", matfile)
        all_materials = config_lookup(&config, "materials");
        if( all_materials == NULL )
            ERRORVAR("ERROR! Couldn't find materialconfiguration settings in '%s' ...\n", matfile)
        n_mat = config_setting_length(all_materials);
        
        if( ( mat = (material*)malloc(n_mat*sizeof(material)) ) == NULL )
            ERRORTEXT("ERROR during memory allocation for 'mat'!\n")
        
        for(i=0; i<n_mat; i++)   // loop over all materials, evaluate whether they use ANEOS, and read lookup tables if so
        {
            one_material = config_setting_get_elem(all_materials, i);
            if( one_material == NULL )
                ERRORVAR("ERROR! Something's messed up with the settings in '%s' ...\n", matfile)
            
            if( !config_setting_lookup_int(one_material, "ID", &(mat[i].mat_type) ) )
                ERRORVAR("ERROR! Found material without ID in '%s' ...\n", matfile)
            config_setting_lookup_string(one_material, "name", &tmpstring );
            strcpy(mat[i].mat_name, tmpstring);    // it's necessary to go via 'tmpstring' because 'config_destroy()' deallocates all memory that was used in 'config_setting_lookup_string' - even if it didn't allocate it ...
            
            fprintf(stdout, "Found material '%s' with ID/mat_type = %d ...\n", mat[i].mat_name, mat[i].mat_type);
            
            subset = config_setting_get_member(one_material, "eos");
            if( subset == NULL )
                ERRORVAR3("ERROR. Can't find eos parameters for material %d ('%s') in '%s' ...\n", mat[i].mat_type, mat[i].mat_name, matfile)
            config_setting_lookup_int(subset, "type", &(mat[i].eos_type) );
            
            if( mat[i].eos_type == EOS_TYPE_ANEOS )
            {
                fprintf(stdout, "    which uses ANEOS:\n");
                
//                if( !config_setting_lookup_float(subset, "aneos_rho_0", &(mat[i].aneos.rho_0) ) )
//                    ERRORVAR2("ERROR! Didn't find 'aneos_rho_0' for material with mat_type %d in '%s' ...\n", mat[i].mat_type, matfile)
                if( !config_setting_lookup_string(subset, "table_path", &tmp_table_file ) )
                    ERRORVAR2("ERROR! Didn't find 'table_path' for material with mat_type %d in '%s' ...\n", mat[i].mat_type, matfile)
                strcpy(mat[i].aneos.table_file, tmp_table_file);  // it's necessary to go via 'tmp_table_file' because 'config_destroy()' deallocates all memory that was used in 'config_setting_lookup_string' - even if it didn't allocate it ...
                if( !config_setting_lookup_int(subset, "n_rho", &(mat[i].aneos.n_rho) ) )
                    ERRORVAR2("ERROR! Didn't find 'n_rho' for material with mat_type %d in '%s' ...\n", mat[i].mat_type, matfile)
                if( !config_setting_lookup_int(subset, "n_e", &(mat[i].aneos.n_e) ) )
                    ERRORVAR2("ERROR! Didn't find 'n_e' for material with mat_type %d in '%s' ...\n", mat[i].mat_type, matfile)
//                if( !config_setting_lookup_float(subset, "aneos_e_norm", &(mat[i].aneos.e_norm) ) )
//                    ERRORVAR2("ERROR! Didn't find 'aneos_e_norm' for material with mat_type %d in '%s' ...\n", mat[i].mat_type, matfile)
//                if( !config_setting_lookup_float(subset, "aneos_bulk_cs", &(mat[i].aneos.bulk_cs) ) )
//                    ERRORVAR2("ERROR! Didn't find 'aneos_bulk_cs' for material with mat_type %d in '%s' ...\n", mat[i].mat_type, matfile)
                fprintf(stdout, "      table_path = %s\n      n_rho = %d\n      n_e = %d\n\n", mat[i].aneos.table_file, mat[i].aneos.n_rho, mat[i].aneos.n_e );
                
                allocate_ANEOS_table_memory( &(mat[i]) );
                load_ANEOS_table( &(mat[i]) );
            }
            else
            {
                fprintf(stdout, "    which does not use ANEOS.\n\n");
            }
        }
        
        
        // read whole infile
        if( ( A_ifl_data = (A_infile_line*)malloc(sizeof(A_infile_line)) ) == NULL )
            ERRORTEXT("ERROR during memory allocation for infile data!\n")
        fscanf(ifl, "%*[^\n]\n");   // ignore first line of infile
        i = 0;
        while( fscanf(ifl, "%le %le %le %*le %*le %d%*[^\n]\n", &(A_ifl_data[i].depth), &(A_ifl_data[i].T), &(A_ifl_data[i].rho), &(A_ifl_data[i].mat_type) ) == 4 )
        {
            A_ifl_data[i].depth *= 1000.0;  // transform from km to m (but it's still a negative value!)
            i++;
            if( ( A_ifl_data = (A_infile_line*)realloc(A_ifl_data,(i+1)*sizeof(A_infile_line)) ) == NULL )
                ERRORTEXT("ERROR during memory allocation for infile data!\n")
        }
        N_ifl_lines = i;
        fprintf(stdout, "Found %d lines of data in the input file.\n", N_ifl_lines);
        
        
        // Convert temperatures to int. energies, and write everything to the outfile (in reverse order, i.e. starting at r = 0)
        R_tot = (-1.0) * A_ifl_data[N_ifl_lines-1].depth;
        fprintf(stdout, "Found R_tot = %e.\n", R_tot);
        fprintf(ofl, "#  r (m)    rho (kg/m^3)    e (J/kg)\n");     // write first line of outfile
        
        for(i=N_ifl_lines-1; i>=0; i--)
        {
            j = A_ifl_data[i].mat_type;
            if( mat[j].eos_type == EOS_TYPE_ANEOS )
            {
                fprintf(ofl, "%e\t%e\t%e\n", R_tot + A_ifl_data[i].depth, A_ifl_data[i].rho, energy(A_ifl_data[i].rho, A_ifl_data[i].T, &(mat[j])) );
            }
            else
                ERRORVAR4("ERROR. Line no. %d in '%s' has mat_type = %d, which doesn't correspond to an ANEOS material, but to an eos_type = %d!\n",
                          i, infile, A_ifl_data[i].mat_type, mat[A_ifl_data[i].mat_type].eos_type)
        }
        
        fprintf(stdout, "Wrote %d lines to the output file '%s'.\n\n", N_ifl_lines, outfile);
        
        
        // clean up
        config_destroy(&config);
        fclose(ifl);
        fclose(ofl);
        free(A_ifl_data);
        for(i=0; i<n_mat; i++)
            if( mat[i].eos_type == EOS_TYPE_ANEOS )
                free_ANEOS_table_memory(&(mat[i]));
        free(mat);
    }   // end 'if( A_flag )'
    
    
    return(0);
}   // end 'main()'




double energy(double rho, double T, material *mat)
// Computes e for a given rho and T, from ANEOS lookup tables with rho and e as independent variables, via bisection.
// The respective ANEOS lookup tables have to be loaded externally.
{
    int i;
    int i_limit = 100;
    const double eps = 1.0e-12;
    double e1, e2, e;
    int n_e = mat->aneos.n_e;
    int n_rho = mat->aneos.n_rho;
    double tmp_T;
    int i_rho, i_e;
    
    
    // start values
    e1 = mat->aneos.e[0];
    e2 = mat->aneos.e[n_e-1];
    e = (e1 + e2) / 2.0;
    
    // iteration
    i = 0;
    do
    {
        i_rho = array_index(rho, mat->aneos.rho, n_rho);
        i_e = array_index(e, mat->aneos.e, n_e);
        tmp_T = bilinear_interpolation(rho, e, mat->aneos.T, mat->aneos.rho, mat->aneos.e, i_rho, i_e, n_rho, n_e);
        
        if( tmp_T < T )
            e1 = e;
        else
            e2 = e;
        
        e = (e1 + e2) / 2.0;
        i++;
    }
    while( ((e2-e1)/e >= eps) && (i <= i_limit) );
    
    if ( i > i_limit )
        ERRORVAR3("ERROR during ANEOS energy calculation for material '%s' with rho = %e and T = %e! Iteration limit exceeded, probably no convergence!\n",
                  mat->mat_name, rho, T);
    
    return(e);
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
    
    
    // if (x,y) lies outside table then extrapolate (somewhat linearly) and print a warning
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
                ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table encountered for rho = %e and e = %e !\n", x, y)
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
                ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table encountered for rho = %e and e = %e !\n", x, y)
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
                ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table encountered for rho = %e and e = %e !\n", x, y)
        }
        else
            ERRORVAR2("ERROR: Some odd behavior during extrapolation from ANEOS table encountered for rho = %e and e = %e !\n", x, y)
        
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


