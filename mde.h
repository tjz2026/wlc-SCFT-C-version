#ifndef	MDE_H
#define MDE_H
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
#include "matrix.h"
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
extern vector<array_struct_int_int> basis_function_SHF; // basis function for spherical harmonics function
extern vector<array_struct_int_int_int_double> GAMA_nonzero_1D;
extern vector<array_struct_int_int_double> J11ij_nonzero_1D;
extern vector<array_struct_int_int_double> J22ij_nonzero_1D;
extern vector<array_struct_int_int_double> J12ij_nonzero_1D;
extern vector<array_struct_int_int_double> J13ij_nonzero_1D;
extern vector<array_struct_int_int_double> J23ij_nonzero_1D;

extern vector<vector<array_struct_int_int_double> > THETA_nonzero_2D;
extern double **matrix_Rx,**matrix_Ry, **matrix_Rz;
extern double ***GAMA;
extern double ***qA;
extern double ***qAstar;
extern double ***qB;
extern double ***qBstar;
extern double *WA;
extern double *WB;
extern double **WA_out;
extern double **WB_out;
extern double **dA_anderson; // WA_out - WA_in
extern double **dB_anderson; // WB_out - WB_in
extern double **M11_out;
extern double **M33_out;
extern double **M12_out;
extern double **M13_out;
extern double **M23_out;
extern double **d11_anderson;
extern double **d33_anderson;
extern double **d12_anderson;
extern double **d13_anderson;
extern double **d23_anderson;
extern double *RA;
extern double *RB;
extern double *R_half;
extern double *R_end;
extern double **RHOA;
extern double **RHOB;

extern double ***M_OP; // M order parameter potential
extern double ***S_OP; // S order parameter
extern double ***SA_OP; 
extern double ***SB_OP;

extern double ***THETAij;
extern double **THETAij_M11_M22;
extern double **THETAij_M33;
extern double **THETAij_M12;
extern double **THETAij_M13;
extern double **THETAij_M23;
extern double **J11ij, **J22ij, **J12ij, **J13ij, **J23ij;

extern double ***sa_G_R_inv,  ***sa_G_I_inv;
extern int   ***ija_G_R_inv, ***ija_G_I_inv;

//extern double   ***GA1_R_inverse,    ***GA1_I_inverse;
//extern double **sa_GA1_R_inverse,  **sa_GA1_I_inverse;
//extern int   **ija_GA1_R_inverse, **ija_GA1_I_inverse;
//
//extern double   ***GA2_R_inverse,    ***GA2_I_inverse;
//extern double **sa_GA2_R_inverse,  **sa_GA2_I_inverse;
//extern int   **ija_GA2_R_inverse, **ija_GA2_I_inverse;
//
//extern double   ***GA3_R_inverse,    ***GA3_I_inverse;
//extern double **sa_GA3_R_inverse,  **sa_GA3_I_inverse;
//extern int   **ija_GA3_R_inverse, **ija_GA3_I_inverse;
//
//extern double   ***GB3_R_inverse,    ***GB3_I_inverse;
//extern double **sa_GB3_R_inverse,  **sa_GB3_I_inverse;
//extern int   **ija_GB3_R_inverse, **ija_GB3_I_inverse;
//
//extern double   ***GA3_star_R_inverse,    ***GA3_star_I_inverse;
//extern double **sa_GA3_star_R_inverse,  **sa_GA3_star_I_inverse;
//extern int   **ija_GA3_star_R_inverse, **ija_GA3_star_I_inverse;
//
//extern double   ***GB1_star_R_inverse,    ***GB1_star_I_inverse;
//extern double **sa_GB1_star_R_inverse,  **sa_GB1_star_I_inverse;
//extern int   **ija_GB1_star_R_inverse, **ija_GB1_star_I_inverse;
//
//extern double   ***GB2_star_R_inverse,    ***GB2_star_I_inverse;
//extern double **sa_GB2_star_R_inverse,  **sa_GB2_star_I_inverse;
//extern int   **ija_GB2_star_R_inverse, **ija_GB2_star_I_inverse;
//
//extern double   ***GB3_star_R_inverse,    ***GB3_star_I_inverse;
//extern double **sa_GB3_star_R_inverse,  **sa_GB3_star_I_inverse;
//extern int   **ija_GB3_star_R_inverse, **ija_GB3_star_I_inverse;



extern double pff_global,FE_global;
extern double ta_diff_global,tb_diff_global,tm_diff_global,M_v;//M_v;
// M_v: integrate on z, that means the volume; 
//////////////////  For  MPI  //////////////////////
// int myid, numprocs;
extern int myid, numprocs;
//extern int local_nx, local_x_start, local_ny_after_transpose, local_y_start_after_transpose, total_local_size;





//extern fftwnd_mpi_plan plan3d;
//extern fftwnd_mpi_plan plan3d_bak;
//extern fftw_complex *local_data;

extern pfft_plan   plan_forw, plan_back;
extern pfft_complex *in, *out;
extern ptrdiff_t local_ni[3], local_i_start[3];
extern ptrdiff_t local_no[3], local_o_start[3];


/////////////////////////////////////////////////////////////////////////////////////////
int index_updated_matrix_anderson(int index_i, int n_r);
void BF();
/////////////////////////////////////////////////////////////////////////////////////////
//void matrix_G_inverse(int dim_M_Bar, double coeff_deltaij, double coeff_Rij, double kapa_temp, 
//					  double **matrix_Rxx, double **matrix_Ryy, double **matrix_Rzz, double ***GAB_R_inverse, double ***GAB_I_inverse); // coeff_Rij=1.0 for G; coeff_Rij=-1.0 for G_star
void matrix_G_inverse(int dim_M_Bar, int K_i,int K_j, int K_k, double coeff_deltaij, double coeff_Rij, double kapa_temp, 
					  double **matrix_Rxx, double **matrix_Ryy, double **matrix_Rzz, double **GAB_R_inverse, double **GAB_I_inverse); 
/////////////////////////////////////////////////////////////////////////////////////////
void initial();
/////////////////////////////////////////////////////////////////////////////////////////
void clean();
void MDE_Pr_cal( int orient,int s, double *Wsp, double **Pr,double ***q_step); 
 void MDE_Pk_cal(double **Pr, double ** Pk_Real, double **Pk_Imag);
void MDE_get_previous_q(int orient, int s, double ***q_step);
void MDE_sparse_matvec(int orient,int s, double **Pk_Real,double **Pk_Imag,double **q_temp_Real,double **q_temp_Imag);
void MDE_get_real_q(int orient,int s, double **q_temp_Real,double **q_temp_Imag);
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
void calc_THETAij();
/////////////////////////////////////////////////////////////////////////////////////////
void MDE_q();
/////////////////////////////////////////////////////////////////////////////////////////
//void MDE_qstar();
/////////////////////////////////////////////////////////////////////////////////////////
void density();
/////////////////////////////////////////////////////////////////////////////////////////
double free_energy();
/////////////////////////////////////////////////////////////////////////////////////////
void iterate();
/////////////////////////////////////////////////////////////////////////////////////////
void output1();
/////////////////////////////////////////////////////////////////////////////////////////
void output();
/////////////////////////////////////////////////////////////////////////////////////////
void readw();
/////////////////////////////////////////////////////////////////////////////////////////
void initw();
/////////////////////////////////////////////////////////////////////////////////////////
int converge();
/////////////////////////////////////////////////////////////////////////////////////////
int converge_final();
/////////////////////////////////////////////////////////////////////////////////////////
void init_parameters();
/////////////////////////////////////////////////////////////////////////////////////////
void cal_scft_final();
/////////////////////////////////////////////////////////////////////////////////////////
double cal_scft(double tz);
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
#endif
