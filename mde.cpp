/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
#include "mde.h"
#include <pfft.h>
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
vector<array_struct_int_int> basis_function_SHF; // basis function for spherical harmonics function
vector<array_struct_int_int_int_double> GAMA_nonzero_1D;
vector<array_struct_int_int_double> J11ij_nonzero_1D;
vector<array_struct_int_int_double> J22ij_nonzero_1D;
vector<array_struct_int_int_double> J12ij_nonzero_1D;
vector<array_struct_int_int_double> J13ij_nonzero_1D;
vector<array_struct_int_int_double> J23ij_nonzero_1D;

//vector<array_struct_int_int_double> J11ij_nonzero_1D_B;
//vector<array_struct_int_int_double> J22ij_nonzero_1D_B;
//vector<array_struct_int_int_double> J12ij_nonzero_1D_B;
//vector<array_struct_int_int_double> J13ij_nonzero_1D_B;
//vector<array_struct_int_int_double> J23ij_nonzero_1D_B;
vector<vector<array_struct_int_int_double> > THETA_nonzero_2D;
double **matrix_Rx,**matrix_Ry, **matrix_Rz;
double ***GAMA;
double ***qA;
double ***qAstar;
double ***qB;
double ***qBstar;
double *WA;
double *WB;
double **WA_out;
double **WB_out;
double **dA_anderson; // WA_out - WA_in
double **dB_anderson; // WB_out - WB_in
double **M11_out;
double **M33_out;
double **M12_out;
double **M13_out;
double **M23_out;
double **d11_anderson;
double **d33_anderson;
double **d12_anderson;
double **d13_anderson;
double **d23_anderson;
double *RA;
double *RB;
double *R_half;
double *R_end;
double **RHOA;
double **RHOB;

double ***M_OP; // M order parameter potential
double ***S_OP; // S order parameter
double ***SA_OP; 
double ***SB_OP;

double ***THETAij;
double **THETAij_M11_M22;
double **THETAij_M33;
double **THETAij_M12;
double **THETAij_M13;
double **THETAij_M23;
double **J11ij, **J22ij, **J12ij, **J13ij, **J23ij;

//double ***THETAij_B;
//double **THETAij_M11_M22_B;
//double **THETAij_M33_B;
//double **THETAij_M12_B;
//double **THETAij_M13_B;
//double **THETAij_M23_B;
//double **J11ij_B, **J22ij_B, **J12ij_B, **J13ij_B, **J23ij_B;

// G_indx=0,1,2,3;4,5,6,7. for GA1,GA2,GA3,GB3;GB1star,GB2star2,GB3star,GA3star.

double ***sa_G_R_inv,  ***sa_G_I_inv;
int   ***ija_G_R_inv, ***ija_G_I_inv;

//double   ***GA1_R_inverse,    ***GA1_I_inverse;
//double **sa_GA1_R_inverse,  **sa_GA1_I_inverse;
//int   **ija_GA1_R_inverse, **ija_GA1_I_inverse;
//
//double   ***GA2_R_inverse,    ***GA2_I_inverse;
//double **sa_GA2_R_inverse,  **sa_GA2_I_inverse;
//int   **ija_GA2_R_inverse, **ija_GA2_I_inverse;
//
//double   ***GA3_R_inverse,    ***GA3_I_inverse;
//double **sa_GA3_R_inverse,  **sa_GA3_I_inverse;
//int   **ija_GA3_R_inverse, **ija_GA3_I_inverse;
//
//double   ***GB3_R_inverse,    ***GB3_I_inverse;
//double **sa_GB3_R_inverse,  **sa_GB3_I_inverse;
//int   **ija_GB3_R_inverse, **ija_GB3_I_inverse;
//
//double   ***GA3_star_R_inverse,    ***GA3_star_I_inverse;
//double **sa_GA3_star_R_inverse,  **sa_GA3_star_I_inverse;
//int   **ija_GA3_star_R_inverse, **ija_GA3_star_I_inverse;
//
//double   ***GB1_star_R_inverse,    ***GB1_star_I_inverse;
//double **sa_GB1_star_R_inverse,  **sa_GB1_star_I_inverse;
//int   **ija_GB1_star_R_inverse, **ija_GB1_star_I_inverse;
//
//double   ***GB2_star_R_inverse,    ***GB2_star_I_inverse;
//double **sa_GB2_star_R_inverse,  **sa_GB2_star_I_inverse;
//int   **ija_GB2_star_R_inverse, **ija_GB2_star_I_inverse;
//
//double   ***GB3_star_R_inverse,    ***GB3_star_I_inverse;
//double **sa_GB3_star_R_inverse,  **sa_GB3_star_I_inverse;
//int   **ija_GB3_star_R_inverse, **ija_GB3_star_I_inverse;




double pff_global,FE_global;
double ta_diff_global,tb_diff_global,tm_diff_global,M_v;//M_v;
// M_v: integrate on z, that means the volume; 
double lambda_WAB_anderson;
int Num_iteration_step_WM;
double lambda_M_anderson;
int Num_iteration_step_M;
double dx;
double dy;
double dz;
//////////////////  For  MPI  //////////////////////
int myid, numprocs;
char name1[80],name2[80],name3[80],name4[80],name5[80],name6[80],name7[80],name8[80],name9[80],name10[80];

// 3D fft with 2d pencil decomposition from PFFT 
// "in" and "out" are used to store the forward and backward fft local data, here, we use in place transform, hence, only "in" is used
pfft_complex *in, *out;
// fft forward (backward) transform plan : plan_forw and plan_back
pfft_plan plan_forw=NULL, plan_back=NULL;
// local_ni[0] is the local grid size on X axis, and local_ni[1],local_ni[2] for Y, Z axis.
 ptrdiff_t local_ni[3], local_i_start[3];
// local_no is the offset on each dimension, local_no[0]=SIDEx/np[0].
 ptrdiff_t local_no[3], local_o_start[3];



/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
int index_updated_matrix_anderson(int index_i, int n_r)
{
	// this function is designed for locating the index of the preceding steps e.g. k-m  or k-n in the notes
	// if n_r=n_r_WAB=5
	// 0  --  0  6   12
	// 1  --  1  7   13
	// 2  --  2  8   14
	// 3  --  3  9   15
	// 4  --  4  10  16
	// 5  --  5  11  17
	///////////////////////
	
	int result_i=index_i;
	
	if (index_i<0)
		result_i = index_i + (n_r+1);
	
	return result_i;
}
/************************************************************************/
/////////////////////////////////////////////////////////////////////////
void BF() // constructing the basis function for spherical harmonics
{
	int l,m;
	array_struct_int_int test;

	for(l=0;l<=L_Bar;l++){
		for(m=-l;m<=l;m++){
			test.l=l;
			test.m=m;
			basis_function_SHF.push_back(test);
		}
	}


}
/*****************************************************************************************/
///////////////////////////////////////////////////////////////////////////////////////////
void matrix_G_inverse(int dim_M_Bar, int K_i,int K_j, int K_k, double coeff_deltaij, double coeff_Rij, double kapa_temp, 
					  double **matrix_Rxx, double **matrix_Ryy, double **matrix_Rzz, double **GAB_R_inverse, double **GAB_I_inverse) 
{// coeff_Rij=1.0 for G; coeff_Rij=-1.0 for G_star
	int i,j;
	int K_ii;
	int K_jj;
	int K_kk;
	int K;


	double **G_R=dmatrix(0,dim_M_Bar-1,0,dim_M_Bar-1);
	double **G_I=dmatrix(0,dim_M_Bar-1,0,dim_M_Bar-1);
	double **G_R_inverse=dmatrix(0,dim_M_Bar-1,0,dim_M_Bar-1);
	double **G_I_inverse=dmatrix(0,dim_M_Bar-1,0,dim_M_Bar-1);
	

		if ((K_i+local_i_start[0])<=(SIDEx/2))
			K_ii = K_i+local_i_start[0];
		else K_ii = (K_i+local_i_start[0])-SIDEx;
		
#ifdef DIM1
// do nothing
#else
		if  ((K_j+local_i_start[1])<=SIDEy/2)
			K_jj = K_j+local_i_start[1];
		else K_jj = (K_j+local_i_start[1])-SIDEy;
#ifdef DIM2
// do nothing 
#else
		if (K_k<=(SIDEz/2))
		    K_kk = K_k;
		else K_kk = K_k-SIDEz;
#endif
#endif
				for(i=0;i<dim_M_Bar;i++){
					for(j=0;j<dim_M_Bar;j++){
						G_R[i][j]=0.0;
#ifdef DIM1
						G_I[i][j]=coeff_Rij*ds*(matrix_Rzz[i][j]*(2.0*PI*K_ii/(SIDEx*dx)));
#endif
#ifdef DIM2
						G_I[i][j]=coeff_Rij*ds*(matrix_Rxx[i][j]*(2.0*PI*K_ii/(SIDEx*dx))
							+  matrix_Ryy[i][j]*(2.0*PI*K_jj/(SIDEy*dy)));
#endif
#ifdef DIM3
						G_I[i][j]=coeff_Rij*ds*(matrix_Rxx[i][j]*(2.0*PI*K_ii/(SIDEx*dx))
							+  matrix_Ryy[i][j]*(2.0*PI*K_jj/(SIDEy*dy)) + matrix_Rzz[i][j]*(2.0*PI*K_kk/(SIDEz*dz)));
#endif

					    if (i==j) G_R[i][j]=coeff_deltaij + ds*NN*basis_function_SHF[j].l*(basis_function_SHF[j].l+1)/(2.0*kapa_temp);
					}
				}

				inverse_matrix_complex(dim_M_Bar, G_R, G_I, G_R_inverse, G_I_inverse);

				for(i=0;i<dim_M_Bar;i++){
					for(j=0;j<dim_M_Bar;j++){
						GAB_R_inverse[i][j]=G_R_inverse[i][j];
						GAB_I_inverse[i][j]=G_I_inverse[i][j];
					}
				}
				
///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////	
	free_dmatrix(G_R,0,dim_M_Bar-1,0,dim_M_Bar-1);
	free_dmatrix(G_I,0,dim_M_Bar-1,0,dim_M_Bar-1);
	free_dmatrix(G_R_inverse,0,dim_M_Bar-1,0,dim_M_Bar-1);
	free_dmatrix(G_I_inverse,0,dim_M_Bar-1,0,dim_M_Bar-1);

}
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
void initial()
{
	matrix_Rx=dmatrix(0,M_Bar-1,0,M_Bar-1);
	matrix_Ry=dmatrix(0,M_Bar-1,0,M_Bar-1);
	matrix_Rz=dmatrix(0,M_Bar-1,0,M_Bar-1);
	GAMA=f3tensor(0,M_Bar-1,0,M_Bar-1,0,M_Bar-1);
	qA=f3tensor(0,NA,0,LOCAL_SIZE-1,0,M_Bar-1); // block A: 0 ~ NA, the segment NA is looked as the component A;
	qAstar=f3tensor(0,NA,0,LOCAL_SIZE-1,0,M_Bar-1);
	qB=f3tensor(0,NB,0,LOCAL_SIZE-1,0,M_Bar-1);
	qBstar=f3tensor(0,NB,0,LOCAL_SIZE-1,0,M_Bar-1);
	WA=dvector(0,LOCAL_SIZE-1);
	WB=dvector(0,LOCAL_SIZE-1);
	WA_out=dmatrix(0,n_r_WAB,0,LOCAL_SIZE-1);
	WB_out=dmatrix(0,n_r_WAB,0,LOCAL_SIZE-1);
	dA_anderson=dmatrix(0,n_r_WAB,0,LOCAL_SIZE-1);
	dB_anderson=dmatrix(0,n_r_WAB,0,LOCAL_SIZE-1);

	M11_out=dmatrix(0,n_r_M,0,LOCAL_SIZE-1);
	M33_out=dmatrix(0,n_r_M,0,LOCAL_SIZE-1);
	M12_out=dmatrix(0,n_r_M,0,LOCAL_SIZE-1);
	M13_out=dmatrix(0,n_r_M,0,LOCAL_SIZE-1);
	M23_out=dmatrix(0,n_r_M,0,LOCAL_SIZE-1);
	d11_anderson=dmatrix(0,n_r_M,0,LOCAL_SIZE-1);
	d33_anderson=dmatrix(0,n_r_M,0,LOCAL_SIZE-1);
	d12_anderson=dmatrix(0,n_r_M,0,LOCAL_SIZE-1);
	d13_anderson=dmatrix(0,n_r_M,0,LOCAL_SIZE-1);
	d23_anderson=dmatrix(0,n_r_M,0,LOCAL_SIZE-1);

	RA=dvector(0,LOCAL_SIZE-1);
	RB=dvector(0,LOCAL_SIZE-1);
	R_half=dvector(0,LOCAL_SIZE-1);
	R_end=dvector(0,LOCAL_SIZE-1);
	RHOA=dmatrix(0,LOCAL_SIZE-1,0,M_Bar-1);
	RHOB=dmatrix(0,LOCAL_SIZE-1,0,M_Bar-1);


	M_OP=f3tensor(0, LOCAL_SIZE-1, 0, N_dim_ddm-1, 0, N_dim_ddm-1);
	S_OP=f3tensor(0, LOCAL_SIZE-1, 0, N_dim_ddm-1, 0, N_dim_ddm-1);
	SA_OP=f3tensor(0, LOCAL_SIZE-1, 0, N_dim_ddm-1, 0, N_dim_ddm-1);
	SB_OP=f3tensor(0, LOCAL_SIZE-1, 0, N_dim_ddm-1, 0, N_dim_ddm-1);

	THETAij=f3tensor(0,LOCAL_SIZE-1,0,M_Bar-1,0,M_Bar-1);
	THETAij_M11_M22=dmatrix(0,M_Bar-1,0,M_Bar-1);
	THETAij_M33=dmatrix(0,M_Bar-1,0,M_Bar-1);
	THETAij_M12=dmatrix(0,M_Bar-1,0,M_Bar-1);
	THETAij_M13=dmatrix(0,M_Bar-1,0,M_Bar-1);
	THETAij_M23=dmatrix(0,M_Bar-1,0,M_Bar-1);

	J11ij=dmatrix(0,M_Bar-1,0,M_Bar-1);
	J22ij=dmatrix(0,M_Bar-1,0,M_Bar-1);
	J12ij=dmatrix(0,M_Bar-1,0,M_Bar-1);
	J13ij=dmatrix(0,M_Bar-1,0,M_Bar-1);
	J23ij=dmatrix(0,M_Bar-1,0,M_Bar-1);
	
// sparse matrix stored as CSR format, 8 matrixs are used.
	 sa_G_R_inv=f3tensor(0,7,1,LOCAL_SIZE,1,namax);
	ija_G_R_inv=f3tensor_int(0,7,1,LOCAL_SIZE,1,namax);
	 sa_G_I_inv=f3tensor(0,7,1,LOCAL_SIZE,1,namax);
	ija_G_I_inv=f3tensor_int(0,7,1,LOCAL_SIZE,1,namax);


    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////for M///////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////

	int i,j,k;
	double temp;
	BF();

	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			for(k=0;k<M_Bar;k++){
				GAMA[i][j][k]=Triple_product(basis_function_SHF[i].l, 
					basis_function_SHF[j].l, basis_function_SHF[k].l, basis_function_SHF[i].m, basis_function_SHF[j].m, basis_function_SHF[k].m);
			}
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////for matrix Rx, Ry, Rz /////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	temp=-1.0/sqrt(3.0);
	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			matrix_Rx[i][j]=temp*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 1, basis_function_SHF[i].m, basis_function_SHF[j].m, 1);
		}
	}

	temp=-1.0/sqrt(3.0);
	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			matrix_Ry[i][j]=temp*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 1, basis_function_SHF[i].m, basis_function_SHF[j].m, -1);
		}
	}

	temp=1.0/sqrt(3.0);
	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			matrix_Rz[i][j]=temp*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 1, basis_function_SHF[i].m, basis_function_SHF[j].m, 0);
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////THETAij for AB //////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	
	temp=1.0/sqrt(15.0);
	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			THETAij_M11_M22[i][j]=temp*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 2, basis_function_SHF[i].m, basis_function_SHF[j].m, 2);
		}
	}
	
	temp=1.0/sqrt(5.0);
	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			THETAij_M33[i][j]=temp*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 2, basis_function_SHF[i].m, basis_function_SHF[j].m, 0);
		}
	}
	
	temp=2.0/sqrt(15.0);
	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			THETAij_M12[i][j]=temp*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 2, basis_function_SHF[i].m, basis_function_SHF[j].m, -2);
		}
	}
	
	temp=-2.0/sqrt(15.0);
	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			THETAij_M13[i][j]=temp*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 2, basis_function_SHF[i].m, basis_function_SHF[j].m, 1);
		}
	}
	
	temp=-2.0/sqrt(15.0);
	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			THETAij_M23[i][j]=temp*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 2, basis_function_SHF[i].m, basis_function_SHF[j].m, -1);
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////


	/////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////J for AB///////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////

	double temp1, temp2;

	temp1= 1.0/sqrt(15.0);
	temp2=-1.0/(3.0*sqrt(5.0));
	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			J11ij[i][j]=temp1*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 2, basis_function_SHF[i].m, basis_function_SHF[j].m, 2)
				+ temp2*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 2, basis_function_SHF[i].m, basis_function_SHF[j].m, 0);
		}
	}

	temp1=-1.0/sqrt(15.0);
	temp2=-1.0/(3.0*sqrt(5.0));
	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			J22ij[i][j]=temp1*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 2, basis_function_SHF[i].m, basis_function_SHF[j].m, 2)
				+ temp2*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 2, basis_function_SHF[i].m, basis_function_SHF[j].m, 0);
		}
	}

	temp1= 1.0/sqrt(15.0);
	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			J12ij[i][j]=temp1*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 2, basis_function_SHF[i].m, basis_function_SHF[j].m, -2);
		}
	}

	temp1=-1.0/sqrt(15.0);
	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			J13ij[i][j]=temp1*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 2, basis_function_SHF[i].m, basis_function_SHF[j].m, 1);
		}
	}

	temp1=-1.0/sqrt(15.0);
	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			J23ij[i][j]=temp1*Triple_product(basis_function_SHF[i].l, 
				basis_function_SHF[j].l, 2, basis_function_SHF[i].m, basis_function_SHF[j].m, -1);
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

	int n_anderson;
	int K;
	
	for(n_anderson=0;n_anderson<=n_r_WAB;n_anderson++){
		for(K=0;K<LOCAL_SIZE;K++){
			WA_out[n_anderson][K]=0.0;
			WB_out[n_anderson][K]=0.0;
			dA_anderson[n_anderson][K]=0.0;
			dB_anderson[n_anderson][K]=0.0;
		}		
	}


	for(n_anderson=0;n_anderson<=n_r_M;n_anderson++){
		for(K=0;K<LOCAL_SIZE;K++){
			M11_out[n_anderson][K]=0.0;
			M33_out[n_anderson][K]=0.0;
			M12_out[n_anderson][K]=0.0;
			M13_out[n_anderson][K]=0.0;
			M23_out[n_anderson][K]=0.0;
			d11_anderson[n_anderson][K]=0.0;
			d33_anderson[n_anderson][K]=0.0;
			d12_anderson[n_anderson][K]=0.0;
			d13_anderson[n_anderson][K]=0.0;
			d23_anderson[n_anderson][K]=0.0;
		}
	}


	/////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////initialize THETA_nonzero_2D_A/////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	
	array_struct_int_int_double test1A;

	for(K=0;K<LOCAL_SIZE;K++){
	vector<array_struct_int_int_double> test_THETA_A;
		for(i=0;i<M_Bar;i++){
			for(j=0;j<M_Bar;j++){
				test1A.i=i;
				test1A.j=j;
				test1A.x=0.0;
				test_THETA_A.push_back(test1A);
			}
		}
		THETA_nonzero_2D.push_back(test_THETA_A);
	}


	/////////////////////////////////////////////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////input J??ij_nonzero_1D////////////////////////
	///////////////////////////////////////////////////////////////////////////////////

	array_struct_int_int_double test1;

	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			if (fabs(J11ij[i][j]) >= Thresh_sprase_matrix) 
			{
				test1.i=i;
				test1.j=j;
				test1.x=J11ij[i][j];
				J11ij_nonzero_1D.push_back(test1);
			}
		}
	}


	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			if (fabs(J22ij[i][j]) >= Thresh_sprase_matrix) 
			{
				test1.i=i;
				test1.j=j;
				test1.x=J22ij[i][j];
				J22ij_nonzero_1D.push_back(test1);
			}
		}
	}


	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			if (fabs(J12ij[i][j]) >= Thresh_sprase_matrix) 
			{
				test1.i=i;
				test1.j=j;
				test1.x=J12ij[i][j];
				J12ij_nonzero_1D.push_back(test1);
			}
		}
	}


	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			if (fabs(J13ij[i][j]) >= Thresh_sprase_matrix) 
			{
				test1.i=i;
				test1.j=j;
				test1.x=J13ij[i][j];
				J13ij_nonzero_1D.push_back(test1);
			}
		}
	}


	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			if (fabs(J23ij[i][j]) >= Thresh_sprase_matrix) 
			{
				test1.i=i;
				test1.j=j;
				test1.x=J23ij[i][j];
				J23ij_nonzero_1D.push_back(test1);
			}
		}
	}


	///////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////input GAMA_nonzero_1D   (ijk) ////////////////
	///////////////////////////////////////////////////////////////////////////////////

	array_struct_int_int_int_double test2;

	for(i=0;i<M_Bar;i++){
		for(j=0;j<M_Bar;j++){
			for(k=0;k<M_Bar;k++){
				if (fabs(GAMA[i][j][k]) >= Thresh_sprase_matrix)
				{
					test2.i=i;
					test2.j=j;
					test2.k=k;
					test2.x=GAMA[i][j][k];
					GAMA_nonzero_1D.push_back(test2);
				}
			}
		}
	}


	printf("GAMA_nonzero_1D.size()=%d\n",GAMA_nonzero_1D.size());


	///////////////////////////////////////////////////////////////////////////////////

	
}
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
void clean()
{
    free_dmatrix(matrix_Rx,0,M_Bar-1,0,M_Bar-1);
	free_dmatrix(matrix_Ry,0,M_Bar-1,0,M_Bar-1);
	free_dmatrix(matrix_Rz,0,M_Bar-1,0,M_Bar-1);
	free_f3tensor(GAMA,0,M_Bar-1,0,M_Bar-1,0,M_Bar-1);
	free_f3tensor(qA,0,NA,0,LOCAL_SIZE-1,0,M_Bar-1);
	free_f3tensor(qAstar,0,NA,0,LOCAL_SIZE-1,0,M_Bar-1);
	free_f3tensor(qB,0,NB,0,LOCAL_SIZE-1,0,M_Bar-1);
	free_f3tensor(qBstar,0,NB,0,LOCAL_SIZE-1,0,M_Bar-1);
	free_dvector(WA,0,LOCAL_SIZE-1);
	free_dvector(WB,0,LOCAL_SIZE-1);
	free_dmatrix(WA_out,0,n_r_WAB,0,LOCAL_SIZE-1);
	free_dmatrix(WB_out,0,n_r_WAB,0,LOCAL_SIZE-1);
	free_dmatrix(dA_anderson,0,n_r_WAB,0,LOCAL_SIZE-1);
	free_dmatrix(dB_anderson,0,n_r_WAB,0,LOCAL_SIZE-1);

	free_dmatrix(M11_out,0,n_r_M,0,LOCAL_SIZE-1);
	free_dmatrix(M33_out,0,n_r_M,0,LOCAL_SIZE-1);
	free_dmatrix(M12_out,0,n_r_M,0,LOCAL_SIZE-1);
	free_dmatrix(M13_out,0,n_r_M,0,LOCAL_SIZE-1);
	free_dmatrix(M23_out,0,n_r_M,0,LOCAL_SIZE-1);
	free_dmatrix(d11_anderson,0,n_r_M,0,LOCAL_SIZE-1);
	free_dmatrix(d33_anderson,0,n_r_M,0,LOCAL_SIZE-1);
	free_dmatrix(d12_anderson,0,n_r_M,0,LOCAL_SIZE-1);
	free_dmatrix(d13_anderson,0,n_r_M,0,LOCAL_SIZE-1);
	free_dmatrix(d23_anderson,0,n_r_M,0,LOCAL_SIZE-1);

	free_dvector(RA,0,LOCAL_SIZE-1);
	free_dvector(RB,0,LOCAL_SIZE-1);
	free_dvector(R_half,0,LOCAL_SIZE-1);
	free_dvector(R_end,0,LOCAL_SIZE-1);
	free_dmatrix(RHOA,0,LOCAL_SIZE-1,0,M_Bar-1);
	free_dmatrix(RHOB,0,LOCAL_SIZE-1,0,M_Bar-1);


	free_f3tensor(M_OP,0,LOCAL_SIZE-1,0,N_dim_ddm-1,0,N_dim_ddm-1);
	free_f3tensor(S_OP,0,LOCAL_SIZE-1,0,N_dim_ddm-1,0,N_dim_ddm-1);
	free_f3tensor(SA_OP,0,LOCAL_SIZE-1,0,N_dim_ddm-1,0,N_dim_ddm-1);
	free_f3tensor(SB_OP,0,LOCAL_SIZE-1,0,N_dim_ddm-1,0,N_dim_ddm-1);

	
	free_f3tensor(THETAij,0,LOCAL_SIZE-1,0,M_Bar-1,0,M_Bar-1);
	free_dmatrix(THETAij_M11_M22,0,M_Bar-1,0,M_Bar-1);
	free_dmatrix(THETAij_M33,0,M_Bar-1,0,M_Bar-1);
	free_dmatrix(THETAij_M12,0,M_Bar-1,0,M_Bar-1);
	free_dmatrix(THETAij_M13,0,M_Bar-1,0,M_Bar-1);
	free_dmatrix(THETAij_M23,0,M_Bar-1,0,M_Bar-1);
	
	free_dmatrix(J11ij,0,M_Bar-1,0,M_Bar-1);
	free_dmatrix(J22ij,0,M_Bar-1,0,M_Bar-1);
	free_dmatrix(J12ij,0,M_Bar-1,0,M_Bar-1);
	free_dmatrix(J13ij,0,M_Bar-1,0,M_Bar-1);
	free_dmatrix(J23ij,0,M_Bar-1,0,M_Bar-1);
	

	 free_f3tensor(sa_G_R_inv,0,7,1,LOCAL_SIZE,1,namax);
	 free_f3tensor_int(ija_G_R_inv,0,7,1,LOCAL_SIZE,1,namax);
	 free_f3tensor(sa_G_I_inv,0,7,1,LOCAL_SIZE,1,namax);
	 free_f3tensor_int(ija_G_I_inv,0,7,1,LOCAL_SIZE,1,namax);


}
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
void calc_THETAij()
{
	int i,j;
	int K;

	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////for A /////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////
	
	for(K=0;K<LOCAL_SIZE;K++){
		for(i=0;i<M_Bar;i++){
			for(j=0;j<M_Bar;j++){
				THETAij[K][i][j]=(M_OP[K][0][0]-M_OP[K][1][1])*THETAij_M11_M22[i][j] 
					+ M_OP[K][2][2]*THETAij_M33[i][j]
					+ M_OP[K][0][1]*THETAij_M12[i][j] 
					+ M_OP[K][0][2]*THETAij_M13[i][j] 
					+ M_OP[K][1][2]*THETAij_M23[i][j];
			}
		}
	}


	THETA_nonzero_2D.clear();   // remove the existing elements in "THETA_nonzero_2D"
	
	array_struct_int_int_double test1A;
	
	for(K=0;K<LOCAL_SIZE;K++){
		vector<array_struct_int_int_double> test_THETA_A;
		for(i=0;i<M_Bar;i++){
			for(j=0;j<M_Bar;j++){
				if (fabs(THETAij[K][i][j]) >= Thresh_sprase_matrix)
				{
					test1A.i=i;
					test1A.j=j;
					test1A.x=THETAij[K][i][j];
					test_THETA_A.push_back(test1A);
				}
			}
		}
		THETA_nonzero_2D.push_back(test_THETA_A);
	}


	
}
/////////////////////////////////////////////////////////////////////////////////////////////
void MDE_Pr_cal( int orient,int s, double *Wsp, double **Pr,double ***q_step) 
// orient : forward 1, backward -1
// s : time variable
// Wsp: WA or WB, depends on s and orient
// Pr: Pr_A or Pr_B
// q_step[3] :: forward : q0: q(s-1); q1: q(s-2); q2: q(s-3), backward: q0: q(s+1),q1: q(s+2); q2: q(s+3).

{
      int K,i,j,s_b;
      int index_nonzero;
      double q_coef1[3][3];
      double q_coef2[3][3];
      double temp1,temp2,Wtemp,tempx;  
// set up coefficient for s=1,2,s(s>=3)
         q_coef1[0][0]=1.0; 
         q_coef1[0][1]=0.0; 
         q_coef1[0][2]=0.0; 

         q_coef1[1][0]=2.0; 
         q_coef1[1][1]=-0.5; 
         q_coef1[1][2]=0.0; 

         q_coef1[2][0]=3.0; 
         q_coef1[2][1]=-1.5; 
         q_coef1[2][2]=1.0/3.0; 

         q_coef2[0][0]=1.0; 
         q_coef2[0][1]=0.0; 
         q_coef2[0][2]=0.0; 

         q_coef2[1][0]=2.0; 
         q_coef2[1][1]=-1.0; 
         q_coef2[1][2]=0.0; 

         q_coef2[2][0]=3.0; 
         q_coef2[2][1]=-3.0; 
         q_coef2[2][2]=1.0; 

       if(orient==1) {
             s_b = 3 < s ? 3 : s;}
       else {
             s_b = 3 < NMAX-s ? 3 : NMAX-s;}

       s_b=s_b-1;  

	for(K=0;K<LOCAL_SIZE;K++){
		for(i=0;i<M_Bar;i++){
			Pr[K][i]=0.0;
		}
	}
	for(K=0;K<LOCAL_SIZE;K++){
		Wtemp=Wsp[K];					
	     for(index_nonzero=0;index_nonzero<THETA_nonzero_2D[K].size();index_nonzero++){
                i=THETA_nonzero_2D[K][index_nonzero].i;
                j=THETA_nonzero_2D[K][index_nonzero].j;
                tempx=THETA_nonzero_2D[K][index_nonzero].x;
                Pr[K][i]=Pr[K][i] + ds*tempx*(q_coef2[s_b][0]*q_step[0][K][j]+q_coef2[s_b][1]*q_step[1][K][j]+q_coef2[s_b][2]*q_step[2][K][j]);
		}
		for(i=0;i<M_Bar;i++){
			Pr[K][i]=Pr[K][i] + q_coef1[s_b][0]*q_step[0][K][i]+q_coef1[s_b][1]*q_step[1][K][i] + q_coef1[s_b][2]*q_step[2][K][i] 
				- ds*Wtemp*(q_coef2[s_b][0]*q_step[0][K][i]+q_coef2[s_b][1]*q_step[1][K][i] + q_coef2[s_b][2]*q_step[2][K][i]);
		}
	}


}

 void MDE_Pk_cal(double **Pr, double ** Pk_Real, double **Pk_Imag)
{
      int K,i,j;

	for(i=0;i<M_Bar;i++){
		for(K=0;K<LOCAL_SIZE;K++){
			in[K][0]=Pr[K][i];
			in[K][1] = 0.0;
		}
		pfft_execute(plan_forw);

		for(K=0;K<LOCAL_SIZE;K++){
			Pk_Real[K][i]=in[K][0];
			Pk_Imag[K][i]=in[K][1];
		}
           }                  



}
//////////////////////////////////////////////////////////////////////////////
void MDE_get_previous_q(int orient, int s, double ***q_step)
{
// make sure q_step is initialized as zeros in MDE!!!

      int K,i,j,n;
      int s_step;

         if(orient==1) {
              s_step = 3 < s ? 3 : s;}
         else {
              s_step = 3 < NMAX-s ? 3 : NMAX-s;}

  if(orient==1) {
       if (s<=NA+1) {
            for (n=0;n<s_step;n++) {  
              for(K=0;K<LOCAL_SIZE;K++){
                 for(i=0;i<M_Bar;i++){
   	         q_step[n][K][i]=qA[s-1-n][K][i];
                   	} 
                  }
               }
             }
        else if (s==NA+2) {
              for(K=0;K<LOCAL_SIZE;K++){
                 for(i=0;i<M_Bar;i++){
   	         q_step[0][K][i]=qB[1][K][i];
   	         q_step[1][K][i]=qA[s-2][K][i];
   	         q_step[2][K][i]=qA[s-3][K][i];
                   	} 
                  }
               }
        else if (s==NA+3) {
              for(K=0;K<LOCAL_SIZE;K++){
                 for(i=0;i<M_Bar;i++){
   	         q_step[0][K][i]=qB[2][K][i];
   	         q_step[1][K][i]=qB[1][K][i];
   	         q_step[2][K][i]=qA[s-3][K][i];
                   	} 
                  }
                }  
        else  {
              for(K=0;K<LOCAL_SIZE;K++){
                 for(i=0;i<M_Bar;i++){
   	         q_step[0][K][i]=qB[s-1-NA][K][i];
   	         q_step[1][K][i]=qB[s-2-NA][K][i];
   	         q_step[2][K][i]=qB[s-3-NA][K][i];
                   	} 
                  }
             }
        } //end block for if orient
//////////////////////else if orient ==-1///////////////
   else {
       if (s>NA-1) {
            for (n=0;n<s_step;n++) {  
              for(K=0;K<LOCAL_SIZE;K++){
                 for(i=0;i<M_Bar;i++){
   	         q_step[n][K][i]=qBstar[s+1+n-NA][K][i];
                   	} 
                  }
               }
             }
        else if (s==NA-1) {
              for(K=0;K<LOCAL_SIZE;K++){
                 for(i=0;i<M_Bar;i++){
   	         q_step[0][K][i]=qAstar[s+1+0][K][i];
   	         q_step[1][K][i]=qBstar[s+1+1-NA][K][i];
   	         q_step[2][K][i]=qBstar[s+1+2-NA][K][i];
                   	} 
                  }
               }

        else if (s==NA-2) {
              for(K=0;K<LOCAL_SIZE;K++){
                 for(i=0;i<M_Bar;i++){
   	         q_step[0][K][i]=qAstar[s+1+0][K][i];
   	         q_step[1][K][i]=qAstar[s+1+1][K][i];
   	         q_step[2][K][i]=qBstar[s+1+2-NA][K][i];
                   	} 
                  }
                }  
        else  {
              for(K=0;K<LOCAL_SIZE;K++){
                 for(i=0;i<M_Bar;i++){
   	         q_step[0][K][i]=qAstar[s+1+0][K][i];
   	         q_step[1][K][i]=qAstar[s+1+1][K][i];
   	         q_step[2][K][i]=qAstar[s+1+2][K][i];
                   	} 
                  }
             }


     } // end block for orient
}

void MDE_sparse_matvec(int orient,int s, double **Pk_Real,double **Pk_Imag,double **q_temp_Real,double **q_temp_Imag)
{
// G_indx 0:GA1; 1:GA2; 2:GA3; 3:GB3; 4:GBstar1; 5:GBstar2;6:GB3star;7:GA3star
      int K,i,j,n;
      int s_step,G_sp,G_indx;

         if(orient==1) {
              s_step = 3 < s ? 3 : s;
              G_sp=s <= NA ? 0:1;
              G_indx=s_step+G_sp-1;} 
         else {
              s_step = 3 < NMAX-s ? 3 : NMAX-s;
              G_sp=s > NA ? 0:1;
              G_indx=s_step+G_sp-1+4;} 

	for(K=0;K<LOCAL_SIZE;K++){		
	compx_sparse_matrx_mul_vector(M_Bar, sa_G_R_inv[G_indx][K+1], ija_G_R_inv[G_indx][K+1], sa_G_I_inv[G_indx][K+1], ija_G_I_inv[G_indx][K+1],
			       Pk_Real[K], Pk_Imag[K],q_temp_Real[K],q_temp_Imag[K]);					
		}

}
////////////////////////////////////////////////////////////////
void MDE_get_real_q(int orient,int s, double **q_temp_Real,double **q_temp_Imag)
{
      int K,i;

        for(i=0;i<M_Bar;i++){
              for(K=0;K<LOCAL_SIZE;K++){
                      in[K][0]=q_temp_Real[K][i];
                      in[K][1]=q_temp_Imag[K][i];
              }

              pfft_execute(plan_back);


     if(orient==1) {
              if(s<=NA) {
              for(K=0;K<LOCAL_SIZE;K++){
                    qA[s][K][i]=in[K][0]/M_grid;
                                       }
                        }
               else {
              for(K=0;K<LOCAL_SIZE;K++){
                    qB[s-NA][K][i]=in[K][0]/M_grid;
                                       }
                    }
                  }     
     else {
              if(s>NA) {
              for(K=0;K<LOCAL_SIZE;K++){
                    qBstar[s-NA][K][i]=in[K][0]/M_grid;
                                       }
                        }
               else {
              for(K=0;K<LOCAL_SIZE;K++){
                    qAstar[s][K][i]=in[K][0]/M_grid;
                                       }
                    }

          } 

      } // end block for i

}   
/////////////////////////////////////////////////////////////////////////////////////////////
void MDE_q()
{
	int i,j,s;
	int index_nonzero;
	int K;
        int s_step;
	double tempA1, tempA2, tempA3;
	double temp;
	double tempx;
	double Wtemp;

	double **Pr=dmatrix(0,LOCAL_SIZE-1,0,M_Bar-1);// matrix P in the coordinate r
	double **Pk_Real=dmatrix(0,LOCAL_SIZE-1,0,M_Bar-1);
	double **Pk_Imag=dmatrix(0,LOCAL_SIZE-1,0,M_Bar-1);
	double **q_temp_Real=dmatrix(0,LOCAL_SIZE-1,0,M_Bar-1);
	double **q_temp_Imag=dmatrix(0,LOCAL_SIZE-1,0,M_Bar-1);
	
        double ***q_step; // previous propagators needed for BDF3
	////////////////////////////////////////////////////////
// forward mde ///////////////////////////////////////////////////////
        q_step=f3tensor(0,2,0,LOCAL_SIZE-1,0,M_Bar-1);
	for(K=0;K<LOCAL_SIZE;K++){
		for(i=0;i<M_Bar;i++){
			q_step[0][K][i]=0.0;
			q_step[1][K][i]=0.0;
			q_step[2][K][i]=0.0;
		}
	}
	for(K=0;K<LOCAL_SIZE;K++){
                qA[0][K][0]=1.0;
		for(i=1;i<M_Bar;i++){
			qA[0][K][i]=0.0;
		}
	}
	////////////////////////////////////////////////////////
    for(s=1;s<=NMAX;s++){

    ////////////////////// s<=NA /////////////////////////////
          MDE_get_previous_q(1,s,q_step);
          if (s<=NA) {
             MDE_Pr_cal( 1, s, WA, Pr, q_step);} 
          else {
             MDE_Pr_cal( 1, s, WB, Pr, q_step);}     

          MDE_Pk_cal(Pr,Pk_Real,Pk_Imag);
          MDE_sparse_matvec(1,s,Pk_Real,Pk_Imag,q_temp_Real,q_temp_Imag);
          MDE_get_real_q(1,s, q_temp_Real,q_temp_Imag);   

        }     // end block for s=0~NMAX

	////////////////////forward mde normalization /////////////////
	for(K=0;K<LOCAL_SIZE;K++){
		for(i=0;i<M_Bar;i++){
			qB[0][K][i]=0.0;
		}
	}
	for(K=0;K<LOCAL_SIZE;K++){
		for(i=0;i<M_Bar;i++){
			qB[0][K][i]=qA[NA][K][i];
		}
	}


// backward mde ///////////////////////////////////////////////////////
	for(K=0;K<LOCAL_SIZE;K++){
		for(i=0;i<M_Bar;i++){
			q_step[0][K][i]=0.0;
			q_step[1][K][i]=0.0;
			q_step[2][K][i]=0.0;
		}
	}
	for(K=0;K<LOCAL_SIZE;K++){
                qBstar[NB][K][0]=1.0;
		for(i=1;i<M_Bar;i++){
			qBstar[NB][K][i]=0.0;
		}
	}
	////////////////////////////////////////////////////////
    for(s=NMAX-1;s>=0;s--){

    ////////////////////// s>NA /////////////////////////////
          MDE_get_previous_q(-1,s,q_step);
          if (s>NA) {
             MDE_Pr_cal( -1, s, WB, Pr, q_step);} 
          else {
             MDE_Pr_cal( -1, s, WA, Pr, q_step);}     

          MDE_Pk_cal(Pr,Pk_Real,Pk_Imag);
          MDE_sparse_matvec(-1,s,Pk_Real,Pk_Imag,q_temp_Real,q_temp_Imag);
          MDE_get_real_q(-1,s, q_temp_Real,q_temp_Imag);   

        }     // end block for s=0~NMAX

	////////////////////backward mde normalization /////////////////
	for(K=0;K<LOCAL_SIZE;K++){
		for(i=0;i<M_Bar;i++){
			qBstar[0][K][i]=0.0;
		}
	}
	
	for(K=0;K<LOCAL_SIZE;K++){
		for(i=0;i<M_Bar;i++){
			qBstar[0][K][i]=qAstar[NA][K][i];
		}
	}


	free_dmatrix(Pr,0,LOCAL_SIZE-1,0,M_Bar-1);
	free_dmatrix(Pk_Real,0,LOCAL_SIZE-1,0,M_Bar-1);
	free_dmatrix(Pk_Imag,0,LOCAL_SIZE-1,0,M_Bar-1);
	free_dmatrix(q_temp_Real,0,LOCAL_SIZE-1,0,M_Bar-1);
	free_dmatrix(q_temp_Imag,0,LOCAL_SIZE-1,0,M_Bar-1);
        free_f3tensor(q_step,0,2,0,LOCAL_SIZE-1,0,M_Bar-1);

}
/////////////////////////////////////////////////////////////////////////////////////////////

/************************************************************************/
/////////////////////////////////////////////////////////////////////////
void density()
{
	int i,j,s;
	int K;
	double pff_temp_global;
	double *ar=dvector(0,LOCAL_SIZE-1);

	for(K=0;K<LOCAL_SIZE;K++){
		ar[K]=qB[NB][K][0];
	}

	double pff_temp=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, ar)/M_v;

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&pff_temp,&pff_temp_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);

	if (myid == 0){
		
		pff_global=log(pff_temp_global);
				
	}
	
	free_dvector(ar,0,LOCAL_SIZE-1);

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	
	double totden_global;
	double *c_A=dvector(0,NA);
	double *c_B=dvector(0,NB);
	double *denz=dvector(0,LOCAL_SIZE-1);


	/////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////

	for(K=0;K<LOCAL_SIZE;K++){
		double sum_iA=0.0;
		double sum_iB=0.0;
		for(i=0;i<M_Bar;i++){
			for(s=0;s<=NA;s++){
				c_A[s]=qA[s][K][i]*qAstar[s][K][i];				
			}
			RHOA[K][i]=simposon_1D_NR(0,NA,ds,c_A);
			sum_iA=sum_iA+RHOA[K][i];
			
		}
		
		for(i=0;i<M_Bar;i++){
			for(s=0;s<=NB;s++){
				c_B[s]=qB[s][K][i]*qBstar[s][K][i];
				
			}
			RHOB[K][i]=simposon_1D_NR(0,NB,ds,c_B);			
			sum_iB=sum_iB+RHOB[K][i];
		}
		denz[K]=sum_iA+sum_iB;
	}
	double totden=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, denz);
	

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&totden,&totden_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Bcast(&totden_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

	
	for(K=0;K<LOCAL_SIZE;K++){
		double sum_iA=0.0;
		double sum_iB=0.0;
		double sum_half=0.0;
		double sum_end_B=0.0;
		for(i=0;i<M_Bar;i++){
			RHOA[K][i]=M_v*RHOA[K][i]/totden_global;
			sum_iA=sum_iA+RHOA[K][i];
			sum_half=sum_half+qA[NA][K][i]*qAstar[NA][K][i];
		}
		for(i=0;i<M_Bar;i++){
			RHOB[K][i]=M_v*RHOB[K][i]/totden_global;
			sum_iB=sum_iB+RHOB[K][i];
			sum_end_B=sum_end_B+qB[NB][K][i]*qBstar[NB][K][i];
		}
		RA[K]=sum_iA;
		RB[K]=sum_iB;
		R_half[K]=M_v*sum_half/totden_global;
		R_end[K]=M_v*sum_end_B/totden_global;
	}

/////////////////////////////////////////////////////////////////////////////////
////////////////////////////// for S ////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

	double nor_coeff=M_v/totden_global;
	double *c11_A=dvector(0,NA);
	double *c22_A=dvector(0,NA);
	double *c12_A=dvector(0,NA);
	double *c13_A=dvector(0,NA);
	double *c23_A=dvector(0,NA);
	
	double *c11_B=dvector(0,NB);
	double *c22_B=dvector(0,NB);
	double *c12_B=dvector(0,NB);
	double *c13_B=dvector(0,NB);
	double *c23_B=dvector(0,NB);
	
	for(K=0;K<LOCAL_SIZE;K++){
		for(s=0;s<=NA;s++){
			c11_A[s]=Sum_sparse_2D(M_Bar, J11ij_nonzero_1D, qA[s][K], qAstar[s][K]);
			c22_A[s]=Sum_sparse_2D(M_Bar, J22ij_nonzero_1D, qA[s][K], qAstar[s][K]);
			c12_A[s]=Sum_sparse_2D(M_Bar, J12ij_nonzero_1D, qA[s][K], qAstar[s][K]);
			c13_A[s]=Sum_sparse_2D(M_Bar, J13ij_nonzero_1D, qA[s][K], qAstar[s][K]);
			c23_A[s]=Sum_sparse_2D(M_Bar, J23ij_nonzero_1D, qA[s][K], qAstar[s][K]);
		}
		for(s=0;s<=NB;s++){
			c11_B[s]=Sum_sparse_2D(M_Bar, J11ij_nonzero_1D, qB[s][K], qBstar[s][K]);
			c22_B[s]=Sum_sparse_2D(M_Bar, J22ij_nonzero_1D, qB[s][K], qBstar[s][K]);
			c12_B[s]=Sum_sparse_2D(M_Bar, J12ij_nonzero_1D, qB[s][K], qBstar[s][K]);
			c13_B[s]=Sum_sparse_2D(M_Bar, J13ij_nonzero_1D, qB[s][K], qBstar[s][K]);
			c23_B[s]=Sum_sparse_2D(M_Bar, J23ij_nonzero_1D, qB[s][K], qBstar[s][K]);
		}
		SA_OP[K][0][0]=simposon_1D_NR(0,NA,ds,c11_A)*nor_coeff;
		SA_OP[K][1][1]=simposon_1D_NR(0,NA,ds,c22_A)*nor_coeff;
		SA_OP[K][2][2]=-(SA_OP[K][0][0] + SA_OP[K][1][1]);
		SA_OP[K][0][1]=simposon_1D_NR(0,NA,ds,c12_A)*nor_coeff;
		SA_OP[K][0][2]=simposon_1D_NR(0,NA,ds,c13_A)*nor_coeff;
		SA_OP[K][1][2]=simposon_1D_NR(0,NA,ds,c23_A)*nor_coeff;
		SA_OP[K][1][0]=SA_OP[K][0][1];
		SA_OP[K][2][0]=SA_OP[K][0][2];
		SA_OP[K][2][1]=SA_OP[K][1][2];
		
		SB_OP[K][0][0]=simposon_1D_NR(0,NB,ds,c11_B)*nor_coeff;
		SB_OP[K][1][1]=simposon_1D_NR(0,NB,ds,c22_B)*nor_coeff;
		SB_OP[K][2][2]=-(SB_OP[K][0][0] + SB_OP[K][1][1]);
		SB_OP[K][0][1]=simposon_1D_NR(0,NB,ds,c12_B)*nor_coeff;
		SB_OP[K][0][2]=simposon_1D_NR(0,NB,ds,c13_B)*nor_coeff;
		SB_OP[K][1][2]=simposon_1D_NR(0,NB,ds,c23_B)*nor_coeff;
		SB_OP[K][1][0]=SB_OP[K][0][1];
		SB_OP[K][2][0]=SB_OP[K][0][2];
		SB_OP[K][2][1]=SB_OP[K][1][2];
		
		S_OP[K][0][0]=SA_OP[K][0][0]+SB_OP[K][0][0];
		S_OP[K][1][1]=SA_OP[K][1][1]+SB_OP[K][1][1];
		S_OP[K][2][2]=SA_OP[K][2][2]+SB_OP[K][2][2];
		S_OP[K][0][1]=SA_OP[K][0][1]+SB_OP[K][0][1];
		S_OP[K][0][2]=SA_OP[K][0][2]+SB_OP[K][0][2];
		S_OP[K][1][2]=SA_OP[K][1][2]+SB_OP[K][1][2];
		S_OP[K][1][0]=S_OP[K][0][1];
		S_OP[K][2][0]=S_OP[K][0][2];
		S_OP[K][2][1]=S_OP[K][1][2];
	}
	
	
	
	free_dvector(c_A,0,NA);
	free_dvector(c_B,0,NB);
	free_dvector(denz,0,LOCAL_SIZE-1);
	free_dvector(c11_A,0,NA);
	free_dvector(c22_A,0,NA);
	free_dvector(c12_A,0,NA);
	free_dvector(c13_A,0,NA);
	free_dvector(c23_A,0,NA);
	free_dvector(c11_B,0,NB);
	free_dvector(c22_B,0,NB);
	free_dvector(c12_B,0,NB);
	free_dvector(c13_B,0,NB);
	free_dvector(c23_B,0,NB);

}
/************************************************************************/
/////////////////////////////////////////////////////////////////////////
double free_energy(){
	int K;
	int j,k;
	double tempEE,tempEE_global;

	double *ar=dvector(0,LOCAL_SIZE-1);
	double **b=dmatrix(0,N_dim_ddm-1,0,N_dim_ddm-1);

	for(K=0;K<LOCAL_SIZE;K++){
		ar[K]=NXab*RA[K]*RB[K]-WA[K]*RA[K]-WB[K]*RB[K]+0.5*(WA[K]+WB[K])*(RA[K]+RB[K]-1.0);
		
		if (fabs(NMu-0.0)<1.0e-4) 
		{
			ar[K]=ar[K];
		}
		else
		{
			for(j=0;j<N_dim_ddm;j++)
				for(k=0;k<N_dim_ddm;k++)
				{
					b[j][k]=M_OP[K][j][k];
				}
				ar[K]=ar[K]+0.5*(double_dot_multi(b, b)/NMu);
		}
	}

	tempEE=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, ar);
	

	MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&tempEE,&tempEE_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);

	if (myid == 0)	FE_global=-pff_global+tempEE_global/M_v;
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&FE_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
	
	free_dvector(ar,0,LOCAL_SIZE-1);
	free_dmatrix(b,0,N_dim_ddm-1,0,N_dim_ddm-1);
	
	return FE_global;
}
/************************************************************************/
void iterate()  
{
	register int K;
	double temp1;
	double temp2;
	double pressure_coeff;
	double ta_diff=0.0;
	double tb_diff=0.0;
	
	int n_r_WAB_temp=(Num_iteration_step_WM - Num_step_simple_mixing_for_WAB < n_r_WAB)? (Num_iteration_step_WM - Num_step_simple_mixing_for_WAB):n_r_WAB; // get the smaller one

	if (Num_iteration_step_WM<Num_step_simple_mixing_for_WAB){

		for(K=0;K<LOCAL_SIZE;K++){
			pressure_coeff=0.5*(WA[K]+WB[K]);
			temp1=NXab*(RB[K]-fb)+pressure_coeff - WA[K];
			temp2=NXab*(RA[K]-fa)+pressure_coeff - WB[K];
			WA[K]=WA[K]+lanbtWA*(temp1);
			WB[K]=WB[K]+lanbtWB*(temp2);

			if(fabs(temp1)>ta_diff) ta_diff=fabs(temp1);
	        if(fabs(temp2)>tb_diff) tb_diff=fabs(temp2);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&ta_diff,&ta_diff_global,1,MPI_DOUBLE, MPI_MAX,0,MPI_COMM_WORLD);
		MPI_Reduce(&tb_diff,&tb_diff_global,1,MPI_DOUBLE, MPI_MAX,0,MPI_COMM_WORLD);
		MPI_Bcast(&ta_diff_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(&tb_diff_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

	}

	else if (Num_iteration_step_WM == Num_step_simple_mixing_for_WAB){
			for(K=0;K<LOCAL_SIZE;K++){
				WA_out[0][K]=NXab*RB[K]+0.5*(WA[K]+WB[K]-NXab);
				WB_out[0][K]=NXab*RA[K]+0.5*(WA[K]+WB[K]-NXab);
				dA_anderson[0][K]=WA_out[0][K]-WA[K];
				dB_anderson[0][K]=WB_out[0][K]-WB[K];

				pressure_coeff=0.5*(WA[K]+WB[K]);
				temp1=NXab*(RB[K]-fb)+pressure_coeff - WA[K];
				temp2=NXab*(RA[K]-fa)+pressure_coeff - WB[K];
				WA[K]=WA[K]+lanbtWA*(temp1);
			    WB[K]=WB[K]+lanbtWB*(temp2);

				if(fabs(temp1)>ta_diff) ta_diff=fabs(temp1);
	            if(fabs(temp2)>tb_diff) tb_diff=fabs(temp2);
			}

			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Reduce(&ta_diff,&ta_diff_global,1,MPI_DOUBLE, MPI_MAX,0,MPI_COMM_WORLD);
			MPI_Reduce(&tb_diff,&tb_diff_global,1,MPI_DOUBLE, MPI_MAX,0,MPI_COMM_WORLD);
			MPI_Bcast(&ta_diff_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		    MPI_Bcast(&tb_diff_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		}

	else {
		double *C_WAB_anderson=dvector_NR(1,n_r_WAB_temp);
		double *V_WAB_anderson=dvector_NR(1,n_r_WAB_temp);
		double **U_WAB_anderson=dmatrix_NR(1,n_r_WAB_temp,1,n_r_WAB_temp);
		double **inverse_U_WAB_anderson=dmatrix_NR(1,n_r_WAB_temp,1,n_r_WAB_temp);
		double *g=dvector(0,LOCAL_SIZE-1);
		double *h=dvector(0,LOCAL_SIZE-1);
		double *gh=dvector(0,LOCAL_SIZE-1);
		double U_temp1, V_temp1, U_temp1_global, V_temp1_global;
		double U_temp2, V_temp2, U_temp2_global, V_temp2_global;

		double WA_temp, WB_temp, DA_temp, DB_temp;
		int m_anderson, n_anderson;
		int k_anderson=(Num_iteration_step_WM - Num_step_simple_mixing_for_WAB)%(n_r_WAB+1);

		for(K=0;K<LOCAL_SIZE;K++){
			WA_out[k_anderson][K]=NXab*RB[K]+0.5*(WA[K]+WB[K]-NXab);
			WB_out[k_anderson][K]=NXab*RA[K]+0.5*(WA[K]+WB[K]-NXab);
			dA_anderson[k_anderson][K]=WA_out[k_anderson][K]-WA[K];
			dB_anderson[k_anderson][K]=WB_out[k_anderson][K]-WB[K];
		}

		for(m_anderson=1;m_anderson<=n_r_WAB_temp;m_anderson++){
			for(n_anderson=m_anderson;n_anderson<=n_r_WAB_temp;n_anderson++){

				for(K=0;K<LOCAL_SIZE;K++){
					g[K]= dA_anderson[k_anderson][K] - dA_anderson[index_updated_matrix_anderson(k_anderson-m_anderson,n_r_WAB)][K];
					h[K]= dA_anderson[k_anderson][K] - dA_anderson[index_updated_matrix_anderson(k_anderson-n_anderson,n_r_WAB)][K];
					gh[K]=g[K]*h[K];
				}
               
				U_temp1=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
				

				for(K=0;K<LOCAL_SIZE;K++){
					g[K]= dB_anderson[k_anderson][K] - dB_anderson[index_updated_matrix_anderson(k_anderson-m_anderson,n_r_WAB)][K];
					h[K]= dB_anderson[k_anderson][K] - dB_anderson[index_updated_matrix_anderson(k_anderson-n_anderson,n_r_WAB)][K];
					gh[K]=g[K]*h[K];
				}
           
				U_temp2=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Reduce(&U_temp1,&U_temp1_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
				MPI_Reduce(&U_temp2,&U_temp2_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
	            MPI_Bcast(&U_temp1_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	            MPI_Bcast(&U_temp2_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
				U_WAB_anderson[m_anderson][n_anderson]=U_temp1_global+U_temp2_global;
			}

			for(K=0;K<LOCAL_SIZE;K++){
				g[K]= dA_anderson[k_anderson][K] - dA_anderson[index_updated_matrix_anderson(k_anderson-m_anderson,n_r_WAB)][K];
				h[K]= dA_anderson[k_anderson][K];
				gh[K]=g[K]*h[K];
			}
		
			V_temp1=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

			for(K=0;K<LOCAL_SIZE;K++){
				g[K]= dB_anderson[k_anderson][K] - dB_anderson[index_updated_matrix_anderson(k_anderson-m_anderson,n_r_WAB)][K];
				h[K]= dB_anderson[k_anderson][K];
				gh[K]=g[K]*h[K];
			}

			V_temp2=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Reduce(&V_temp1,&V_temp1_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
			MPI_Reduce(&V_temp2,&V_temp2_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
			MPI_Bcast(&V_temp1_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	        MPI_Bcast(&V_temp2_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
			
			V_WAB_anderson[m_anderson]=V_temp1_global+V_temp2_global;
		}


		for(m_anderson=2;m_anderson<=n_r_WAB_temp;m_anderson++){
			for(n_anderson=1;n_anderson<m_anderson;n_anderson++){
				U_WAB_anderson[m_anderson][n_anderson]=U_WAB_anderson[n_anderson][m_anderson];
			}
		}

		inverse_matrix_NR(U_WAB_anderson, inverse_U_WAB_anderson, n_r_WAB_temp);

		multi_matrix2_NR(inverse_U_WAB_anderson, V_WAB_anderson, C_WAB_anderson, n_r_WAB_temp);

		
		for(K=0;K<LOCAL_SIZE;K++){
			WA_temp=WA_out[k_anderson][K];
			WB_temp=WB_out[k_anderson][K];
			DA_temp=dA_anderson[k_anderson][K];
			DB_temp=dB_anderson[k_anderson][K];
			for(n_anderson=1;n_anderson<=n_r_WAB_temp;n_anderson++){
				WA_temp= WA_temp + C_WAB_anderson[n_anderson]*(WA_out[index_updated_matrix_anderson(k_anderson-n_anderson,n_r_WAB)][K] - WA_out[k_anderson][K]);
				WB_temp= WB_temp + C_WAB_anderson[n_anderson]*(WB_out[index_updated_matrix_anderson(k_anderson-n_anderson,n_r_WAB)][K] - WB_out[k_anderson][K]);
				DA_temp= DA_temp + C_WAB_anderson[n_anderson]*(dA_anderson[index_updated_matrix_anderson(k_anderson-n_anderson,n_r_WAB)][K] - dA_anderson[k_anderson][K]);
				DB_temp= DB_temp + C_WAB_anderson[n_anderson]*(dB_anderson[index_updated_matrix_anderson(k_anderson-n_anderson,n_r_WAB)][K] - dB_anderson[k_anderson][K]);
			}

			//lambda_WAB_anderson=1.0 - pow(0.9, double(k_anderson));
			lambda_WAB_anderson=lambda_WAB_anderson_const;
			WA[K]=WA_temp+lambda_WAB_anderson*DA_temp;
			WB[K]=WB_temp+lambda_WAB_anderson*DB_temp;
		}
//////////////// convergency criterion ///////
	////////	For A   /////////
		double tempd, tempw;
		double tempd_global, tempw_global;

		for(K=0;K<LOCAL_SIZE;K++){
			g[K]= dA_anderson[k_anderson][K];
			h[K]= dA_anderson[k_anderson][K];
			gh[K]=g[K]*h[K];
		}
		
		tempd=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

		for(K=0;K<LOCAL_SIZE;K++){
			g[K]= WA_out[k_anderson][K];
			h[K]= WA_out[k_anderson][K];
			gh[K]=g[K]*h[K];
		}
		
		tempw=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&tempd,&tempd_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&tempw,&tempw_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Bcast(&tempd_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(&tempw_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		
		ta_diff_global=sqrt(fabs(tempd_global/tempw_global));

	
	////////	For B   /////////
		for(K=0;K<LOCAL_SIZE;K++){
			g[K]= dB_anderson[k_anderson][K];
			h[K]= dB_anderson[k_anderson][K];
			gh[K]=g[K]*h[K];
		}
		
		tempd=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
		
		for(K=0;K<LOCAL_SIZE;K++){
			g[K]= WB_out[k_anderson][K];
			h[K]= WB_out[k_anderson][K];
			gh[K]=g[K]*h[K];
		}
		
		tempw=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&tempd,&tempd_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&tempw,&tempw_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Bcast(&tempd_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(&tempw_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		
		
		tb_diff_global=sqrt(fabs(tempd_global/tempw_global));
		

		free_dvector_NR(C_WAB_anderson,1,n_r_WAB_temp);
		free_dvector_NR(V_WAB_anderson,1,n_r_WAB_temp);
		free_dmatrix_NR(U_WAB_anderson,1,n_r_WAB_temp,1,n_r_WAB_temp);
		free_dmatrix_NR(inverse_U_WAB_anderson,1,n_r_WAB_temp,1,n_r_WAB_temp);
		free_dvector(g,0,LOCAL_SIZE-1);
		free_dvector(h,0,LOCAL_SIZE-1);
		free_dvector(gh,0,LOCAL_SIZE-1);

	}


	////////////////////////////////////////////////////////////////
	///////////////////  for M iteration //////////////////////////
	///////////////////////////////////////////////////////////////

	int n_r_M_temp;
	int j,k;
	double tm_diff=0.0;
	double **temp3m=dmatrix(0,N_dim_ddm-1,0,N_dim_ddm-1);
	
	if (fabs(NMu-0.0)<1.0e-4){
		for(K=0;K<LOCAL_SIZE;K++)
			for(j=0;j<N_dim_ddm;j++)
				for(k=0;k<N_dim_ddm;k++)
					M_OP[K][j][k]=0.0;
	}
	else {

		if (Num_iteration_step_WM < Num_step_simple_mixing_for_M){

			tm_diff=0.0;
			for(K=0;K<LOCAL_SIZE;K++){
				for(j=0;j<N_dim_ddm;j++){
					for(k=0;k<N_dim_ddm;k++){
						temp3m[j][k]=(S_OP[K][j][k] - M_OP[K][j][k]/NMu);
						M_OP[K][j][k]=M_OP[K][j][k]+lanbtM*temp3m[j][k];
						if (fabs(temp3m[j][k])>tm_diff) tm_diff=fabs(temp3m[j][k]);
					}
				}
				
			}
			
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Reduce(&tm_diff,&tm_diff_global,1,MPI_DOUBLE, MPI_MAX,0,MPI_COMM_WORLD);
		    MPI_Bcast(&tm_diff_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		}
		else if (Num_iteration_step_WM == Num_step_simple_mixing_for_M){
			
			Num_iteration_step_M++;
			
			if (method=='A'){ //uniaxial phase: Smectic A
				for(K=0;K<LOCAL_SIZE;K++){
					M33_out[0][K]=S_OP[K][2][2]*NMu;
					d33_anderson[0][K]=M33_out[0][K] - M_OP[K][2][2];
				}
			}
			else if (method=='B') {//uniaxial phase: Smectic C
				for(K=0;K<LOCAL_SIZE;K++){
					M33_out[0][K]=S_OP[K][2][2]*NMu;
					M12_out[0][K]=S_OP[K][0][1]*NMu;
					M13_out[0][K]=S_OP[K][0][2]*NMu;
					M23_out[0][K]=S_OP[K][1][2]*NMu;
					d33_anderson[0][K]=M33_out[0][K] - M_OP[K][2][2];
					d12_anderson[0][K]=M12_out[0][K] - M_OP[K][0][1];
					d13_anderson[0][K]=M13_out[0][K] - M_OP[K][0][2];
					d23_anderson[0][K]=M23_out[0][K] - M_OP[K][1][2];
				}
			}
			else {// biaxial phase
				for(K=0;K<LOCAL_SIZE;K++){
					M11_out[0][K]=S_OP[K][0][0]*NMu;
					M33_out[0][K]=S_OP[K][2][2]*NMu;
					M12_out[0][K]=S_OP[K][0][1]*NMu;
					M13_out[0][K]=S_OP[K][0][2]*NMu;
					M23_out[0][K]=S_OP[K][1][2]*NMu;
					d11_anderson[0][K]=M11_out[0][K] - M_OP[K][0][0];
					d33_anderson[0][K]=M33_out[0][K] - M_OP[K][2][2];
					d12_anderson[0][K]=M12_out[0][K] - M_OP[K][0][1];
					d13_anderson[0][K]=M13_out[0][K] - M_OP[K][0][2];
					d23_anderson[0][K]=M23_out[0][K] - M_OP[K][1][2];
				}
			}
			
			tm_diff=0.0;
			for(K=0;K<LOCAL_SIZE;K++){
				for(j=0;j<N_dim_ddm;j++){
					for(k=0;k<N_dim_ddm;k++){
						temp3m[j][k]=(S_OP[K][j][k] - M_OP[K][j][k]/NMu);
						M_OP[K][j][k]=M_OP[K][j][k]+lanbtM*temp3m[j][k];
						if (fabs(temp3m[j][k])>tm_diff) tm_diff=fabs(temp3m[j][k]);
					}
				}
				
			}
			
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Reduce(&tm_diff,&tm_diff_global,1,MPI_DOUBLE, MPI_MAX,0,MPI_COMM_WORLD);
		    MPI_Bcast(&tm_diff_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

		}

		else {
			
			if ((Num_iteration_step_WM - Num_step_simple_mixing_for_M)%Num_step_iterae_M==0){
				Num_iteration_step_M++;

				n_r_M_temp=(Num_iteration_step_M - 1 < n_r_M)? (Num_iteration_step_M - 1):n_r_M; // get the smaller one

				double *C_M_anderson=dvector_NR(1,n_r_M_temp);
				double *V_M_anderson=dvector_NR(1,n_r_M_temp);
				double **U_M_anderson=dmatrix_NR(1,n_r_M_temp,1,n_r_M_temp);
				double **inverse_U_M_anderson=dmatrix_NR(1,n_r_M_temp,1,n_r_M_temp);
				double *g=dvector(0,LOCAL_SIZE-1);
				double *h=dvector(0,LOCAL_SIZE-1);
				double *gh=dvector(0,LOCAL_SIZE-1);

				double UM_temp1, VM_temp1, UM_temp1_global, VM_temp1_global;
		        double UM_temp2, VM_temp2, UM_temp2_global, VM_temp2_global;
				double UM_temp3, VM_temp3, UM_temp3_global, VM_temp3_global;
				double UM_temp4, VM_temp4, UM_temp4_global, VM_temp4_global;
				double UM_temp5, VM_temp5, UM_temp5_global, VM_temp5_global;

				int mM_anderson, nM_anderson;
				int kM_anderson=(Num_iteration_step_M - 1)%(n_r_M+1);
				double M11_temp, M33_temp, d11_temp, d33_temp;
			    double M12_temp, M13_temp, M23_temp, d12_temp, d13_temp, d23_temp;	

				if (method=='A'){ //uniaxial phase: Smectic A
					for(K=0;K<LOCAL_SIZE;K++){
						M33_out[kM_anderson][K]=S_OP[K][2][2]*NMu;
						d33_anderson[kM_anderson][K]=M33_out[kM_anderson][K] - M_OP[K][2][2];
					}

				}
				else if (method=='B') {//uniaxial phase: Smectic C
					for(K=0;K<LOCAL_SIZE;K++){
						M33_out[kM_anderson][K]=S_OP[K][2][2]*NMu;
						M12_out[kM_anderson][K]=S_OP[K][0][1]*NMu;
						M13_out[kM_anderson][K]=S_OP[K][0][2]*NMu;
						M23_out[kM_anderson][K]=S_OP[K][1][2]*NMu;
						d33_anderson[kM_anderson][K]=M33_out[kM_anderson][K] - M_OP[K][2][2];
						d12_anderson[kM_anderson][K]=M12_out[kM_anderson][K] - M_OP[K][0][1];
						d13_anderson[kM_anderson][K]=M13_out[kM_anderson][K] - M_OP[K][0][2];
						d23_anderson[kM_anderson][K]=M23_out[kM_anderson][K] - M_OP[K][1][2];
					}
				}
				else {//biaxial phase
					for(K=0;K<LOCAL_SIZE;K++){
						M11_out[kM_anderson][K]=S_OP[K][0][0]*NMu;
						M33_out[kM_anderson][K]=S_OP[K][2][2]*NMu;
						M12_out[kM_anderson][K]=S_OP[K][0][1]*NMu;
						M13_out[kM_anderson][K]=S_OP[K][0][2]*NMu;
						M23_out[kM_anderson][K]=S_OP[K][1][2]*NMu;
						d11_anderson[kM_anderson][K]=M11_out[kM_anderson][K] - M_OP[K][0][0];
						d33_anderson[kM_anderson][K]=M33_out[kM_anderson][K] - M_OP[K][2][2];
						d12_anderson[kM_anderson][K]=M12_out[kM_anderson][K] - M_OP[K][0][1];
						d13_anderson[kM_anderson][K]=M13_out[kM_anderson][K] - M_OP[K][0][2];
						d23_anderson[kM_anderson][K]=M23_out[kM_anderson][K] - M_OP[K][1][2];
					}
				}

				for(mM_anderson=1;mM_anderson<=n_r_M_temp;mM_anderson++){
					for(nM_anderson=mM_anderson;nM_anderson<=n_r_M_temp;nM_anderson++){

						if (method=='A'){ //uniaxial phase: Smectic A
							for(K=0;K<LOCAL_SIZE;K++){
								g[K]= d33_anderson[kM_anderson][K] - d33_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
								h[K]= d33_anderson[kM_anderson][K] - d33_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K];
								gh[K]=g[K]*h[K];
							}

							UM_temp1=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;


							MPI_Barrier(MPI_COMM_WORLD);
							MPI_Reduce(&UM_temp1,&UM_temp1_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
							MPI_Bcast(&UM_temp1_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

							U_M_anderson[mM_anderson][nM_anderson]=UM_temp1_global;
							
						}
						else if (method=='B') {//uniaxial phase: Smectic C
							for(K=0;K<LOCAL_SIZE;K++){
								g[K]= d33_anderson[kM_anderson][K] - d33_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
								h[K]= d33_anderson[kM_anderson][K] - d33_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K];
								gh[K]=g[K]*h[K];
							}
							UM_temp1=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

							for(K=0;K<LOCAL_SIZE;K++){
								g[K]= d12_anderson[kM_anderson][K] - d12_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
								h[K]= d12_anderson[kM_anderson][K] - d12_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K];
								gh[K]=g[K]*h[K];
							}
							UM_temp2=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

							for(K=0;K<LOCAL_SIZE;K++){
								g[K]= d13_anderson[kM_anderson][K] - d13_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
								h[K]= d13_anderson[kM_anderson][K] - d13_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K];
								gh[K]=g[K]*h[K];
							}
							UM_temp3=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

							for(K=0;K<LOCAL_SIZE;K++){
								g[K]= d23_anderson[kM_anderson][K] - d23_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
								h[K]= d23_anderson[kM_anderson][K] - d23_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K];
								gh[K]=g[K]*h[K];
							}
							UM_temp4=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

							MPI_Barrier(MPI_COMM_WORLD);
							MPI_Reduce(&UM_temp1,&UM_temp1_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
							MPI_Reduce(&UM_temp2,&UM_temp2_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
							MPI_Reduce(&UM_temp3,&UM_temp3_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
							MPI_Reduce(&UM_temp4,&UM_temp4_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
							
							MPI_Bcast(&UM_temp1_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
							MPI_Bcast(&UM_temp2_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
							MPI_Bcast(&UM_temp3_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
							MPI_Bcast(&UM_temp4_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
							

							U_M_anderson[mM_anderson][nM_anderson]=UM_temp1_global + UM_temp2_global
								                                  +UM_temp3_global + UM_temp4_global;

						}
						else {//biaxial phase
							for(K=0;K<LOCAL_SIZE;K++){
								g[K]= d11_anderson[kM_anderson][K] - d11_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
								h[K]= d11_anderson[kM_anderson][K] - d11_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K];
								gh[K]=g[K]*h[K];
							}
							UM_temp5=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

							for(K=0;K<LOCAL_SIZE;K++){
								g[K]= d33_anderson[kM_anderson][K] - d33_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
								h[K]= d33_anderson[kM_anderson][K] - d33_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K];
								gh[K]=g[K]*h[K];
							}
							UM_temp1=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
							
							for(K=0;K<LOCAL_SIZE;K++){
								g[K]= d12_anderson[kM_anderson][K] - d12_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
								h[K]= d12_anderson[kM_anderson][K] - d12_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K];
								gh[K]=g[K]*h[K];
							}
							UM_temp2=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
							
							for(K=0;K<LOCAL_SIZE;K++){
								g[K]= d13_anderson[kM_anderson][K] - d13_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
								h[K]= d13_anderson[kM_anderson][K] - d13_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K];
								gh[K]=g[K]*h[K];
							}
							UM_temp3=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
							
							for(K=0;K<LOCAL_SIZE;K++){
								g[K]= d23_anderson[kM_anderson][K] - d23_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
								h[K]= d23_anderson[kM_anderson][K] - d23_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K];
								gh[K]=g[K]*h[K];
							}
							UM_temp4=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
							
							MPI_Barrier(MPI_COMM_WORLD);
							MPI_Reduce(&UM_temp1,&UM_temp1_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
							MPI_Reduce(&UM_temp2,&UM_temp2_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
							MPI_Reduce(&UM_temp3,&UM_temp3_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
							MPI_Reduce(&UM_temp4,&UM_temp4_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
							MPI_Reduce(&UM_temp5,&UM_temp5_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
							
							MPI_Bcast(&UM_temp1_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
							MPI_Bcast(&UM_temp2_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
							MPI_Bcast(&UM_temp3_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
							MPI_Bcast(&UM_temp4_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
							MPI_Bcast(&UM_temp5_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
							
							
							U_M_anderson[mM_anderson][nM_anderson]=UM_temp1_global + UM_temp2_global
								                                  +UM_temp3_global + UM_temp4_global
																  +UM_temp5_global;
						}

						}


					if (method=='A'){//uniaxial phase: Smectic A
						for(K=0;K<LOCAL_SIZE;K++){
							g[K]= d33_anderson[kM_anderson][K] - d33_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
							h[K]= d33_anderson[kM_anderson][K];
							gh[K]=g[K]*h[K];
						}						
						VM_temp1=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

						MPI_Barrier(MPI_COMM_WORLD);
						MPI_Reduce(&VM_temp1,&VM_temp1_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
						MPI_Bcast(&VM_temp1_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

						V_M_anderson[mM_anderson]=VM_temp1_global;
					    
					}
					else if (method=='B') {//uniaxial phase: Smectic C
						for(K=0;K<LOCAL_SIZE;K++){
							g[K]= d33_anderson[kM_anderson][K] - d33_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
							h[K]= d33_anderson[kM_anderson][K];
							gh[K]=g[K]*h[K];
						}
						VM_temp1=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					    
						
						for(K=0;K<LOCAL_SIZE;K++){
							g[K]= d12_anderson[kM_anderson][K] - d12_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
							h[K]= d12_anderson[kM_anderson][K];
							gh[K]=g[K]*h[K];
						}
						VM_temp2=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
						
						for(K=0;K<LOCAL_SIZE;K++){
							g[K]= d13_anderson[kM_anderson][K] - d13_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
							h[K]= d13_anderson[kM_anderson][K];
							gh[K]=g[K]*h[K];
						}
						VM_temp3=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
						
						for(K=0;K<LOCAL_SIZE;K++){
							g[K]= d23_anderson[kM_anderson][K] - d23_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
							h[K]= d23_anderson[kM_anderson][K];
							gh[K]=g[K]*h[K];
						}
						VM_temp4=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

						MPI_Barrier(MPI_COMM_WORLD);
						MPI_Reduce(&VM_temp1,&VM_temp1_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
						MPI_Reduce(&VM_temp2,&VM_temp2_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
						MPI_Reduce(&VM_temp3,&VM_temp3_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
						MPI_Reduce(&VM_temp4,&VM_temp4_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
						
						MPI_Bcast(&VM_temp1_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
						MPI_Bcast(&VM_temp2_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
						MPI_Bcast(&VM_temp3_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
						MPI_Bcast(&VM_temp4_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

						V_M_anderson[mM_anderson]=VM_temp1_global + VM_temp2_global
							                     +VM_temp3_global + VM_temp4_global;
					}
					else {//biaxial phase
						for(K=0;K<LOCAL_SIZE;K++){
							g[K]= d11_anderson[kM_anderson][K] - d11_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
							h[K]= d11_anderson[kM_anderson][K];
							gh[K]=g[K]*h[K];
						}
						VM_temp5=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

						for(K=0;K<LOCAL_SIZE;K++){
							g[K]= d33_anderson[kM_anderson][K] - d33_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
							h[K]= d33_anderson[kM_anderson][K];
							gh[K]=g[K]*h[K];
						}
						VM_temp1=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
						
						
						for(K=0;K<LOCAL_SIZE;K++){
							g[K]= d12_anderson[kM_anderson][K] - d12_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
							h[K]= d12_anderson[kM_anderson][K];
							gh[K]=g[K]*h[K];
						}
						VM_temp2=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
						
						for(K=0;K<LOCAL_SIZE;K++){
							g[K]= d13_anderson[kM_anderson][K] - d13_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
							h[K]= d13_anderson[kM_anderson][K];
							gh[K]=g[K]*h[K];
						}
						VM_temp3=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
						
						for(K=0;K<LOCAL_SIZE;K++){
							g[K]= d23_anderson[kM_anderson][K] - d23_anderson[index_updated_matrix_anderson(kM_anderson-mM_anderson,n_r_M)][K];
							h[K]= d23_anderson[kM_anderson][K];
							gh[K]=g[K]*h[K];
						}
						VM_temp4=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
						
						MPI_Barrier(MPI_COMM_WORLD);
						MPI_Reduce(&VM_temp1,&VM_temp1_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
						MPI_Reduce(&VM_temp2,&VM_temp2_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
						MPI_Reduce(&VM_temp3,&VM_temp3_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
						MPI_Reduce(&VM_temp4,&VM_temp4_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
						MPI_Reduce(&VM_temp5,&VM_temp5_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
						
						MPI_Bcast(&VM_temp1_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
						MPI_Bcast(&VM_temp2_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
						MPI_Bcast(&VM_temp3_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
						MPI_Bcast(&VM_temp4_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
						MPI_Bcast(&VM_temp5_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
						
						V_M_anderson[mM_anderson]=VM_temp1_global + VM_temp2_global
							                     +VM_temp3_global + VM_temp4_global
												 +VM_temp5_global;
					}
				}

				for(mM_anderson=2;mM_anderson<=n_r_M_temp;mM_anderson++){
					for(nM_anderson=1;nM_anderson<mM_anderson;nM_anderson++){
						U_M_anderson[mM_anderson][nM_anderson]=U_M_anderson[nM_anderson][mM_anderson];
					}
				}

				inverse_matrix_NR(U_M_anderson, inverse_U_M_anderson, n_r_M_temp);
				
		        multi_matrix2_NR(inverse_U_M_anderson, V_M_anderson, C_M_anderson, n_r_M_temp);

				if (method=='A') {//uniaxial phase: Smectic A
					for(K=0;K<LOCAL_SIZE;K++){
						M33_temp=M33_out[kM_anderson][K];
						d33_temp=d33_anderson[kM_anderson][K];

						for(nM_anderson=1;nM_anderson<=n_r_M_temp;nM_anderson++){
							M33_temp= M33_temp + C_M_anderson[nM_anderson]*(M33_out[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - M33_out[kM_anderson][K]);
							d33_temp= d33_temp + C_M_anderson[nM_anderson]*(d33_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - d33_anderson[kM_anderson][K]);
						}

						//lambda_M_anderson=1.0 - pow(0.9, double(kM_anderson));
						lambda_M_anderson=lambda_M_anderson_const;
						M_OP[K][2][2]=M33_temp+lambda_M_anderson*d33_temp;
						M_OP[K][0][0]=-0.5*M_OP[K][2][2];
						M_OP[K][1][1]=M_OP[K][0][0];
						M_OP[K][1][0]=M_OP[K][0][1]=0.0;
						M_OP[K][2][0]=M_OP[K][0][2]=0.0;
						M_OP[K][2][1]=M_OP[K][1][2]=0.0;
					}
				}
				else if (method=='B') {//uniaxial phase: Smectic C
					for(K=0;K<LOCAL_SIZE;K++){
						M33_temp=M33_out[kM_anderson][K];
						M12_temp=M12_out[kM_anderson][K];
						M13_temp=M13_out[kM_anderson][K];
						M23_temp=M23_out[kM_anderson][K];
						d33_temp=d33_anderson[kM_anderson][K];
						d12_temp=d12_anderson[kM_anderson][K];
						d13_temp=d13_anderson[kM_anderson][K];
						d23_temp=d23_anderson[kM_anderson][K];

						for(nM_anderson=1;nM_anderson<=n_r_M_temp;nM_anderson++){
							M33_temp= M33_temp + C_M_anderson[nM_anderson]*(M33_out[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - M33_out[kM_anderson][K]);
							M12_temp= M12_temp + C_M_anderson[nM_anderson]*(M12_out[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - M12_out[kM_anderson][K]);
							M13_temp= M13_temp + C_M_anderson[nM_anderson]*(M13_out[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - M13_out[kM_anderson][K]);
							M23_temp= M23_temp + C_M_anderson[nM_anderson]*(M23_out[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - M23_out[kM_anderson][K]);
							d33_temp= d33_temp + C_M_anderson[nM_anderson]*(d33_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - d33_anderson[kM_anderson][K]);
							d12_temp= d12_temp + C_M_anderson[nM_anderson]*(d12_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - d12_anderson[kM_anderson][K]);
							d13_temp= d13_temp + C_M_anderson[nM_anderson]*(d13_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - d13_anderson[kM_anderson][K]);
							d23_temp= d23_temp + C_M_anderson[nM_anderson]*(d23_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - d23_anderson[kM_anderson][K]);
						}
						//lambda_M_anderson=1.0 - pow(0.9, double(kM_anderson));
						lambda_M_anderson=lambda_M_anderson_const;
						M_OP[K][2][2]=M33_temp+lambda_M_anderson*d33_temp;
						M_OP[K][0][1]=M12_temp+lambda_M_anderson*d12_temp;
						M_OP[K][0][2]=M13_temp+lambda_M_anderson*d13_temp;
						M_OP[K][1][2]=M23_temp+lambda_M_anderson*d23_temp;
						M_OP[K][0][0]=-0.5*M_OP[K][2][2];
						M_OP[K][1][1]=M_OP[K][0][0];
						M_OP[K][1][0]=M_OP[K][0][1];
						M_OP[K][2][0]=M_OP[K][0][2];
						M_OP[K][2][1]=M_OP[K][1][2];
					}
				}
				else {//biaxial phase
					for(K=0;K<LOCAL_SIZE;K++){
						M11_temp=M11_out[kM_anderson][K];
						M33_temp=M33_out[kM_anderson][K];
						M12_temp=M12_out[kM_anderson][K];
						M13_temp=M13_out[kM_anderson][K];
						M23_temp=M23_out[kM_anderson][K];
						d11_temp=d11_anderson[kM_anderson][K];
						d33_temp=d33_anderson[kM_anderson][K];
						d12_temp=d12_anderson[kM_anderson][K];
						d13_temp=d13_anderson[kM_anderson][K];
						d23_temp=d23_anderson[kM_anderson][K];

						for(nM_anderson=1;nM_anderson<=n_r_M_temp;nM_anderson++){
							M11_temp= M11_temp + C_M_anderson[nM_anderson]*(M11_out[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - M11_out[kM_anderson][K]);
							M33_temp= M33_temp + C_M_anderson[nM_anderson]*(M33_out[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - M33_out[kM_anderson][K]);
							M12_temp= M12_temp + C_M_anderson[nM_anderson]*(M12_out[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - M12_out[kM_anderson][K]);
							M13_temp= M13_temp + C_M_anderson[nM_anderson]*(M13_out[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - M13_out[kM_anderson][K]);
							M23_temp= M23_temp + C_M_anderson[nM_anderson]*(M23_out[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - M23_out[kM_anderson][K]);
							d11_temp= d11_temp + C_M_anderson[nM_anderson]*(d11_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - d11_anderson[kM_anderson][K]);
							d33_temp= d33_temp + C_M_anderson[nM_anderson]*(d33_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - d33_anderson[kM_anderson][K]);
							d12_temp= d12_temp + C_M_anderson[nM_anderson]*(d12_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - d12_anderson[kM_anderson][K]);
							d13_temp= d13_temp + C_M_anderson[nM_anderson]*(d13_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - d13_anderson[kM_anderson][K]);
							d23_temp= d23_temp + C_M_anderson[nM_anderson]*(d23_anderson[index_updated_matrix_anderson(kM_anderson-nM_anderson,n_r_M)][K] - d23_anderson[kM_anderson][K]);
						}
						//lambda_M_anderson=1.0 - pow(0.9, double(kM_anderson));
						lambda_M_anderson=lambda_M_anderson_const;
						M_OP[K][0][0]=M11_temp+lambda_M_anderson*d11_temp;
						M_OP[K][2][2]=M33_temp+lambda_M_anderson*d33_temp;
						M_OP[K][0][1]=M12_temp+lambda_M_anderson*d12_temp;
						M_OP[K][0][2]=M13_temp+lambda_M_anderson*d13_temp;
						M_OP[K][1][2]=M23_temp+lambda_M_anderson*d23_temp;
						M_OP[K][1][1]=-M_OP[K][0][0] - M_OP[K][2][2];
						M_OP[K][1][0]=M_OP[K][0][1];
						M_OP[K][2][0]=M_OP[K][0][2];
						M_OP[K][2][1]=M_OP[K][1][2];
					}
				}

//////////////// convergency criterion //////////////
////////////////////////////////////////////////////
	/////////////////////for M ////////////////////

				double tempdd, tempMM, tempdd_global, tempMM_global;
				double tm11, tm33, tm12, tm13, tm23;
				double maxtemp1, maxtemp2, maxtemp3, maxtemp;

				if (method=='A'){//uniaxial phase: Smectic A
					
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= d33_anderson[kM_anderson][K];
						h[K]= d33_anderson[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempdd=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
										
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= M33_out[kM_anderson][K];
						h[K]= M33_out[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempMM=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
		            MPI_Barrier(MPI_COMM_WORLD);
					MPI_Reduce(&tempdd,&tempdd_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Reduce(&tempMM,&tempMM_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempdd_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempMM_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

					tm_diff_global=sqrt(fabs(tempdd_global/tempMM_global));

				}
				else if (method=='B') {//uniaxial phase: Smectic C
					
//////////////////////////////////////////////////////////////////////////
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= d33_anderson[kM_anderson][K];
						h[K]= d33_anderson[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempdd=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= M33_out[kM_anderson][K];
						h[K]= M33_out[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempMM=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

					MPI_Barrier(MPI_COMM_WORLD);
					MPI_Reduce(&tempdd,&tempdd_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Reduce(&tempMM,&tempMM_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempdd_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempMM_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

					tm33=sqrt(fabs(tempdd_global/tempMM_global));
////////////////////////////////////////////////////////////////////////////////
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= d12_anderson[kM_anderson][K];
						h[K]= d12_anderson[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempdd=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= M12_out[kM_anderson][K];
						h[K]= M12_out[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempMM=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

					MPI_Barrier(MPI_COMM_WORLD);
					MPI_Reduce(&tempdd,&tempdd_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Reduce(&tempMM,&tempMM_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempdd_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempMM_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

					tm12=sqrt(fabs(tempdd_global/tempMM_global));
///////////////////////////////////////////////////////////////////////////////////
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= d13_anderson[kM_anderson][K];
						h[K]= d13_anderson[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempdd=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= M13_out[kM_anderson][K];
						h[K]= M13_out[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempMM=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

					MPI_Barrier(MPI_COMM_WORLD);
					MPI_Reduce(&tempdd,&tempdd_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Reduce(&tempMM,&tempMM_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempdd_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempMM_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

					tm13=sqrt(fabs(tempdd_global/tempMM_global));
///////////////////////////////////////////////////////////////////////////////////
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= d23_anderson[kM_anderson][K];
						h[K]= d23_anderson[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempdd=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= M23_out[kM_anderson][K];
						h[K]= M23_out[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempMM=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;

					MPI_Barrier(MPI_COMM_WORLD);
					MPI_Reduce(&tempdd,&tempdd_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Reduce(&tempMM,&tempMM_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempdd_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempMM_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

					tm23=sqrt(fabs(tempdd_global/tempMM_global));
///////////////////////////////////////////////////////////////////////////////////
					maxtemp1=(tm33 > tm12)? tm33:tm12;
					maxtemp2=(tm13 > tm23)? tm13:tm23;
					maxtemp=(maxtemp1 > maxtemp2)? maxtemp1:maxtemp2;

					tm_diff_global=maxtemp;
				}
				else {// biaxial phase

					//////////////////////////////////////////////////////////////////////////
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= d11_anderson[kM_anderson][K];
						h[K]= d11_anderson[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempdd=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= M11_out[kM_anderson][K];
						h[K]= M11_out[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempMM=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
					MPI_Barrier(MPI_COMM_WORLD);
					MPI_Reduce(&tempdd,&tempdd_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Reduce(&tempMM,&tempMM_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempdd_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempMM_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					
					tm11=sqrt(fabs(tempdd_global/tempMM_global));
					
					//////////////////////////////////////////////////////////////////////////
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= d33_anderson[kM_anderson][K];
						h[K]= d33_anderson[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempdd=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= M33_out[kM_anderson][K];
						h[K]= M33_out[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempMM=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
					MPI_Barrier(MPI_COMM_WORLD);
					MPI_Reduce(&tempdd,&tempdd_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Reduce(&tempMM,&tempMM_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempdd_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempMM_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					
					tm33=sqrt(fabs(tempdd_global/tempMM_global));
					////////////////////////////////////////////////////////////////////////////////
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= d12_anderson[kM_anderson][K];
						h[K]= d12_anderson[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempdd=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= M12_out[kM_anderson][K];
						h[K]= M12_out[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempMM=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
					MPI_Barrier(MPI_COMM_WORLD);
					MPI_Reduce(&tempdd,&tempdd_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Reduce(&tempMM,&tempMM_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempdd_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempMM_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					
					tm12=sqrt(fabs(tempdd_global/tempMM_global));
					///////////////////////////////////////////////////////////////////////////////////
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= d13_anderson[kM_anderson][K];
						h[K]= d13_anderson[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempdd=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= M13_out[kM_anderson][K];
						h[K]= M13_out[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempMM=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
					MPI_Barrier(MPI_COMM_WORLD);
					MPI_Reduce(&tempdd,&tempdd_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Reduce(&tempMM,&tempMM_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempdd_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempMM_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					
					tm13=sqrt(fabs(tempdd_global/tempMM_global));
					///////////////////////////////////////////////////////////////////////////////////
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= d23_anderson[kM_anderson][K];
						h[K]= d23_anderson[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempdd=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
					for(K=0;K<LOCAL_SIZE;K++){
						g[K]= M23_out[kM_anderson][K];
						h[K]= M23_out[kM_anderson][K];
						gh[K]=g[K]*h[K];
					}
					tempMM=simposon_mpi (DIM,local_ni[0], local_ni[1], SIDEz, dx, dy, dz, gh)/M_v;
					
					MPI_Barrier(MPI_COMM_WORLD);
					MPI_Reduce(&tempdd,&tempdd_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Reduce(&tempMM,&tempMM_global,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempdd_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					MPI_Bcast(&tempMM_global,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
					
					tm23=sqrt(fabs(tempdd_global/tempMM_global));
					///////////////////////////////////////////////////////////////////////////////////
					maxtemp1=(tm33 > tm12)? tm33:tm12;
					maxtemp2=(tm13 > tm23)? tm13:tm23;
					maxtemp3=(maxtemp1 > maxtemp2)? maxtemp1:maxtemp2;
					 maxtemp=(maxtemp3 > tm11)? maxtemp3:tm11;
					
					tm_diff_global=maxtemp;
				}

				free_dvector_NR(C_M_anderson,1,n_r_M_temp);
				free_dvector_NR(V_M_anderson,1,n_r_M_temp);
				free_dmatrix_NR(U_M_anderson,1,n_r_M_temp,1,n_r_M_temp);
				free_dmatrix_NR(inverse_U_M_anderson,1,n_r_M_temp,1,n_r_M_temp);
				free_dvector(g,0,LOCAL_SIZE-1);
				free_dvector(h,0,LOCAL_SIZE-1);
		        free_dvector(gh,0,LOCAL_SIZE-1);
			
			}

			else tm_diff_global=0.5;

		}

	}

	free_dmatrix(temp3m,0,N_dim_ddm-1,0,N_dim_ddm-1);
    
}
/************************************************************************/
/////////////////////////////////////////////////////////////////////////
void output1()
{
	FILE *pf;
	
	//////////////////////////////////////////////////////////////////
	if (myid == 0){
		if (pf=fopen("results.dat","a"))
		{
			fprintf(pf,"dx*SIDEx=%lf  dy*SIDEy=%lf  dz*SIDEz=%lf  FE=%lf  pff=%lf  %e %e %e\n",dx*SIDEx,dy*SIDEy,dz*SIDEz,FE_global,pff_global,ta_diff_global,tb_diff_global,tm_diff_global);
		    fclose(pf);	
		}
	}
	
}
/************************************************************************/
/////////////////////////////////////////////////////////////////////////
void output()
{
	register int i;
	register int K,K_i, K_j, K_k;
	FILE *pf;

///////////////////////////////////////////////////////////////////////////////
	if (myid == 0){
			
		if (pf=fopen("free_energy.dat","a"))
		{	
			fprintf(pf,"dx*SIDEx=%lf  dy*SIDEy=%lf  dz*SIDEz=%lf  dx=%lf  dy=%lf  dz=%lf  pff_global=%lf  FE_global=%lf   FE_global - FE(homogeneous)=%lf\n",dx*SIDEx,dy*SIDEy,dz*SIDEz,dx,dy,dz,pff_global,FE_global,FE_global-NXab*fa*fb);
			
			fclose(pf);	
		}
		
	}
////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
	sprintf(name1,"RHO_total%d.dat",myid);

	if (pf=fopen(name1,"wb"))
	{
		fprintf(pf,"local_i_start[0] = %d\n",local_i_start[0]);
		fprintf(pf,"local_i_start[1] = %d\n",local_i_start[1]);

		for(K_i=0,K=0;K_i<local_ni[0];K_i++){
			for(K_j=0;K_j<local_ni[1];K_j++){
				for(K_k=0;K_k<SIDEz;K_k++,K++){
					fprintf(pf,"%lf %lf %lf %lf %lf %lf\n",(K_i+local_i_start[0])*dx, (K_j+local_i_start[1])*dy, K_k*dz, RA[K],RB[K],(1.0-RA[K]-RB[K]));
				}
			}
		}
			
		
		fclose(pf);	
	}


/////////////////////////////////////////////////////////////////////
	sprintf(name2,"RHO_A%d.dat",myid);

	if (pf=fopen(name2,"wb"))
	{	
		fprintf(pf,"local_i_start[0] = %d\n",local_i_start[0]);
		fprintf(pf,"local_i_start[1] = %d\n",local_i_start[1]);
		for(K_i=0,K=0;K_i<local_ni[0];K_i++){
			for(K_j=0;K_j<local_ni[1];K_j++){
				for(K_k=0;K_k<SIDEz;K_k++,K++){
					fprintf(pf,"%lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, RA[K]);
				}
			}
		}
			
		
		fclose(pf);	
	}
////////////////////////////////////////////////////////////////////////
	sprintf(name3,"RHO_B%d.dat",myid);

	if (pf=fopen(name3,"wb"))
	{	
		fprintf(pf,"local_i_start[0] = %d\n",local_i_start[0]);
		fprintf(pf,"local_i_start[1] = %d\n",local_i_start[1]);

		for(K_i=0,K=0;K_i<local_ni[0];K_i++){
			for(K_j=0;K_j<local_ni[1];K_j++){
				for(K_k=0;K_k<SIDEz;K_k++,K++){
					fprintf(pf,"%lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, RB[K]);
				}
			}
		}
			
		
		fclose(pf);	
	}
////////////////////////////////////////////////////////////////////////
	sprintf(name3,"RHO_half%d.dat",myid);
	
	if (pf=fopen(name3,"wb"))
	{	
		fprintf(pf,"local_i_start[0] = %d\n",local_i_start[0]);
		fprintf(pf,"local_i_start[1] = %d\n",local_i_start[1]);
		
		for(K_i=0,K=0;K_i<local_ni[0];K_i++){
			for(K_j=0;K_j<local_ni[1];K_j++,K++){
				fprintf(pf,"%lf %lf %lf\n",K_i*dx, K_j*dy,R_half[K]);
			}
		}
		fclose(pf);	
	}
////////////////////////////////////////////////////////////////////////
	sprintf(name3,"RHO_end%d.dat",myid);
	
	if (pf=fopen(name3,"wb"))
	{	
		fprintf(pf,"local_i_start[0] = %d\n",local_i_start[0]);
		fprintf(pf,"local_i_start[1] = %d\n",local_i_start[1]);
		
		for(K_i=0,K=0;K_i<local_ni[0];K_i++){
			for(K_j=0;K_j<local_ni[1];K_j++,K++){
				fprintf(pf,"%lf %lf %lf\n",K_i*dx, K_j*dy,R_end[K]);
			}
		}
		fclose(pf);	
	}
////////////////////////////////////////////////////////////////////////
/*
	sprintf(name4,"RHO_A_complete%d.dat",myid);

	if (pf=fopen(name4,"wb"))
	{	
		fprintf(pf,"local_i_start[0] = %d\n",local_i_start[0]);

		for(K_i=0,K=0;K_i<local_ni[0];K_i++){
			for(K_j=0;K_j<local_ni[1];K_j++){
				for(K_k=0;K_k<SIDEz;K_k++,K++){
					for(i=0;i<M_Bar;i++){
						fprintf(pf,"%lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, RHOA[K][i]);	
					}
				}
			}
		}
				
			
		
					
		fclose(pf);	
	}
*/
////////////////////////////////////////////////////////////////////////
/*
	sprintf(name5,"RHO_B_complete%d.dat",myid);

	if (pf=fopen(name5,"wb"))
	{	
		fprintf(pf,"local_i_start[0] = %d\n",local_i_start[0]);

		for(K_i=0,K=0;K_i<local_ni[0];K_i++){
			for(K_j=0;K_j<local_ni[1];K_j++){
				for(K_k=0;K_k<SIDEz;K_k++,K++){
					for(i=0;i<M_Bar;i++){
						fprintf(pf,"%lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, RHOB[K][i]);	
					}
				}
			}
		}
		fclose(pf);	
	}
*/
////////////////////////////////////////////////////////////////////////
	sprintf(name6,"W%d.dat",myid);

	if (pf=fopen(name6,"wb"))
	{	
		fprintf(pf,"local_i_start[0] = %d\n",local_i_start[0]);
		fprintf(pf,"local_i_start[1] = %d\n",local_i_start[1]);

		for(K=0;K<LOCAL_SIZE;K++){
		    fprintf(pf,"%d %lf %lf\n",K,WA[K],WB[K]);
		}
		fclose(pf);	
	}
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
	sprintf(name7,"S_OP%d.dat",myid);

	if (pf=fopen(name7,"wb"))
	{	
		fprintf(pf,"local_i_start[0] = %d\n",local_i_start[0]);
		fprintf(pf,"local_i_start[1] = %d\n",local_i_start[1]);
		for(K_i=0,K=0;K_i<local_ni[0];K_i++){
			for(K_j=0;K_j<local_ni[1];K_j++){
				for(K_k=0;K_k<SIDEz;K_k++,K++){
					fprintf(pf," %lf %lf %lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, S_OP[K][0][0],S_OP[K][0][1],S_OP[K][0][2]);
			        fprintf(pf," %lf %lf %lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, S_OP[K][1][0],S_OP[K][1][1],S_OP[K][1][2]);
			        fprintf(pf," %lf %lf %lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, S_OP[K][2][0],S_OP[K][2][1],S_OP[K][2][2]);
				}
			}
		}
					
		fclose(pf);	
	}
/////////////////////////////////////////////////////////////////////
	sprintf(name8,"SA_OP%d.dat",myid);
	
	if (pf=fopen(name8,"wb"))
	{	
		fprintf(pf,"local_i_start[0] = %d\n",local_i_start[0]);
		fprintf(pf,"local_i_start[1] = %d\n",local_i_start[1]);
		for(K_i=0,K=0;K_i<local_ni[0];K_i++){
			for(K_j=0;K_j<local_ni[1];K_j++){
				for(K_k=0;K_k<SIDEz;K_k++,K++){
					fprintf(pf," %lf %lf %lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, SA_OP[K][0][0],SA_OP[K][0][1],SA_OP[K][0][2]);
					fprintf(pf," %lf %lf %lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, SA_OP[K][1][0],SA_OP[K][1][1],SA_OP[K][1][2]);
					fprintf(pf," %lf %lf %lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, SA_OP[K][2][0],SA_OP[K][2][1],SA_OP[K][2][2]);
				}
			}
		}
		
		fclose(pf);	
	}
/////////////////////////////////////////////////////////////////////
	sprintf(name9,"SB_OP%d.dat",myid);
	
	if (pf=fopen(name9,"wb"))
	{	
		fprintf(pf,"local_i_start[0] = %d\n",local_i_start[0]);
		fprintf(pf,"local_i_start[1] = %d\n",local_i_start[1]);
		for(K_i=0,K=0;K_i<local_ni[0];K_i++){
			for(K_j=0;K_j<local_ni[1];K_j++){
				for(K_k=0;K_k<SIDEz;K_k++,K++){
					fprintf(pf," %lf %lf %lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, SB_OP[K][0][0],SB_OP[K][0][1],SB_OP[K][0][2]);
					fprintf(pf," %lf %lf %lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, SB_OP[K][1][0],SB_OP[K][1][1],SB_OP[K][1][2]);
					fprintf(pf," %lf %lf %lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, SB_OP[K][2][0],SB_OP[K][2][1],SB_OP[K][2][2]);
				}
			}
		}
		
		fclose(pf);	
	}

	/////////////////////////////////////////////////////////////////////
	sprintf(name10,"M_OP%d.dat",myid);
	
	if (pf=fopen(name10,"wb"))
	{	
		fprintf(pf,"local_i_start[0] = %d\n",local_i_start[0]);
		fprintf(pf,"local_i_start[1] = %d\n",local_i_start[1]);
		for(K_i=0,K=0;K_i<local_ni[0];K_i++){
			for(K_j=0;K_j<local_ni[1];K_j++){
				for(K_k=0;K_k<SIDEz;K_k++,K++){
					fprintf(pf," %lf %lf %lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, M_OP[K][0][0],M_OP[K][0][1],M_OP[K][0][2]);
					fprintf(pf," %lf %lf %lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, M_OP[K][1][0],M_OP[K][1][1],M_OP[K][1][2]);
					fprintf(pf," %lf %lf %lf %lf %lf %lf\n",K_i*dx, K_j*dy, K_k*dz, M_OP[K][2][0],M_OP[K][2][1],M_OP[K][2][2]);
				}
			}
		}
		
		fclose(pf);	
	}
//////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
}
/************************************************************************/
/////////////////////////////////////////////////////////////////////////
void readw()
{
	float wa,wb;
	int i1;
	int K;
	FILE *fp;
	int temp_local_i_start[0];
	float tempx, tempy, tempz, temp11, temp22,temp33;


	sprintf(name6,"W%d.dat",myid);

	if(fp=fopen(name6,"rb"))
	{
		fscanf (fp, "local_i_start[0] = %d\n", &temp_local_i_start[0]);
		fscanf (fp, "local_i_start[1] = %d\n", &temp_local_i_start[1]);
		for(K=0;K<LOCAL_SIZE;K++){
			fscanf(fp,"%d\t%f\t%f\n",&i1,&wa,&wb);
			WA[K]=wa;
			WB[K]=wb;
		}
		fclose (fp);
	}	
	
	
	
	if (fabs(NMu-0.0)<1.0e-4) 
	{
		tempx=0.0;
	}
	else 
	{
		sprintf(name10,"M_OP%d.dat",myid);
		
		if(fp=fopen(name10,"rb"))
		{
			fscanf (fp, "local_i_start[0] = %d\n", &temp_local_i_start[0]);
			fscanf (fp, "local_i_start[1] = %d\n", &temp_local_i_start[1]);
			for(K=0;K<LOCAL_SIZE;K++){
				
				fscanf(fp,"%f\t%f\t%f\t%f\t%f\t%f\n",&tempx,&tempy,&tempz,&temp11,&temp22,&temp33);
				M_OP[K][0][0]=double(temp11);
				M_OP[K][0][1]=double(temp22);
				M_OP[K][0][2]=double(temp33);
				
				fscanf(fp,"%f\t%f\t%f\t%f\t%f\t%f\n",&tempx,&tempy,&tempz,&temp11,&temp22,&temp33);
				M_OP[K][1][0]=double(temp11);
				M_OP[K][1][1]=double(temp22);
				M_OP[K][1][2]=double(temp33);
				
				fscanf(fp,"%f\t%f\t%f\t%f\t%f\t%f\n",&tempx,&tempy,&tempz,&temp11,&temp22,&temp33);
				M_OP[K][2][0]=double(temp11);
				M_OP[K][2][1]=double(temp22);
				M_OP[K][2][2]=double(temp33);
			}
			fclose (fp);
		}
		
	}

	
}
/************************************************************************/
/////////////////////////////////////////////////////////////////////////
void initw()
{
	int K_i, K_j, K_k;
	int K;
	int j,k;


/////////////////////////////  Random initial ///////////////////////
/*
	double ramini=1.0;
	int iseed=1;
	int initial_step=10000;

    iseed=(myid)*10+1;
  
	for(int i=0;i<(myid + 1)*initial_step;i++) art_rand(iseed);


	for(K_i=0,K=0;K_i<local_ni[0];K_i++)
		for(K_j=0;K_j<local_ni[1];K_j++)
			for(K_k=0;K_k<SIDEz;K_k++,K++)
			{
				WA[K] = art_rand(iseed)*ramini;				
                WB[K] = art_rand(iseed)*ramini;
			}


			
	for(K=0;K<LOCAL_SIZE;K++)
		for(j=0;j<N_dim_ddm;j++)
			for(k=0;k<N_dim_ddm;k++){
				S_OP[K][j][k]=0.0;
			}
					
					
					
	if (method=='A') {//uniaxial phase: Smectic A
		for(K=0;K<LOCAL_SIZE;K++){
			S_OP[K][0][0]=art_rand(iseed)*ramini;
			S_OP[K][1][1]=art_rand(iseed)*ramini;
			S_OP[K][2][2]=art_rand(iseed)*ramini;
		}
	}
						
					
	else if (method=='B') {//uniaxial phase: Smectic C
		for(K=0;K<LOCAL_SIZE;K++){
			S_OP[K][2][2]=art_rand(iseed)*ramini;
			S_OP[K][0][1]=art_rand(iseed)*ramini;
			S_OP[K][0][2]=art_rand(iseed)*ramini;
			S_OP[K][1][2]=art_rand(iseed)*ramini;
							
			S_OP[K][0][0]=-0.5*S_OP[K][2][2];
			S_OP[K][1][1]=S_OP[K][0][0];
			S_OP[K][1][0]=S_OP[K][0][1];
			S_OP[K][2][0]=S_OP[K][0][2];
			S_OP[K][2][1]=S_OP[K][1][2];
		}
	}
						
					
	else {//biaxial phase
		for(K=0;K<LOCAL_SIZE;K++){
			S_OP[K][0][0]=art_rand(iseed)*ramini;
			S_OP[K][2][2]=art_rand(iseed)*ramini;
			S_OP[K][0][1]=art_rand(iseed)*ramini;
			S_OP[K][0][2]=art_rand(iseed)*ramini;
			S_OP[K][1][2]=art_rand(iseed)*ramini;
							
			S_OP[K][1][1]=-S_OP[K][0][0]-S_OP[K][2][2];
			S_OP[K][1][0]=S_OP[K][0][1];
			S_OP[K][2][0]=S_OP[K][0][2];
			S_OP[K][2][1]=S_OP[K][1][2];
		}
	}
			
	
	for(K=0;K<LOCAL_SIZE;K++)
		for(j=0;j<N_dim_ddm;j++)
			for(k=0;k<N_dim_ddm;k++){
				M_OP[K][j][k]=M_initial*S_OP[K][j][k];
			}
							
			
*/			
/////////////////////////////  FCC initial 3D ////////////////////////
/*
	for(K_i=0,K=0;K_i<local_ni[0];K_i++)
		for(K_j=0;K_j<local_ni[1];K_j++)
			for(K_k=0;K_k<SIDEz;K_k++,K++)
			{
				WA[K] = NXab*(1-fa*(1+0.7*(cos(2.0*PI*(local_i_start[0]+K_i+1)/SIDEx)
					*cos(2.0*PI*(K_j+1)/local_ni[1])*cos(2.0*PI*(K_k+1)/SIDEz))));
				
                WB[K] = NXab*fa*(1+0.7*(cos(2.0*PI*(local_i_start[0]+K_i+1)/SIDEx)
					*cos(2.0*PI*(K_j+1)/local_ni[1])*cos(2.0*PI*(K_k+1)/SIDEz)));	
			}

*/
/////////////////////////////  BCC initial 3D ////////////////////////
/*
	for(K_i=0,K=0;K_i<local_ni[0];K_i++)
		for(K_j=0;K_j<local_ni[1];K_j++)
			for(K_k=0;K_k<SIDEz;K_k++,K++)
			{
				WA[K] = NXab*(1-fa*(1+0.7*(cos(2.0*PI*(local_i_start[0]+K_i+1)/SIDEx)*cos(2.0*PI*(K_j+1)/SIDEx)
						          +cos(2.0*PI*(K_j+1)/local_ni[1])*cos(2.0*PI*(K_k+1)/SIDEz)
								  +cos(2.0*PI*(local_i_start[0]+K_i+1)/SIDEx)*cos(2.0*PI*(K_k+1)/SIDEz))));

                WB[K] = NXab*fa*(1+0.7*(cos(2.0*PI*(local_i_start[0]+K_i+1)/SIDEx)*cos(2.0*PI*(K_j+1)/local_ni[1])
						          +cos(2.0*PI*(K_j+1)/local_ni[1])*cos(2.0*PI*(K_k+1)/SIDEz)
								  +cos(2.0*PI*(local_i_start[0]+K_i+1)/SIDEx)*cos(2.0*PI*(K_k+1)/SIDEz)));	
			}
					
*/
//////////////////////////// Gyroid initial 3D /////////////////////////

        assert(DIM==3); 

	for(K_i=0,K=0;K_i<local_ni[0];K_i++)
		for(K_j=0;K_j<local_ni[1];K_j++)
			for(K_k=0;K_k<SIDEz;K_k++,K++)
			{
	WA[K] = NXab*(1-fa*(1+0.7*(cos(2.0*PI*(local_i_start[0]+K_i+1)/SIDEx)*sin(2.0*PI*(local_i_start[1]+K_j+1)/SIDEy)*sin(4.0*PI*(K_k+1)/SIDEz)
	+cos(2.0*PI*(local_i_start[1]+K_j+1)/SIDEy)*sin(2.0*PI*(K_k+1)/SIDEz)*sin(4.0*PI*(local_i_start[0]+K_i+1)/SIDEx)
	+cos(2.0*PI*(K_k+1)/SIDEz)*sin(2.0*PI*(local_i_start[0]+K_i+1)/SIDEx)*sin(4.0*PI*(local_i_start[1]+K_j+1)/SIDEy))));

                WB[K] = NXab*fa*(1+0.7*(cos(2.0*PI*(local_i_start[0]+K_i+1)/SIDEx)*sin(2.0*PI*(local_i_start[1]+K_j+1)/SIDEy)*sin(4.0*PI*(K_k+1)/SIDEz) 
		+cos(2.0*PI*(local_i_start[1]+K_j+1)/SIDEy)*sin(2.0*PI*(K_k+1)/SIDEz)*sin(4.0*PI*(local_i_start[0]+K_i+1)/SIDEx)
		+cos(2.0*PI*(K_k+1)/SIDEz)*sin(2.0*PI*(local_i_start[0]+K_i+1)/SIDEx)*sin(4.0*PI*(local_i_start[1]+K_j+1)/SIDEy)));
			}
					




//////////////////////////// Cylinder initial 3D /////////////////////////
/*

	for(K=0;K<LOCAL_SIZE;K++){
		WA[K] =0.0;
		WB[K] =0.0;
	}


	for(K_i=0,K=0;K_i<local_ni[0];K_i++)
		for(K_j=0;K_j<local_ni[1];K_j++)
			for(K_k=0;K_k<SIDEz;K_k++,K++)
			{
				WA[K] = NXab*(1-fa*(1+0.8*cos(2.0*PI*(local_i_start[0]+K_i+1)/SIDEx)*cos(2.0*PI*(K_j+1)/local_ni[1])));
				WB[K] = NXab*fa*(1+0.8*cos(2.0*PI*(local_i_start[0]+K_i+1)/SIDEx)*cos(2.0*PI*(K_j+1)/local_ni[1])); 
			}
					
*/

//////////////////////////// Lamellar initial 3D /////////////////////////

/*

	for(K=0;K<LOCAL_SIZE;K++)
	{
		WA[K] =0.0;
		WB[K] =0.0;
	} 
	

	for(K_i=0,K=0;K_i<local_ni[0];K_i++)
		for(K_j=0;K_j<local_ni[1];K_j++)
			for(K_k=0;K_k<SIDEz;K_k++,K++)
			{
				WA[K] = NXab*(1-fa*(1+0.8*sin(2.0*PI*(K_k+1)/SIDEz)));
				WB[K] = NXab*fa*(1+0.8*sin(2.0*PI*(K_k+1)/SIDEz));
			}
*/					

///////////////////////////////////for M//////////////////////////////////////////////////	
/*			
	
			
	for(K=0;K<LOCAL_SIZE;K++)
		for(j=0;j<N_dim_ddm;j++)
			for(k=0;k<N_dim_ddm;k++){
				S_OP[K][j][k]=0.0;
			}
					

	if (method=='A') {//uniaxial phase: Smectic A

		for(K=0;K<LOCAL_SIZE;K++){
			S_OP[K][0][0]=-0.5;
			S_OP[K][1][1]=-0.5;
			S_OP[K][2][2]=1.0;
		}
	}
	else if (method=='B') {//uniaxial phase: Smectic C

		for(K=0;K<LOCAL_SIZE;K++){
			S_OP[K][2][2]=1.0;
			S_OP[K][0][1]=0.5;
			S_OP[K][0][2]=0.5;
			S_OP[K][1][2]=0.5;

			S_OP[K][0][0]=-0.5*S_OP[K][2][2];
			S_OP[K][1][1]=S_OP[K][0][0];
			S_OP[K][1][0]=S_OP[K][0][1];
			S_OP[K][2][0]=S_OP[K][0][2];
			S_OP[K][2][1]=S_OP[K][1][2];
		}
	}
	else {//biaxial phase
		for(K=0;K<LOCAL_SIZE;K++){
			S_OP[K][0][0]=-0.5;
			S_OP[K][2][2]=1.0;
			S_OP[K][0][1]=0.5;
			S_OP[K][0][2]=0.5;
			S_OP[K][1][2]=0.5;

			S_OP[K][1][1]=-S_OP[K][0][0]-S_OP[K][2][2];
			S_OP[K][1][0]=S_OP[K][0][1];
			S_OP[K][2][0]=S_OP[K][0][2];
			S_OP[K][2][1]=S_OP[K][1][2];
		}
	}


	for(K=0;K<LOCAL_SIZE;K++)
		for(j=0;j<N_dim_ddm;j++)
			for(k=0;k<N_dim_ddm;k++){
				M_OP[K][j][k]=M_initial*S_OP[K][j][k];
			}
*/					
}
/************************************************************************/
/////////////////////////////////////////////////////////////////////////
int converge()
{
	
	if (fabs(NMu-0.0)<1.0e-4) {

		if((ta_diff_global<CC)&&(tb_diff_global<CC))   return 1;
		else	return 0;
	}
	else {
		if((ta_diff_global<CC)&&(tb_diff_global<CC)&&(tm_diff_global<CC_M))		return 1;
		else	return 0;
	}	
} 
/************************************************************************/
/////////////////////////////////////////////////////////////////////////
int converge_final()
{
	

	if (fabs(NMu-0.0)<1.0e-4) {
		if((ta_diff_global<CC_FINAL)&&(tb_diff_global<CC_FINAL))	return 1;
		else  	return 0;
	}
	else {
		if((ta_diff_global<CC_FINAL)&&(tb_diff_global<CC_FINAL)&&(tm_diff_global<CC_M_FINAL))	return 1;
		else  	return 0;
	}
} 
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
void init_parameters()
{
	int K;

	double *denz=dvector(0,SIDEx*SIDEy*SIDEz-1);
	
	for(K=0;K<SIDEx*SIDEy*SIDEz;K++)
	{
		denz[K]=1.0;
	}
	M_v=simposon_mpi (DIM,SIDEx, SIDEy, SIDEz, dx, dy, dz, denz);
	
    free_dvector(denz,0,SIDEx*SIDEy*SIDEz-1);
	
/*
	int n_anderson;
	
	for(n_anderson=0;n_anderson<=n_r_WAB;n_anderson++){
		for(K=0;K<LOCAL_SIZE;K++){
			WA_out[n_anderson][K]=0.0;
			WB_out[n_anderson][K]=0.0;
			dA_anderson[n_anderson][K]=0.0;
			dB_anderson[n_anderson][K]=0.0;
		}		
	}
	
*/	

	pff_global=0.0;
	ta_diff_global=tb_diff_global=tm_diff_global=100.0;

/////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////


		
	int i,j;
	int G_indx;
        int K_i,K_j,K_k;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////// initial matrix sa and ija ///////////////////////////////////////////
	
    for(G_indx=0;G_indx<=7;G_indx++){
	for(K=1;K<=LOCAL_SIZE;K++){
		for(i=1;i<=namax;i++){
		    sa_G_R_inv[G_indx][K][i]=0.0;  sa_G_I_inv[G_indx][K][i]=0.0;
		    ija_G_R_inv[G_indx][K][i]=0;   ija_G_I_inv[G_indx][K][i]=0;
		  }
	  }
     }

	////////////////////////////////////////////////////////////////////////////////////////////
	
	double **atemp_R0=dmatrix_NR(0,M_Bar-1,0,M_Bar-1);
	double **atemp_I0=dmatrix_NR(0,M_Bar-1,0,M_Bar-1);
	double **atemp_R=dmatrix_NR(1,M_Bar,1,M_Bar);
	double **atemp_I=dmatrix_NR(1,M_Bar,1,M_Bar);
        double G_para[8][3];
        G_para[0][0]=1.0;
        G_para[0][1]=1.0;
        G_para[0][2]=kapa_A;
        G_para[1][0]=1.5;
        G_para[1][1]=1.0;
        G_para[1][2]=kapa_A;
        G_para[2][0]=11.0/6.0;
        G_para[2][1]=1.0;
        G_para[2][2]=kapa_A;
        G_para[3][0]=11.0/6.0;
        G_para[3][1]=1.0;
        G_para[3][2]=kapa_B;
        
        G_para[4][0]=1.0;
        G_para[4][1]=-1.0;
        G_para[4][2]=kapa_B;
        G_para[5][0]=1.5;
        G_para[5][1]=-1.0;
        G_para[5][2]=kapa_B;
        G_para[6][0]=11.0/6.0;
        G_para[6][1]=-1.0;
        G_para[6][2]=kapa_B;
        G_para[7][0]=11.0/6.0;
        G_para[7][1]=-1.0;
        G_para[7][2]=kapa_A;


int nam[2];
int nam_temp;
nam_temp=0;
for(G_indx=0;G_indx<=7;G_indx++){
    for(K_i=0,K=1;K_i<local_ni[0];K_i++){
      for(K_j=0;K_j<local_ni[1];K_j++){
	for(K_k=0;K_k<SIDEz;K_k++,K++){
        K_i=2;
        K_j=2;
        K_k=2;
     matrix_G_inverse(M_Bar, K_i,K_j,K_k,G_para[G_indx][0],  G_para[G_indx][1], G_para[G_indx][2], matrix_Rx, matrix_Ry, matrix_Rz, atemp_R0, atemp_I0); 
		for(i=1;i<=M_Bar;i++){
			for(j=1;j<=M_Bar;j++){
				atemp_R[i][j]=atemp_R0[i-1][j-1];
				atemp_I[i][j]=atemp_I0[i-1][j-1];
			}
		}
		sprsin(atemp_R, M_Bar, Thresh_sprase_matrix, namax, sa_G_R_inv[G_indx][K],ija_G_R_inv[G_indx][K]);
		sprsin(atemp_I, M_Bar, Thresh_sprase_matrix, namax, sa_G_I_inv[G_indx][K],ija_G_I_inv[G_indx][K]);


	}
     }
    }
  }

        exit(0);	
	
	free_dmatrix_NR(atemp_R0,0,M_Bar-1,0,M_Bar-1);
	free_dmatrix_NR(atemp_I0,0,M_Bar-1,0,M_Bar-1);
	free_dmatrix_NR(atemp_R,1,M_Bar,1,M_Bar);
	free_dmatrix_NR(atemp_I,1,M_Bar,1,M_Bar);


}
/************************************************************************/
/////////////////////////////////////////////////////////////////////////
void cal_scft_final()
{
	Num_iteration_step_WM=1;
	Num_iteration_step_M=0;

	// be used in the function iterate()
	while(!converge_final()){
		calc_THETAij();
		MDE_q();
		//MDE_qstar();
		density();
		free_energy();
		iterate();
		output1();
		Num_iteration_step_WM++;
		
	}
	free_energy();
	output();
}
/************************************************************************/
/////////////////////////////////////////////////////////////////////////
double cal_scft(double tz){
	
	dx=tz;
	//dy=dx*SIDEx/(sqrt(3.0)*SIDEy);
	dy=dx;
	dz=dx;
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&dx,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&dy,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&dz,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

	init_parameters();
	Num_iteration_step_WM=1;
	Num_iteration_step_M=0;
	
	
	//initw();
	//readw();

	
	while(!converge()){
		calc_THETAij();
		MDE_q();
		//MDE_qstar();
		density();
		free_energy();
                //exit(0);
		iterate();
        output1();
		
		Num_iteration_step_WM++;
		
	}
	free_energy();
	output();
	return FE_global;
}
