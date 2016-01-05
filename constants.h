#ifndef	_CONSTANTS_H_
#define _CONSTANTS_H_
//////////////////////////////////////////////////////
/////////////////////////////////////////////////////
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream> 
#include <fstream>
#include <vector>
//#include <fftw.h>
//#include <fftw_mpi.h>
#include <pfft.h>
#include <mpi.h>

using namespace std;
//////////////////////////////////////////////////////
/////////////////////////////////////////////////////
#define CC 1e-5 
#define CC_FINAL 1e-6

#define CC_M 1e-4  
#define CC_M_FINAL 1e-4
//////////////////////////////////////////////////////
/////////////////////////////////////////////////////
#define LLS 1e-8  
#define TINY 1.0e-40;
#define NR_END 1
#define FREE_ARG char*
#define MAXITS 10000
#define EPS 1.0e-10
#define TOLF 1.0e-4
#define TOLX EPS
#define STPMX 100.0
#define TOLMIN 1.0e-6
#define ALF 1.0e-4
#define PI 3.141592653589792
#define N_dim_ddm 3 // the dimension for the tensor of double dot multiply (ddm) on the basis of Maier-Saupe potential
////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//#define L_Bar_A 5
//#define M_Bar_A (L_Bar_A+1)*(L_Bar_A+1)
//#define Dim_ordering_A  (M_Bar_A*M_Bar_A+L_Bar_A)

#define L_Bar 6  // L_Bar_B >= L_Bar_A  ***********
#define M_Bar (L_Bar+1)*(L_Bar+1)
#define Dim_ordering  (M_Bar*M_Bar+L_Bar)
#define namax (M_Bar*M_Bar+L_Bar)
//for 3D case:
//L=6,M=49,namax=640*2
//L=7,M=64,namax=1100*2
//L=8  2*1800
//L=9  2*2600
//L=10 2*3800
//L=11 2*5300
//L=12 2*7000
//L=13 2*9300
//L=14 2*12000
//L=15 2*14600
//L=16 2*18000
//L=17 2*22000
//L=18 2*26000
//L=19 2*30000
//L=20 2*24000

#define Thresh_sprase_matrix 1.0e-10
/////////////////////////////////////////////////////
const int NDIM=1;
///////////////////////////////////////////////////
const double NN=3.0;  // polymerization
const int SIDEx=32;
const int SIDEy=32;
const int SIDEz=32;
const int DIM=3;
#define Dim3  // Dimension, Dim1,Dim2,Dim3

const double M_grid=1.0*SIDEx*SIDEy*SIDEz; // the number of grids for the spacial division
extern int LOCAL_SIZE;
const int NMAX=100;
const double ds=1.0/NMAX;
const double kapa_A=0.5;
const double kapa_B=0.5;  
extern double dx;
extern double dy;
extern double dz;
////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
#define IADD   453806245
#define IMUL   314159269
#define MASK   2147483647
#define SCALE  0.4656612873e-9
///////////////////////////////////////////////////////////
/////////////////// Anderson Mixing ////////////////////////////////////////
const int n_r_WAB=10; // the steps including the preceding n_r_WAB historic steps
const int Num_step_simple_mixing_for_WAB=1;
extern double lambda_WAB_anderson;
const double lambda_WAB_anderson_const=0.0;


const int Num_step_simple_mixing_for_M=5000000;
const int Num_step_iterae_M=20;  // update the orientational field "M" after every Num_step_iterae_M steps for anderson mixing
const int n_r_M=4;
extern double lambda_M_anderson;
const double lambda_M_anderson_const=0.0;
extern int Num_iteration_step_M; // counting the practical number of iterated M field

const char method='C'; 
// NOTE, uniaxial phase i.e. M_xx  = M_yy = -(1/2)M_zz; xx-->00, yy-->11, zz-->22
// NOTE, biaxial phase i.e. M_xx != M_yy ; xx-->00, yy-->11, zz-->22
// A represents uniaxial phase and only elements on the diagonal are nonezero (i.e. Smectic A); 
// B represents general uniaxial phase (i.e. Smectic C);
// not A or not B represents biaxial phase;
///////////////////////////////////////////////////
const double fa=0.4; 
const double fb=1.0-fa;
const int NA=int(NMAX*fa); // NA is looked as segment A
const int NB=NMAX-NA;
const double lanbtWA=0.2;
const double lanbtWB=0.2;
const double lanbtM=0.1;
extern int Num_iteration_step_WM; // counting the number of total iteration steps
const double NXab=9.00;


const double NMu_NXab=0.0; // NMu/NXab
const double NMu=NXab*NMu_NXab;
const double M_initial=NXab*NMu_NXab*0.1;
///////////////////////////////////////////////////
#endif
