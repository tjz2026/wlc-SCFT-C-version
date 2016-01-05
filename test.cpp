///////////////////////////////////////////////////////////////////////
#include <complex.h>
//#include <pfft.h>
#include "mde.h"
#include "minimal.h"
#include <ctime>
#define tol 1.0e-2
int LOCAL_SIZE;

////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){

      int np[2]; // np[0] cpu cores on X axis, np[1] on Y axis.
      ptrdiff_t n[3]; 
      ptrdiff_t alloc_local; // local data size
      double err;
      MPI_Comm comm_cart_2d;
    
      /* Set size of FFT and process mesh */
      n[0] = SIDEx; n[1] = SIDEy; n[2] = SIDEz;
      np[0] = 8; np[1] = 2;
    
        clock_t start;
	start=clock();
	FILE *fp;
	
/// this is the 2d pencil decomposition used in pfft  //
      /* Set size of FFT and process mesh */


  /* Initialize MPI and PFFT */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  pfft_init();

  /* Create two-dimensional process grid of size np[0] x np[1], if possible */
  if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d) ){
    pfft_fprintf(MPI_COMM_WORLD, stderr, "Error: This test file only works with %d processes.\n", np[0]*np[1]);
    MPI_Finalize();
    return 1;
  }

  /* Get parameters of data distribution */
  alloc_local = pfft_local_size_dft_3d(n, comm_cart_2d, PFFT_TRANSPOSED_NONE,
      local_ni, local_i_start, local_no, local_o_start);

//  print the data decomposition info
	cout<<alloc_local<<"local size"<<endl<<"myid="<<myid;
       if (myid==0) {
	cout<<alloc_local<<"local size"<<endl;
	cout<<local_ni[0]*local_ni[1]*local_ni[2]<<"local size for x,y,z"<<endl;
	cout<<local_no[0]<<" "<<local_no[1]<<" "<<local_no[2]<<"local offset for x,y,z"<<endl;
	cout<<local_i_start[0]<<" "<<local_i_start[1]<<" "<<local_i_start[2]<<"local start for x,y,z"<<endl;
	cout<<local_o_start[0]<<" "<<local_o_start[1]<<" "<<local_o_start[2]<<"local start for kx,ky,kz"<<endl;}

// note that every local_ni should >=2 !!!         		
     LOCAL_SIZE=local_ni[0]*local_ni[1]*local_ni[2];

  /* Allocate memory */
  in  = pfft_alloc_complex(alloc_local);
  out = pfft_alloc_complex(alloc_local);


  /* Plan parallel forward FFT */
  plan_forw = pfft_plan_dft_3d(
      n, in, in, comm_cart_2d, PFFT_FORWARD, PFFT_TRANSPOSED_NONE| PFFT_MEASURE);
  
  /* Plan parallel backward FFT */
  plan_back = pfft_plan_dft_3d(
      n, in, in, comm_cart_2d, PFFT_BACKWARD, PFFT_TRANSPOSED_NONE| PFFT_MEASURE);

  /* Initialize input with random numbers */
  pfft_init_input_c2c_3d(n, local_ni, local_i_start,
      in);


//*************************************************************************
//// origin code, nothing changed below.
//
	initial();
	initw();
	double ax,bx,cx;
	//readw();
	

	ax=1.3/double(SIDEx);
	bx=1.6/double(SIDEx);
	cx=1.9/double(SIDEx);
	
	
	double minFE;
	double minx;
	//minFE=golden(ax,bx,cx,cal_scft,tol,&minx);
        minFE=brent(ax,bx,cx,cal_scft,tol,&minx);	
	

	cal_scft_final();
	
	clean();

//	fftwnd_mpi_destroy_plan (plan3d);
//      fftwnd_mpi_destroy_plan (plan3d_bak);
//
	MPI_Finalize ();
//
        double speed=(double)(clock()-start)/CLOCKS_PER_SEC;
	cout<<speed<<"seconds"<<endl;		
	return 0;
 }
