/**
 * Implementation of the synchronization method described in:
 *
 * "Multiview Registration via Graph Diffusion of Dual Quaternions"
 * A.Torsello, E.Rodola, and A.Albarelli
 * Proc. CVPR 2011
 *
 * To compile: mex diffuse_dualquat.cpp
 *
 * Emanuele Rodola <emanuele.rodola@usi.ch>
 * USI Lugano, July 2016
 */

#include "dual_quaternions.h"
#include <vector>
#include "mex.h"

using std::vector;
using math3d::matrix3x3;
using math3d::rigid_motion_t;

/**
 * Applies dual quaternion error diffusion on the given view graph.
 *
 * @param pairs    Input collection of pairs of shapes (just the indices).
 * @param pairwise Input collection of pairwise motions.
 * @param diffused Output _absolute_ motions after error diffusion.
 */
void run_diffusion(
	const vector< std::pair<int,int> >& pairs,
	const vector<rigid_motion_t>& pairwise,
    vector<rigid_motion_t>& diffused
){
   const int n_views = diffused.size(); 
   transducer x(n_views);
   
   for (int i=0; i<pairs.size(); ++i)
   {
	   // Add this relative motion to the view graph.
       // Note that we are assigning uniform weights (=1.0) to the different views,
       // but this can be changed if confidence information is available (e.g. coming
       // from the ICP process).

       x.add_transformation(
               pairs[i].first, pairs[i].second,
               math3d::rot_matrix_to_quaternion( pairwise[i].first ),
               pairwise[i].second,
               1.0);
   }

   x.get_estimate();
   mexPrintf("Initial RMSTE %.8f\n", x.rmste());

   x.linear_transduce();
   mexPrintf("Final RMSTE %.8f\n", x.rmste());

   for (int i=0; i<n_views; ++i)
   {
      dual_quaternion tt;
      x.get_position(i,tt);

      matrix3x3<double> R = math3d::quaternion_to_rot_matrix(tt.R);
      point3d T = tt.get_translation();

      diffused[i].first = R;
      diffused[i].second = T;
      math3d::invert(diffused[i].first, diffused[i].second);
   }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
	if (nrhs != 2 || nlhs > 1)
		mexErrMsgTxt("Usage: absolute_diffused = diffuse_dualquat(pairwise_displaced, n_views).");
	
	const int n_pairwise_motions = mxGetN(prhs[0]);
	const int n_views = int(*mxGetPr(prhs[1]));
	
	if (n_pairwise_motions <= 2)
		mexErrMsgTxt("More than 2 pairwise motions should be given as input.");
	
	if (n_views <= 2)
		mexErrMsgTxt("More than 2 views should be given as input.");
	
	if (n_pairwise_motions < n_views)
		mexErrMsgTxt("Each view must be connected to at least one other view.");
		
	vector<rigid_motion_t> pairwise_displaced(n_pairwise_motions);
	vector< std::pair<int,int> > pairs(n_pairwise_motions);

	mexPrintf("%d input views\n", n_views);
	mexPrintf("%d input pairwise motions\n", n_pairwise_motions);
	
	vector<int> is_connected(n_views, 0);

	for (int i=0; i<n_pairwise_motions; ++i)
	{
		mxArray* fld;

		fld = mxGetField(prhs[0], i, "i1");
		if (fld==0)
			mexErrMsgTxt("Parse error for input pairwise motions.");
		pairs[i].first = int(*mxGetPr(fld));
		is_connected.at(pairs[i].first) = 1;
		
		fld = mxGetField(prhs[0], i, "i2");
		if (fld==0)
			mexErrMsgTxt("Parse error for input pairwise motions.");
		pairs[i].second = int(*mxGetPr(fld));
		is_connected.at(pairs[i].second) = 1;
		
		fld = mxGetField(prhs[0], i, "R");
		if (fld==0)
			mexErrMsgTxt("Parse error: rotation matrices must be given.");
		if (mxGetM(fld) != mxGetN(fld) || mxGetM(fld) != 3)
			mexErrMsgTxt("Input rotation matrices must be 3x3.");
		const double* const R_pr = mxGetPr(fld);
		matrix3x3<double>& R = pairwise_displaced[i].first;
		R.r00 = R_pr[0]; R.r01 = R_pr[3]; R.r02 = R_pr[6];
		R.r10 = R_pr[1]; R.r11 = R_pr[4]; R.r12 = R_pr[7];
		R.r20 = R_pr[2]; R.r21 = R_pr[5]; R.r22 = R_pr[8];
		
		fld = mxGetField(prhs[0], i, "T");
		if (fld==0)
			mexErrMsgTxt("Parse error: translation vectors must be given.");
		if ((mxGetM(fld) != 3 && mxGetN(fld) != 1) && (mxGetM(fld) != 1 && mxGetN(fld) != 3))
			mexErrMsgTxt("Input translation vector must be 1x3 or 3x1.");
		const double* const T_pr = mxGetPr(fld);
		point3d& T = pairwise_displaced[i].second;
		T.x = T_pr[0]; T.y = T_pr[1]; T.z = T_pr[2];
	}
	
	int n_connected = 0;
	for (int i=0; i<n_views; ++i)
		if (is_connected[i]) ++n_connected;
	
	if (n_views != n_connected)
		mexErrMsgTxt("Each view must be connected to at least one other view.");
	
	// Run dual quaternion diffusion
	
	vector<rigid_motion_t> out_diffused(n_views);
	run_diffusion(pairs, pairwise_displaced, out_diffused);
	
	// Output as a struct array
	
	mwSize dims[2] = {1, n_views};
	const char *field_names[] = {"R", "T"};
	
	plhs[0] = mxCreateStructArray(2, dims, 2, field_names);
	
	const int R_field = mxGetFieldNumber(plhs[0], "R");
	const int T_field = mxGetFieldNumber(plhs[0], "T");
	
	for (int i=0; i<n_views; ++i)
	{
		mxArray* R_ = mxCreateDoubleMatrix(3,3,mxREAL);
		double* const R_pr = mxGetPr(R_);
		
		const matrix3x3<double>& R = out_diffused[i].first;
		R_pr[0] = R.r00; R_pr[1] = R.r10; R_pr[2] = R.r20;
		R_pr[3] = R.r01; R_pr[4] = R.r11; R_pr[5] = R.r21;
		R_pr[6] = R.r02; R_pr[7] = R.r12; R_pr[8] = R.r22;
		
		mxSetFieldByNumber(plhs[0], i, R_field, R_);
		
		mxArray* T_ = mxCreateDoubleMatrix(1,3,mxREAL);
		double* const T_pr = mxGetPr(T_);
		
		const point3d& T = out_diffused[i].second;
		T_pr[0] = T.x; T_pr[1] = T.y; T_pr[2] = T.z;
		
		mxSetFieldByNumber(plhs[0], i, T_field, T_);
	}
}
