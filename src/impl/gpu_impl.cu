#include "gpu_impl.hpp"

#include <cassert>
#include <cstdio>
#include <iterator>

__global__ void kernel1(double* SimplexTableau, int width, double* theta, double* columnK, int k);
__global__ void kernel2(double* SimplexTableau, int width, const double* columnK, int r);
__global__ void kernel3(double* SimplexTableau, int width, int height, const double* columnK, int r);
__global__ void kernel4(double* SimplexTableau, int width, const double* columnK, int k, int r);

namespace simplex {
namespace gpu {

// Tableau<double> create_tableau(const Problem& problem_stmt) {
// 	(void)problem_stmt;

// 	Tableau<double> result (
// 		0,
// 		4,
// 		5
// 	);

// 	return result;
// }

#define K1_BLOCK_HEIGHT ((int)8)
ThetaValuesAndEnteringColumn<double> get_theta_values_and_entering_column(const Tableau<double>& tab, VariableIndex entering) {
	ThetaValuesAndEnteringColumn<double> result (
		tab.height()
	);

	// for (int irow = 0; irow < tab.height(); ++irow) {
	// 	const auto& val_at_entering = tab.at(irow, entering);
	// 	result.entering_column.at((std::size_t)irow) = val_at_entering;
	// 	result.theta_values.at((std::size_t)irow) = tab.at(irow, 0)/val_at_entering;
	// }

	// // std::cout << "theta_values computed: ";
	// // // util::print_container(std::cout, result.theta_values);
	// // std::cout << "\nentering_column copied: ";
	// // // util::print_container(std::cout, result.entering_column);
	// // std::cout << '\n';

	return result;
}

VariableIndex find_leaving_variable(const ThetaValuesAndEnteringColumn<double>& tvals_and_centering) {
	// double lowest_theta_value = std::numeric_limits<double>::max();
	VariableIndex result;

	// for (int irow = 1; irow < (int)tvals_and_centering.theta_values.size(); ++irow) {
	// 	const auto& theta_val = tvals_and_centering.theta_values.at((std::size_t)irow);
	// 	const auto& tab_val = tvals_and_centering.entering_column.at((std::size_t)irow);
	// 	if (tab_val > 0 && (!result || theta_val < lowest_theta_value)) {
	// 		lowest_theta_value = theta_val;
	// 		result = util::make_id<VariableIndex>(irow);
	// 	}
	// }

	// if (result) {
	// 	// std::cout << "found leaving variable: " << *result << '\n';
	// } else {
	// 	// std::cout << "did not find a leaving variable\n";
	// }

	return result;
}

#define K2_BLOCK_WIDTH ((int)8)
void update_leaving_row(Tableau<double>& tab, const util::PointerAndSize<double>& entering_column, VariablePair leaving_and_entering) {

	int numBlocks = tab.width()/K2_BLOCK_WIDTH;
	int threadsPerBlock = K2_BLOCK_WIDTH;
	kernel2<<<numBlocks, threadsPerBlock>>>(tab.data(), tab.width(), entering_column.data(), leaving_and_entering.leaving.getValue());
}

#define K3_BLOCK_WIDTH ((int)8)
#define K3_BLOCK_HEIGHT ((int)4)
void update_rest_of_basis(Tableau<double>& tab, const util::PointerAndSize<double>& entering_column, VariableIndex leaving) {
	assert(tab.height() % K3_BLOCK_HEIGHT == 0);
	assert(tab.width()  % K3_BLOCK_WIDTH  == 0);

	dim3 numBlocks(tab.width()/K3_BLOCK_WIDTH, tab.height()/K3_BLOCK_HEIGHT);
	dim3 threadsPerBlock(K3_BLOCK_WIDTH, K3_BLOCK_HEIGHT);

	kernel3<<<numBlocks, threadsPerBlock>>>(tab.data(), tab.width(), tab.height(), entering_column.data(), leaving.getValue());
}

#define K4_BLOCK_HEIGHT ((int)8)
void update_entering_column(Tableau<double>& tab, const util::PointerAndSize<double>& entering_column, VariablePair leaving_and_entering) {

	int numBlocks = tab.height()/K4_BLOCK_HEIGHT;
	int threadsPerBlock = K4_BLOCK_HEIGHT;

	kernel4<<<numBlocks, threadsPerBlock>>>(tab.data(), tab.width(), entering_column.data(), leaving_and_entering.entering.getValue(), leaving_and_entering.leaving.getValue());
}


template<typename INTEGRAL>
static INTEGRAL greatest_common_multiple(INTEGRAL a, INTEGRAL b) {
	while (true) {
		if (a == 0) return b;
		b %= a;
		if (b == 0) return a;
		a %= b;
	}
}

template<typename IT>
static int least_common_multiple(IT&& nums_begin, IT&& nums_end) {
	int result = *nums_begin;
	if (result == 0) { return 0; }

	for (IT it = nums_begin; it != nums_end; ++it) {
		int num = *it;
		if (num == 0) { return 0; }
		result = (result*num)/greatest_common_multiple(num, result);
	}

	return result;
}

ProblemContraints problem_constraints() {
	int height_moduli[3] = {K1_BLOCK_HEIGHT, K3_BLOCK_HEIGHT, K4_BLOCK_HEIGHT, };
	int  width_moduli[2] = {K2_BLOCK_WIDTH,  K3_BLOCK_WIDTH, };
	return {
		least_common_multiple(&height_moduli[0], &height_moduli[sizeof(height_moduli)/sizeof(height_moduli[0])]),
		least_common_multiple(&width_moduli[0], &width_moduli[sizeof(width_moduli)/sizeof(width_moduli[0])]),
	};
}

} // end namespace simplex
} // end namespace gpu


// kernel codes

__global__ void kernel1(double* SimplexTableau, int width, double* theta, double* columnK, int k) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	double w = SimplexTableau[idx*width + k];
	/*Copy the weights of entering index k*/
	columnK[idx] = w;
	theta[idx] = SimplexTableau[idx*width +1]/w;
}

__global__ void kernel2(double* SimplexTableau, int width, const double* columnK, int r) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ double w;

	if (threadIdx.x == 0) {
		w = columnK[r];
		// printf("index: %d\n", r);
		// printf("denom: %f\n", w);
		// printf("width: %d\n", width);
	}
	__syncthreads();


	// printf("idx: %d %f\n", idx, SimplexTableau[r*width + idx]);
	SimplexTableau[r*width + idx] = SimplexTableau[r*width + idx]/w;
}

__global__ void kernel3(double* SimplexTableau, int width, int height, const double* columnK, int r) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int jdx = blockIdx.y * blockDim.y + threadIdx.y;
	__shared__ double w[K3_BLOCK_WIDTH];

	/*Get the column of entering index k in shared memory */
	// The K3_BLOCK_WIDTH test is redundant??
	if(threadIdx.y == 0 && threadIdx.x < K3_BLOCK_WIDTH)
	{
		w[threadIdx.x] = columnK[blockIdx.y * blockDim.y+threadIdx.x];
		// printf("columnK[%d*%d + %d = %d] %f\n", blockIdx.y, blockDim.y, threadIdx.x, blockIdx.y * blockDim.y+threadIdx.x, columnK[blockIdx.y * blockDim.y+threadIdx.x]);
	}
	__syncthreads();
	/*Update the basis except the line r*/
	if(jdx == r) return;
	// printf("w[threadIdx.y]: %f SimplexTableau[r*width + idx]: %f r: %d idx: %d\n", w[threadIdx.y], SimplexTableau[r*width + idx], r, idx);
	SimplexTableau[jdx*width + idx] -= w[threadIdx.y] * SimplexTableau[r*width + idx];
}

__global__ void kernel4(double* SimplexTableau, int width, const double* columnK, int k, int r) {
	int jdx = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ double w;
	/*Get the pivot element : SimplexT ableau[r][k] in the
	shared memory */
	if(threadIdx.x == 0){
		w = columnK[r];
		// printf("r: %d w %f\n", r,w);
	}
	__syncthreads();
	/*Update the column of the entering index k*/
	SimplexTableau[jdx*width + k] = -columnK[jdx]/w;

	/*Update the pivot element SimplexT ableau[r][k]*/
	if(jdx == r) SimplexTableau[jdx*width + k]=1/w;

}