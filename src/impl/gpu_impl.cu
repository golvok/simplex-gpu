#include "gpu_impl.hpp"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <limits>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>

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
	assert(tab.height() % K1_BLOCK_HEIGHT == 0);

	ThetaValuesAndEnteringColumn<double> result (
		tab.height()
	);

	int numBlocks = tab.height()/K1_BLOCK_HEIGHT;
	int threadsPerBlock = K1_BLOCK_HEIGHT;
	kernel1<<<numBlocks, threadsPerBlock>>>(tab.data(), tab.width(), result.theta_values.data(), result.entering_column.data(), entering.getValue());
    cudaDeviceSynchronize();

	return result;
}

template<typename NUMERIC>
struct FLVComputation {
	typedef thrust::tuple<NUMERIC,NUMERIC,ptrdiff_t> NumericTuple;

	__host__ __device__
	static NumericTuple initial_value() {
		return thrust::make_tuple(std::numeric_limits<NUMERIC>::max(), 0.0f, (ptrdiff_t)-1);
	}

	struct transformer : thrust::unary_function<NumericTuple, NumericTuple> {
		__host__ __device__
		NumericTuple operator()(const NumericTuple& v) const {
			if (thrust::get<1>(v) > 0) {
				return v;
			} else {
				return initial_value();
			}
		}
	};

	struct reducer : thrust::binary_function<NumericTuple, NumericTuple, NumericTuple> {
		__host__ __device__
		NumericTuple operator()(const NumericTuple& lhs, const NumericTuple& rhs) const {
			return thrust::get<0>(lhs) < thrust::get<0>(rhs) ? lhs : rhs;
		}
	};
};

ptrdiff_t find_leaving_variable(const ThetaValuesAndEnteringColumn<double>& tvals_and_centering) {
	// double lowest_theta_value = std::numeric_limits<double>::max();
	VariableIndex result;

	ptrdiff_t row_index = thrust::get<2>(thrust::transform_reduce(
		thrust::cuda::par,
		thrust::make_zip_iterator(thrust::make_tuple(
			tvals_and_centering.theta_values.begin() + 1,
			tvals_and_centering.entering_column.begin() + 1,
			thrust::counting_iterator<ptrdiff_t>(1)
		)),
		thrust::make_zip_iterator(thrust::make_tuple(
			tvals_and_centering.theta_values.end(),
			tvals_and_centering.entering_column.end(),
			thrust::counting_iterator<ptrdiff_t>(-1)
		)),
		FLVComputation<double>::transformer(),
		FLVComputation<double>::initial_value(),
		FLVComputation<double>::reducer()
	));

	if (row_index >= 0) {
		// emulate logging indentation
		std::cout << "      found leaving variable: var" << row_index << '\n';
	} else {
		std::cout << "      did not find a leaving variable\n";
	}

	return row_index;
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
	theta[idx] = SimplexTableau[idx*width]/w;
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