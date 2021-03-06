#include "gpu_impl.hpp"

#include <impl/reduction_kernel.cu>

#include <cassert>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <limits>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>

__global__ void kernel1(double* SimplexTableau, int width, double* theta, double* columnK, int k);
__global__ void kernel2(double* SimplexTableau, int width, const double* columnK, int r);
__global__ void kernel3(double* SimplexTableau, int width, int height, const double* columnK, int r);
__global__ void kernel4(double* SimplexTableau, int width, const double* columnK, int k, int r);

namespace {
	template<typename T>
	struct my_numeric_limits;

	template<> struct my_numeric_limits<double> {
		__host__ __device__ static double max() { return 1.0/0; }
	};

	template<> struct my_numeric_limits<float> {
		__host__ __device__ static float max() { return 1.0f/0; }
	};
}

namespace simplex {
namespace gpu {

template<typename NUMERIC>
struct FEVComputation {
	typedef ptrdiff_t Index;
	typedef thrust::tuple<NUMERIC,Index> Element;

	struct Direct {
		__host__ __device__
		static Element identity() {
			return thrust::make_tuple(my_numeric_limits<NUMERIC>::max(), (Index)-1);
		}

		struct transformer : thrust::unary_function<Element, Element> {
			__host__ __device__
			Element operator()(const Element& v) const {
				return v;
			}
		};

		struct reducer : thrust::binary_function<Element, Element, Element> {
			__host__ __device__
			Element operator()(const Element& lhs, const Element& rhs) const {
				return thrust::get<0>(lhs) < thrust::get<0>(rhs) ? lhs : rhs;
			}
		};
	};
};

enum FEVMode {
	FEV_THRUST,
	// FEV_REDUCE_K0,
};

#ifndef FEV_MODE_DEFAULT
	#define FEV_MODE_DEFAULT FEV_THRUST
#endif
static const bool fev_mode = FEV_MODE_DEFAULT;

#define FEV_BLOCK_HEIGHT ((int)16)
ptrdiff_t find_entering_variable(const util::PointerAndSize<double>& first_row) {
	typedef FEVComputation<double> FEVComp;
	typedef FEVComp::Index Index;

	ptrdiff_t col_index = -1;
	if (fev_mode == FEV_THRUST) {
		FEVComp::Element reduced = thrust::transform_reduce(
			thrust::cuda::par,
			thrust::make_zip_iterator(thrust::make_tuple(
				first_row.begin() + 1,
				thrust::counting_iterator<ptrdiff_t>(1)
			)),
			thrust::make_zip_iterator(thrust::make_tuple(
				first_row.end(),
				thrust::counting_iterator<ptrdiff_t>(-1)
			)),
			FEVComp::Direct::transformer(),
			FEVComp::Direct::identity(),
			FEVComp::Direct::reducer()
		);

		if (thrust::get<0>(reduced) < 0) {
			col_index = thrust::get<1>(reduced);
		}
	}

	if (col_index > 0) {
		std::cout << "      found entering variable: var" << col_index << '\n';
	} else {
		std::cout << "      did not find a entering variable\n";
	}

	return col_index;
}

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
	typedef ptrdiff_t Index;
	typedef thrust::tuple<NUMERIC,NUMERIC,Index> NumericTuple;

	struct Direct {
		__host__ __device__
		static NumericTuple identity() {
			return thrust::make_tuple(my_numeric_limits<NUMERIC>::max(), 0.0f, (Index)-1);
		}

		struct transformer : thrust::unary_function<NumericTuple, NumericTuple> {
			__host__ __device__
			NumericTuple operator()(const NumericTuple& v) const {
				if (thrust::get<1>(v) > 0) {
					return v;
				} else {
					return identity();
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

	struct Indirect {
		const NUMERIC* const theta_values;
		const NUMERIC* const entering_column;

		Indirect(
			const ThetaValuesAndEnteringColumn<NUMERIC>& tv_and_centering_column
		)
			: theta_values(tv_and_centering_column.theta_values.data())
			, entering_column(tv_and_centering_column.entering_column.data())
		{ }

		__host__ __device__
		Index identity() const { return -1; }

		__host__ __device__
		Index should_reduce(const Index& v) const {
			if (v != identity() && entering_column[v] > 0) {
				return true;
			} else {
				// printf("ignoring %ld\n", v);
				return false;
			}
		}

		__host__ __device__
		Index transform(const Index& v) const {
			if (should_reduce(v)) {
				return v;
			} else {
				return identity();
			}
		}

		__host__ __device__
		Index reduce(const Index& lhs, const Index& rhs) const {
			if (lhs == identity()) {
				// printf("chooose: lhs=%ld, rhs=%d; RHS\n", lhs, rhs);
				return rhs;
			} else if (rhs == identity()) {
				// printf("chooose: lhs=%ld, rhs=%d; LHS\n", lhs, rhs);
				return lhs;
			} else {
				if (theta_values[lhs] < theta_values[rhs]) {
					// printf("chooose: lhs=%ld, rhs=%d; LHS\n", lhs, rhs);
					return lhs;
				} else {
					// printf("chooose: lhs=%ld, rhs=%d; RHS\n", lhs, rhs);
					return rhs;
				}
			}
		}
	};
};

enum FLVMode {
	FLV_THRUST,
	FLV_REDUCE_K0,
	FLV_REDUCE_K6,
};

#ifndef FLV_MODE_DEFAULT
	#define FLV_MODE_DEFAULT FLV_THRUST
#endif
static const bool flv_mode = FLV_MODE_DEFAULT;

#define FLV_BLOCK_HEIGHT ((int)16)
ptrdiff_t find_leaving_variable(const ThetaValuesAndEnteringColumn<double>& tvals_and_centering) {
	typedef FLVComputation<double> FLVComp;
	typedef FLVComp::Index Index;

	Index row_index = -1;
	if (flv_mode == FLV_THRUST) {
		row_index = thrust::get<2>(thrust::transform_reduce(
			thrust::cuda::par,
			thrust::make_zip_iterator(thrust::make_tuple(
				tvals_and_centering.theta_values.begin() + 1,
				tvals_and_centering.entering_column.begin() + 1,
				thrust::counting_iterator<Index>(1)
			)),
			thrust::make_zip_iterator(thrust::make_tuple(
				tvals_and_centering.theta_values.end(),
				tvals_and_centering.entering_column.end(),
				thrust::counting_iterator<Index>(-1)
			)),
			FLVComp::Direct::transformer(),
			FLVComp::Direct::identity(),
			FLVComp::Direct::reducer()
		));
	} else {
		const ptrdiff_t tab_height = tvals_and_centering.theta_values.size();
		assert(tab_height % FLV_BLOCK_HEIGHT == 0);
		assert(tvals_and_centering.entering_column.size() == tab_height);

		thrust::device_vector<Index> row_index_storage(1);

		static const int numBlocks = tab_height/FLV_BLOCK_HEIGHT;
		static const int threadsPerBlock = FLV_BLOCK_HEIGHT;
		static const int smemSize = (threadsPerBlock <= 32) ? 2 * threadsPerBlock * sizeof(Index) : threadsPerBlock * sizeof(Index);

		if (flv_mode == FLV_REDUCE_K0) {
			reductions::reduce0<Index><<< numBlocks, threadsPerBlock, smemSize >>>(
				thrust::counting_iterator<Index>(1),
				row_index_storage.data().get(),
				tab_height - 1,
				FLVComp::Indirect(tvals_and_centering)
			);
			row_index = *row_index_storage.data();
		} else if (flv_mode == FLV_REDUCE_K6) {
			reductions::reduce6<Index, threadsPerBlock, true><<< numBlocks, threadsPerBlock, smemSize >>>(
				thrust::counting_iterator<Index>(1),
				row_index_storage.data().get(),
				tab_height - 1,
				FLVComp::Indirect(tvals_and_centering)
			);
			row_index = *row_index_storage.data();
		}
	}

	if (row_index >= 0) {
		// emulate logging indentation
		std::cout << "      found leaving variable: var" << row_index << '\n';
	} else {
		std::cout << "      did not find a leaving variable\n";
	}

	return row_index;
}

#define K2_BLOCK_WIDTH ((int)512)
void update_leaving_row(Tableau<double>& tab, const util::PointerAndSize<double>& entering_column, VariablePair leaving_and_entering) {

	int numBlocks = tab.width()/K2_BLOCK_WIDTH;
	int threadsPerBlock = K2_BLOCK_WIDTH;
	kernel2<<<numBlocks, threadsPerBlock>>>(tab.data(), tab.width(), entering_column.data(), leaving_and_entering.leaving.getValue());
}

#define K3_BLOCK_WIDTH ((int)128)
#define K3_BLOCK_HEIGHT ((int)128)
void update_rest_of_basis(Tableau<double>& tab, const util::PointerAndSize<double>& entering_column, VariableIndex leaving) {
	assert(tab.height() % K3_BLOCK_HEIGHT == 0);
	assert(tab.width()  % K3_BLOCK_WIDTH  == 0);

	dim3 numBlocks(tab.width()/K3_BLOCK_WIDTH, tab.height()/K3_BLOCK_HEIGHT);
	dim3 threadsPerBlock(K3_BLOCK_WIDTH, K3_BLOCK_HEIGHT);

	kernel3<<<numBlocks, threadsPerBlock>>>(tab.data(), tab.width(), tab.height(), entering_column.data(), leaving.getValue());
}

#define K4_BLOCK_HEIGHT ((int)512)
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
	int height_moduli[4] = {K1_BLOCK_HEIGHT, FLV_BLOCK_HEIGHT,  K3_BLOCK_HEIGHT, K4_BLOCK_HEIGHT, };
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