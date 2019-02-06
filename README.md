# simplex-gpu
The simplex algorithm, implemented in Cuda and for CPU.
Uses Eigen for CPU implementation; Cuda (with some use of Thrust) for GPU code.

Implements and verifies the findings of
> M. E. Lalami, V. Boyer and D. El-Baz “Efficient Implementation of the Simplex Method on a CPU-GPU System”.
> In: 2011 IEEE International Symposium on Parallel and Distributed Processing Workshops and PhD Forum.
> May 2011, pp. 1999–2006.
> [doi:10.1109/IPDPS.2011.362](http://dx.doi.org/10.1109/IPDPS.2011.362)

Speedup is maximum 463x when comparing to the single-threaded CPU implementation,
	at the maximum tableau size that fits into the GPU memory (16384x16384, 2GiB).
As reported by the paper, it was faster to
	perform the leaving and entering variable selections (reduction operations) on the CPU.
	
## Repository Layout
High-level calls to algorithm components can be found in [src/algo](./src/algo),
	and component implementation can be found in [src/impl](./src/impl).
	
## Dependencies
 - Install Thrust and Cuda
 - Download and unpack Eigen in [include](./include) according to the version in the symlink there.
