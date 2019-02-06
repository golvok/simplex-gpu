# simplex-gpu
The simplex algorithm, implemented in Cuda and for CPU

Implements and verifies the findings of
> M. E. Lalami, V. Boyer and D. El-Baz “Efficient Implementation of the Simplex Method on a CPU-GPU System”. In: 2011 IEEE International Symposium on Parallel and Distributed Processing Workshops and Phd Forum. May 2011, pp. 1999–2006. doi:10.1109/IPDPS.2011.362

Speedup is maximum 463x when comparing to the single-threaded CPU implementation,
	at the maximum tableau size that fits into the GPU memory (16384x16384, 2GiB).
As reported by the paper, it was faster to
	perform the leaving and entering variable selictions (reduction operations) on the CPU
