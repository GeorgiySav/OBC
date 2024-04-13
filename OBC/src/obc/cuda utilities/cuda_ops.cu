#include "cuda_ops.h"

#include <cstdio>
#include <cstdlib>

#include "cublas_utils.h"

namespace obc {
	namespace cuda {
		cublasHandle_t cublasH = NULL;
		cudaStream_t stream = NULL;

        void Init() {
			CUDA_CHECK(cudaSetDevice(0));

			/* step 1: create cublas handle, bind a stream */
			CUBLAS_CHECK(cublasCreate(&cublasH));

			CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
			CUBLAS_CHECK(cublasSetStream(cublasH, stream));
		}	

		void Sync() {
			CUDA_CHECK(cudaStreamSynchronize(stream));
		}

		void MatrixVecMul(const double* A, const int m, const int n, bool transpose,
			const double* x,
			double* y) {

			const int lda = m;
			double alpha = 1.0;
			double beta = 0.0;

			cublasOperation_t transa = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

			CUBLAS_CHECK(
				cublasDgemv(cublasH, transa, m, n, &alpha, A, lda, x, 1, &beta, y, 1));
		}

        void MatrixVecMul(const std::vector<double>& A, const int m, const int n, bool transpose,
			const std::vector<double>& x, 
			std::vector<double>& y) {

			double* d_A = nullptr;
			double* d_x = nullptr;
			double* d_y = nullptr;

			/* step 2: copy data to device */
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(double) * A.size()));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_x), sizeof(double) * x.size()));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y), sizeof(double) * y.size()));

			CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice,
                				stream));
			CUDA_CHECK(cudaMemcpyAsync(d_x, x.data(), sizeof(double) * x.size(), cudaMemcpyHostToDevice,
                				stream));

			/* step 3: compute */
			MatrixVecMul(d_A, m, n, transpose, d_x, d_y);	
			
			CUDA_CHECK(cudaMemcpyAsync(y.data(), d_y, sizeof(double) * y.size(), cudaMemcpyDeviceToHost,
                				stream));

			CUDA_CHECK(cudaStreamSynchronize(stream));

			/* free resources */
			CUDA_CHECK(cudaFree(d_A));
			CUDA_CHECK(cudaFree(d_x));
			CUDA_CHECK(cudaFree(d_y));
        }

		void MatrixMatrixMul(
			const int m, const int k, const int n,
			const double* A, bool transposeA,
			const double* B, bool transposeB,
			double* C) {

			const int lda = transposeA ? k : m;
			const int ldb = transposeB ? n : k;
			const int ldc = m;

			double alpha = 1.0;
			double beta = 0.0;

			cublasOperation_t transa = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
			cublasOperation_t transb = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

			CUBLAS_CHECK(
				cublasDgemm(cublasH, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));

		}

		void MatrixMatrixMul(
			const int m, const int k, const int n,
			const std::vector<double>& A, bool transposeA,
			const std::vector<double>& B, bool transposeB,
			std::vector<double>& C) {

			double* d_A = nullptr;
			double* d_B = nullptr;
			double* d_C = nullptr;

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(double) * A.size()));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(double) * B.size()));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), sizeof(double) * C.size()));

			CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice,
				stream));
			CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(double) * B.size(), cudaMemcpyHostToDevice,
				stream));

			MatrixMatrixMul(m, k, n, d_A, transposeA, d_B, transposeB, d_C);

			CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(double) * C.size(), cudaMemcpyDeviceToHost, stream));

			CUDA_CHECK(cudaStreamSynchronize(stream));

			CUDA_CHECK(cudaFree(d_A));
			CUDA_CHECK(cudaFree(d_B));
			CUDA_CHECK(cudaFree(d_C));
		}

		__global__ void VecVecAddKernel(const double* A, const double* B, double* C, int N) {
			// Get our global thread ID
			int id = blockIdx.x * blockDim.x + threadIdx.x;

			// Make sure we do not go out of bounds
			if (id < N)
				C[id] = A[id] + B[id];
		}

		void VecVecAdd(const double* A, const double* B, double* C, int N) {
			const int threads_per_block = 256;
			const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

			VecVecAddKernel<<<num_blocks, threads_per_block>>>(A, B, C, N);
			cudaDeviceSynchronize();
		}

		void VecVecAdd(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
			const int N = A.size();
			const int size = N * sizeof(double);

			double* d_A = nullptr;
			double* d_B = nullptr;
			double* d_C = nullptr;

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), size));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), size));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), size));

			CUDA_CHECK(cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice));

			VecVecAdd(d_A, d_B, d_C, N);

			CUDA_CHECK(cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost));

			CUDA_CHECK(cudaFree(d_A));
			CUDA_CHECK(cudaFree(d_B));
			CUDA_CHECK(cudaFree(d_C));
		}

		__global__ void ScalarAddKernel(const double* A, double scalar, double* y, int N) {
			// Get our global thread ID
			int id = blockIdx.x * blockDim.x + threadIdx.x;

			// Make sure we do not go out of bounds
			if (id < N) {
				y[id] = y[id] + (A[id] * scalar);
			}
		}

		void ScalarAdd(const double* A, double scalar, double* y, int N) {
			const int threads_per_block = 256;
			const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

			ScalarAddKernel<<<num_blocks, threads_per_block>>>(A, scalar, y, N);
			cudaDeviceSynchronize();
		}

		void ScalarAdd(const std::vector<double>& A, double scalar, std::vector<double>& y) {
			const int N = A.size();
			const int size = N * sizeof(double);

			double* d_A = nullptr;
			double* d_y = nullptr;

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), size));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y), size));

			CUDA_CHECK(cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(d_y, y.data(), size, cudaMemcpyHostToDevice));

			ScalarAdd(d_A, scalar, d_y, N);

			CUDA_CHECK(cudaMemcpy(y.data(), d_y, size, cudaMemcpyDeviceToHost));

			CUDA_CHECK(cudaFree(d_A));
			CUDA_CHECK(cudaFree(d_y));
		}

		__global__ void ElementwiseMulKernel(const double* A, const double* B, double* C, int N) {
			// Get our global thread ID
			int id = blockIdx.x * blockDim.x + threadIdx.x;

			// Make sure we do not go out of bounds
			if (id < N) {
				C[id] = A[id] * B[id];
			}
		}

		void ElementwiseMul(const double* A, const double* B, double* C, int N) {
			const int threads_per_block = 256;
			const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

			ElementwiseMulKernel<<<num_blocks, threads_per_block>>>(A, B, C, N);
			cudaDeviceSynchronize();
		}

		void ElementwiseMul(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
			const int N = A.size();
			const int size = N * sizeof(double);

			double* d_A = nullptr;
			double* d_B = nullptr;
			double* d_C = nullptr;

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), size));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), size));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), size));

			CUDA_CHECK(cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice));

			ElementwiseMul(d_A, d_B, d_C, N);

			CUDA_CHECK(cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost));

			CUDA_CHECK(cudaFree(d_A));
			CUDA_CHECK(cudaFree(d_B));
			CUDA_CHECK(cudaFree(d_C));
		}

		__global__ void ValidCrossCorrelateKernel(const double* A, int a_offset, int a_height, int a_width,
												  const double* B, int b_offset, int b_height, int b_width, bool rot180,
														double* C, int c_offset, int c_height, int c_width) {
			int i = threadIdx.x + blockIdx.x * blockDim.x;
			int j = threadIdx.y + blockIdx.y * blockDim.y;

			int c_index = c_offset + (j * c_width + i);

			for (int x = 0; x < b_width; x++) {
				for (int y = 0; y < b_height; y++) {
					int a_index = a_offset + ((j + y) * a_width + (i + x));
					int b_index = y * b_width + x;
					if (rot180)	
						b_index = (b_height * b_width) - 1 - b_index;
					b_index += b_offset;


					C[c_index] += A[a_index] * B[b_index];
				}
			}
		}

		__global__ void FullCrossCorrelateKernel(const double* A, int a_offset, int a_height, int a_width,
												 const double* B, int b_offset, int b_height, int b_width, bool rot180,
													   double* C, int c_offset, int c_height, int c_width) {
			int adj_i = threadIdx.x + blockIdx.x * blockDim.x;
			int adj_j = threadIdx.y + blockIdx.y * blockDim.y;

			int i = adj_i - (b_width - 1);
			int j = adj_j - (b_height - 1);

			int c_index = c_offset + (adj_j * c_width + adj_i);

			for (int x = 0; x < b_width; x++) {

				if (i + x < 0 || i + x >= a_width)
					continue;

				for (int y = 0; y < b_height; y++) {

					if (j + y < 0 || j + y >= a_height)
						continue;

					int a_index = a_offset + ((j + y) * a_width + (i + x));
					int b_index = y * b_width + x;
					if (rot180)
						b_index = (b_height * b_width) - 1 - b_index;
					b_index += b_offset;

					C[c_index] += A[a_index] * B[b_index];
				}
			}
		}

		void CrossCorrelate(
			const double* A, int a_offset, int a_height, int a_width,
			const double* B, int b_offset, int b_height, int b_width, bool rot180,
			double* C, int c_offset, int c_height, int c_width,
			bool full) {

			dim3 grid(1, 1, 1);
			dim3 block;

			if (full) {
				block = dim3(a_width - (1 - b_width), a_height - (1 - b_height), 1);
				FullCrossCorrelateKernel<<<grid, block>>>(
					A, a_offset, a_height, a_width, 
					B, b_offset, b_height, b_width, rot180, 
					C, c_offset, c_height, c_width);
			}
			else {
				block = dim3(a_width - b_width + 1, a_height - b_height + 1, 1);
				ValidCrossCorrelateKernel<<<grid, block>>>(
					A, a_offset, a_height, a_width,
					B, b_offset, b_height, b_width, rot180,
					C, c_offset, c_height, c_width);
			}
			cudaDeviceSynchronize();

		}

		void CrossCorrelate(
			const std::vector<double>& A, int a_offset, int a_height, int a_width,
			const std::vector<double>& B, int b_offset, int b_height, int b_width, bool rot180,
			std::vector<double>& C, int c_offset, int c_height, int c_width,
			bool full) {

			double* d_A = nullptr;
			double* d_B = nullptr;
			double* d_C = nullptr;

			const int a_size = sizeof(double) * (a_height * a_width);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), a_size));
			const int b_size = sizeof(double) * (b_height * b_width);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), b_size));
			const int c_size = sizeof(double) * (c_height * c_width);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), c_size));
		
			CUDA_CHECK(cudaMemcpy(d_A, A.data() + a_offset, a_size, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(d_B, B.data() + b_offset, b_size, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(d_C, C.data() + c_offset, c_size, cudaMemcpyHostToDevice));

			CrossCorrelate(
				d_A, 0, a_height, a_width,
				d_B, 0, b_height, b_width, rot180,
				d_C, 0, c_height, c_width,
				false);

			CUDA_CHECK(cudaMemcpy(C.data() + c_offset, d_C, c_size, cudaMemcpyDeviceToHost));

			CUDA_CHECK(cudaFree(d_A));
			CUDA_CHECK(cudaFree(d_B));
			CUDA_CHECK(cudaFree(d_C));
		}

		void Copy(const double* A, double* B, int N) {
			CUBLAS_CHECK(cublasDcopy(cublasH, N, A, 1, B, 1));
		}

		void Copy(const std::vector<double>& A, std::vector<double>& B) {
			double* d_A = nullptr;
			double* d_B = nullptr;

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(double) * A.size()));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(double) * B.size()));

			CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice,
				stream));
			CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(double) * B.size(), cudaMemcpyHostToDevice,
				stream));

			Copy(d_A, d_B, A.size());

			CUDA_CHECK(cudaMemcpyAsync(B.data(), d_B, sizeof(double) * B.size(), cudaMemcpyDeviceToHost,
				stream));

			CUDA_CHECK(cudaStreamSynchronize(stream));

			CUDA_CHECK(cudaFree(d_A));
			CUDA_CHECK(cudaFree(d_B));
		}

		// template instantiation
		template void ApplyFunc<FunctionType::kSigmoid>(double* A, int N);
		template void ApplyFunc<FunctionType::kSigmoid>(std::vector<double>& A);
		template void ApplyFunc<FunctionType::kSigmoid>(const double* A, double* y, int N);
		template void ApplyFunc<FunctionType::kSigmoid>(const std::vector<double>& A, std::vector<double>& y);

		template void ApplyFunc<FunctionType::kSigmoidPrime>(double* A, int N);
		template void ApplyFunc<FunctionType::kSigmoidPrime>(std::vector<double>& A);
		template void ApplyFunc<FunctionType::kSigmoidPrime>(const double* A, double* y, int N);
		template void ApplyFunc<FunctionType::kSigmoidPrime>(const std::vector<double>& A, std::vector<double>& y);

		template void ApplyFunc<FunctionType::kReLu>(double* A, int N);
		template void ApplyFunc<FunctionType::kReLu>(std::vector<double>& A);
		template void ApplyFunc<FunctionType::kReLu>(const double* A, double* y, int N);
		template void ApplyFunc<FunctionType::kReLu>(const std::vector<double>& A, std::vector<double>& y);

		template void ApplyFunc<FunctionType::kReLuPrime>(double* A, int N);
		template void ApplyFunc<FunctionType::kReLuPrime>(std::vector<double>& A);
		template void ApplyFunc<FunctionType::kReLuPrime>(const double* A, double* y, int N);
		template void ApplyFunc<FunctionType::kReLuPrime>(const std::vector<double>& A, std::vector<double>& y);

		template void ApplyFunc<FunctionType::kTanh>(double* A, int N);
		template void ApplyFunc<FunctionType::kTanh>(std::vector<double>& A);
		template void ApplyFunc<FunctionType::kTanh>(const double* A, double* y, int N);
		template void ApplyFunc<FunctionType::kTanh>(const std::vector<double>& A, std::vector<double>& y);

		template void ApplyFunc<FunctionType::kTanhPrime>(double* A, int N);
		template void ApplyFunc<FunctionType::kTanhPrime>(std::vector<double>& A);
		template void ApplyFunc<FunctionType::kTanhPrime>(const double* A, double* y, int N);
		template void ApplyFunc<FunctionType::kTanhPrime>(const std::vector<double>& A, std::vector<double>& y);

		__device__ double Sigmoid(double x) {
			return 1 / (1 + exp(-x));
		}
		__device__ double SigmoidPrime(double x) {
			double s = Sigmoid(x);
			return s * (1 - s);
		}
		__device__ double ReLu(double x) {
			return x > 0 ? x : 0;
		}
		__device__ double ReLuPrime(double x) {
			return x > 0 ? 1 : 0;
		}
		__device__ double Tanh(double x) {
			return tanh(x);
		}
		__device__ double TanhPrime(double x) {
			double t = tanh(x);
			return 1 - t * t;
		}

		template <FunctionType func>
		__global__ void ApplyFuncKernel(double* A, int N) {
			// Get our global thread ID
			int id = blockIdx.x * blockDim.x + threadIdx.x;

			// Make sure we do not go out of bounds
			if (id < N) {
				if constexpr (func == FunctionType::kSigmoid) {
					A[id] = Sigmoid(A[id]);
				}
				else if constexpr (func == FunctionType::kSigmoidPrime) {
					A[id] = SigmoidPrime(A[id]);
				}
				else if constexpr (func == FunctionType::kReLu) {
					A[id] = ReLu(A[id]);
				}	
				else if constexpr (func == FunctionType::kReLuPrime) {
					A[id] = ReLuPrime(A[id]);
				}
				else if constexpr (func == FunctionType::kTanh) {
					A[id] = Tanh(A[id]);
				}
				else if constexpr (func == FunctionType::kTanhPrime) {
					A[id] = TanhPrime(A[id]);
				}
			}
		}
		template<FunctionType func>
		void ApplyFunc(double* A, int N) {
			const int threads_per_block = 256;
			const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

			ApplyFuncKernel<func><<<num_blocks, threads_per_block>>>(A, N);
			cudaDeviceSynchronize();
		}
		template <FunctionType func>
		void ApplyFunc(std::vector<double>& A) {
			const int N = A.size();
			const int size = N * sizeof(double);

			double* d_A = nullptr;

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), size));

			CUDA_CHECK(cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice));

			ApplyFunc<func>(d_A, N);

			CUDA_CHECK(cudaMemcpy(A.data(), d_A, size, cudaMemcpyDeviceToHost));

			CUDA_CHECK(cudaFree(d_A));
		}

		template <FunctionType func>
		__global__ void ApplyFuncKernel(const double* A, double* y, int N) {
			// Get our global thread ID
			int id = blockIdx.x * blockDim.x + threadIdx.x;

			// Make sure we do not go out of bounds
			if (id < N) {
				if constexpr (func == FunctionType::kSigmoid) {
					y[id] = Sigmoid(A[id]);
				}
				else if constexpr (func == FunctionType::kSigmoidPrime) {
					y[id] = SigmoidPrime(A[id]);
				}
				else if constexpr (func == FunctionType::kReLu) {
					y[id] = ReLu(A[id]);
				}
				else if constexpr (func == FunctionType::kReLuPrime) {
					y[id] = ReLuPrime(A[id]);
				}
				else if constexpr (func == FunctionType::kTanh) {
					y[id] = Tanh(A[id]);
				}
				else if constexpr (func == FunctionType::kTanhPrime) {
					y[id] = TanhPrime(A[id]);
				}
			}
		}
		template <FunctionType func>
		void ApplyFunc(const double* A, double* y, int N) {
			const int threads_per_block = 256;
			const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

			ApplyFuncKernel<func><<<num_blocks, threads_per_block>>>(A, y, N);
			cudaDeviceSynchronize();
		}
		template <FunctionType func>
		void ApplyFunc(const std::vector<double>& A, std::vector<double>& y) {
			const int N = A.size();
			const int size = N * sizeof(double);

			double* d_A = nullptr;
			double* d_y = nullptr;

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), size));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y), size));

			CUDA_CHECK(cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice));

			ApplyFunc<func>(d_A, d_y, N);

			CUDA_CHECK(cudaMemcpy(y.data(), d_y, size, cudaMemcpyDeviceToHost));

			CUDA_CHECK(cudaFree(d_A));
			CUDA_CHECK(cudaFree(d_y));
		}

        void Shutdown() {
            CUBLAS_CHECK(cublasDestroy(cublasH));
            CUDA_CHECK(cudaStreamDestroy(stream));
            CUDA_CHECK(cudaDeviceReset());
        }
	}
}
