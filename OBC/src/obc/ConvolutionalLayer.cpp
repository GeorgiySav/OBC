#include "ConvolutionalLayer.h"

namespace obc {

	void CrossCorrelate(
		const std::vector<double>& A, int a_offset, int a_height, int a_width,
		const std::vector<double>& B, int b_offset, int b_height, int b_width, bool rot180,
		std::vector<double>& C, int c_offset, int c_height, int c_width) {

		for (int i = 0; i <= a_width - b_width; i++) {
			for (int j = 0; j <= a_height - b_height; j++) {

				int c_index = c_offset + (j * c_width + i);

				for (int x = 0; x < b_width; x++) {
					for (int y = 0; y < b_height; y++) {

						int a_index = a_offset + ((j + y) * a_width + (i + x));
						int b_index = (y * b_width + x);
						if (rot180)
							b_index = (b_width * b_height) - 1 - b_index;
						b_index += b_offset;

						C[c_index] += A[a_index] * B[b_index];
					}
				}
			}
		}

	}

	void FullCrossCorrelate(
		const std::vector<double>& A, int a_offset, int a_height, int a_width,
		const std::vector<double>& B, int b_offset, int b_height, int b_width, bool rot180,
		std::vector<double>& C, int c_offset, int c_height, int c_width) {
	
		for (int i = 1 - b_width; i < a_width; i++) {
			for (int j = 1 - b_height; j < a_height; j++) {

				int adj_i = i + (b_width - 1);
				int adj_j = j + (b_height - 1);

				int c_index = c_offset + (adj_j * c_width + adj_i);

				for (int x = 0; x < b_width; x++) {
					for (int y = 0; y < b_height; y++) {

						if (i + x < 0 || j + y < 0
							|| i + x >= a_width || j + y >= a_height)
							continue;

						int a_index = a_offset + ((j + y) * a_width + (i + x));
						int b_index = (y * b_width + x);
						if (rot180)
							b_index = (b_width * b_height) - 1 - b_index;
						b_index += b_offset;

						C[c_index] += A[a_index] * B[b_index];

					}
				}

			}
		}
	
	}

	const std::vector<double>* ConvolutionalLayer::Forward(const std::vector<double>* input) {
		input_ = input;

		std::copy(biases_.begin(), biases_.end(), output_.begin());

		for (int s = 0; s < kernel_sets_; s++) {
			for (int d = 0; d < kernel_depth_; d++) {
				int kernel_offset = GetKernelOffset(s, d);
				int output_offset = GetOutputOffset(d);
				int input_offset = d * input_width_ * input_height_;

				CrossCorrelate(*input_, input_offset, input_height_, input_width_,
								kernels_, kernel_offset, kernel_size_, kernel_size_, false,
								output_, output_offset, output_height_, output_width_);
			}
		}

		return &output_;
	}

	const std::vector<double>* ConvolutionalLayer::ForwardGpu(const std::vector<double>* input) {
		input_ = input;

		double* d_input = nullptr;
		double* d_kernels = nullptr;
		double* d_biases = nullptr;
		double* d_output = nullptr;

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), input_->size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_kernels), kernels_.size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_biases), biases_.size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), output_.size() * sizeof(double)));

		CUDA_CHECK(cudaMemcpy(d_input, input_->data(), input_->size() * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_kernels, kernels_.data(), kernels_.size() * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_biases, biases_.data(), biases_.size() * sizeof(double), cudaMemcpyHostToDevice));

		cuda::Copy(d_biases, d_output, biases_.size());
		cuda::Sync();

		for (int s = 0; s < kernel_sets_; s++) {
			for (int d = 0; d < kernel_depth_; d++) {
				int kernel_offset = GetKernelOffset(s, d);
				int output_offset = GetOutputOffset(d);
				int input_offset = d * input_width_ * input_height_;

				cuda::CrossCorrelate(
					d_input, input_offset, input_height_, input_width_,
					d_kernels, kernel_offset, kernel_size_, kernel_size_, false,
					d_output, output_offset, output_height_, output_width_, true);
			}
		}

		CUDA_CHECK(cudaMemcpy(output_.data(), d_output, output_.size() * sizeof(double), cudaMemcpyDeviceToHost));

		CUDA_CHECK(cudaFree(d_input));
		CUDA_CHECK(cudaFree(d_kernels));
		CUDA_CHECK(cudaFree(d_biases));
		CUDA_CHECK(cudaFree(d_output));

		return &output_;
	}

	const std::vector<std::vector<double>> ConvolutionalLayer::Backward(std::vector<double> output_gradients) {
		// dE/dK_ij = X_j cross correlate dE/dY_i
		// dE/dB_i = dE/dY_I
		// dE/dX_j = sum(dE/dY_i full correlate K_ij)

		std::vector<double> kernel_gradients(kernel_size_ * kernel_size_ * kernel_depth_ * kernel_sets_, 0.0);
		std::vector<double> input_gradients(input_width_ * input_height_ * kernel_depth_, 0.0);

		for (int s = 0; s < kernel_sets_; s++) {
			for (int d = 0; d < kernel_depth_; d++) {
				int kernel_offset = GetKernelOffset(s, d);
				int output_offset = GetOutputOffset(d);
				int input_offset = d * input_width_ * input_height_;

				CrossCorrelate(*input_, input_offset, input_height_, input_width_,
					output_gradients, output_offset, output_height_, output_width_, false,
					kernel_gradients, kernel_offset, kernel_size_, kernel_size_);

				FullCrossCorrelate(output_gradients, output_offset, output_height_, output_width_,
					kernels_, kernel_offset, kernel_size_, kernel_size_, true,
					input_gradients, input_offset, input_height_, input_width_);
			}
		}

		return { input_gradients, kernel_gradients, output_gradients };
	}

	const std::vector<double> ConvolutionalLayer::Backward(const std::vector<double> output_gradients, double learning_rate) {

		// dE/dK_ij = X_j cross correlate dE/dY_i
		// dE/dB_i = dE/dY_I
		// dE/dX_j = sum(dE/dY_i full correlate K_ij)

		std::vector<double> kernel_gradients(kernel_size_ * kernel_size_ * kernel_depth_ * kernel_sets_, 0.0);
		std::vector<double> input_gradients(input_width_ * input_height_ * kernel_depth_, 0.0);

		for (int s = 0; s < kernel_sets_; s++) {
			for (int d = 0; d < kernel_depth_; d++) {
				int kernel_offset = GetKernelOffset(s, d);
				int output_offset = GetOutputOffset(d);
				int input_offset = d * input_width_ * input_height_;

				CrossCorrelate(*input_, input_offset, input_height_, input_width_,
								output_gradients, output_offset, output_height_, output_width_, false,
								kernel_gradients, kernel_offset, kernel_size_, kernel_size_);

				FullCrossCorrelate(output_gradients, output_offset, output_height_, output_width_,
								   kernels_, kernel_offset, kernel_size_, kernel_size_, true,
								   input_gradients, input_offset, input_height_, input_width_);
			}
		}

		for (int i = 0; i < kernel_gradients.size(); i++) {
			kernels_[i] -= learning_rate * kernel_gradients[i];
		}

		for (int i = 0; i < biases_.size(); i++) {
			biases_[i] -= learning_rate * output_gradients[i];
		}

		return input_gradients;
	}

	const std::vector<double> ConvolutionalLayer::BackwardGpu(const std::vector<double> output_gradients, double learning_rate) {
		
		std::vector<double> input_gradients(input_width_ * input_height_ * input_depth_, 0.0);

		double* d_kernel_gradients = nullptr;
		double* d_input_gradients = nullptr;
		double* d_output_gradients = nullptr;
		double* d_input = nullptr;
		double* d_kernels = nullptr;
		double* d_biases = nullptr;

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_kernel_gradients), 
			kernel_size_ * kernel_size_ * kernel_depth_ * kernel_sets_ * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input_gradients), 
			input_width_ * input_height_ * kernel_depth_ * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output_gradients),
			output_gradients.size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), input_->size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_kernels), kernels_.size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_biases), biases_.size() * sizeof(double)));

		CUDA_CHECK(cudaMemcpy(d_output_gradients, output_gradients.data(), output_gradients.size() * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_input, input_->data(), input_->size() * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_kernels, kernels_.data(), kernels_.size() * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_biases, biases_.data(), biases_.size() * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemset(d_kernel_gradients, 0, kernel_size_ * kernel_size_ * kernel_depth_ * kernel_sets_ * sizeof(double)));
		CUDA_CHECK(cudaMemset(d_input_gradients, 0, input_width_ * input_height_ * kernel_depth_ * sizeof(double)));

		for (int s = 0; s < kernel_sets_; s++) {
			for (int d = 0; d < kernel_depth_; d++) {
				int kernel_offset = GetKernelOffset(s, d);
				int output_offset = GetOutputOffset(d);
				int input_offset = d * input_width_ * input_height_;
	
				cuda::CrossCorrelate(
					d_input, input_offset, input_height_, input_width_,
					d_output_gradients, output_offset, output_height_, output_width_, false,
					d_kernel_gradients, kernel_offset, kernel_size_, kernel_size_, false);
				
				cuda::CrossCorrelate(
					d_output_gradients, output_offset, output_height_, output_width_,
					d_kernels, kernel_offset, kernel_size_, kernel_size_, true,
					d_input_gradients, input_offset, input_height_, input_width_, true);
			}
		}

		cuda::ScalarAdd(d_kernel_gradients, -learning_rate, d_kernels, kernels_.size());
		cuda::ScalarAdd(d_output_gradients, -learning_rate, d_biases, biases_.size());

		CUDA_CHECK(cudaMemcpy(input_gradients.data(), d_input_gradients, input_gradients.size() * sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(kernels_.data(), d_kernels, kernels_.size() * sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(biases_.data(), d_biases, biases_.size() * sizeof(double), cudaMemcpyDeviceToHost));

		CUDA_CHECK(cudaFree(d_kernel_gradients));
		CUDA_CHECK(cudaFree(d_input_gradients));
		CUDA_CHECK(cudaFree(d_output_gradients));
		CUDA_CHECK(cudaFree(d_input));
		CUDA_CHECK(cudaFree(d_kernels));
		CUDA_CHECK(cudaFree(d_biases));

		return input_gradients;
	}	

}
