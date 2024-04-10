#include "ActivationLayer.h"

namespace obc {

	const std::vector<double>* ActivationLayer::Forward(const std::vector<double>* input) {
		input_ = input;
		for (int i = 0; i < output_.size(); i++) {
			output_[i] = activation_function_(input_->at(i));
		}
		return &output_;
	}
	const std::vector<double>* ActivationLayer::ForwardGpu(const std::vector<double>* input) {
		input_ = input;
		activation_function_gpu_(*input, output_);
		return &output_;
	}

	const std::vector<double> ActivationLayer::Backward(const std::vector<double> output_gradients, double learning_rate) {
		// dE/dx = dE/dy elementwise_mul f'(x)
		std::vector<double> input_gradients(input_->size());
		for (int i = 0; i < input_gradients.size(); i++) {
			input_gradients[i] = output_gradients[i] * activation_derivative_(input_->at(i));
		}
		return input_gradients;
	}
	const std::vector<double> ActivationLayer::BackwardGpu(const std::vector<double> output_gradients, double learning_rate) {

		double* d_input = nullptr;
		double* d_input_gradients = nullptr;
		double* d_output_gradients = nullptr;
		std::vector<double> input_gradients(input_->size());

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), input_->size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input_gradients), input_->size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output_gradients), output_.size() * sizeof(double)));

		CUDA_CHECK(cudaMemcpy(d_input, input_->data(), input_->size() * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_output_gradients, output_gradients.data(), output_gradients.size() * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemset(d_input_gradients, 0, input_->size() * sizeof(double)));

		activation_derivative_gpu_(d_input, d_input_gradients, input_->size());
		cuda::ElementwiseMul(d_output_gradients, d_input_gradients, d_input_gradients, input_->size());

		CUDA_CHECK(cudaMemcpy(input_gradients.data(), d_input_gradients, input_->size() * sizeof(double), cudaMemcpyDeviceToHost));

		CUDA_CHECK(cudaFree(d_input));
		CUDA_CHECK(cudaFree(d_input_gradients));
		CUDA_CHECK(cudaFree(d_output_gradients));

		return input_gradients;
	}

}