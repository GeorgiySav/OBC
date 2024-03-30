#pragma once

#include "Layer.h"

namespace obc {
	class ActivationLayer : public Layer {
	public:
		// constructor should take in an activation function as a function pointer
		ActivationLayer(size_t output_size, 
			double (*activation_function)(double), 
			void (*activation_function_gpu)(const std::vector<double>&, std::vector<double>&))
			: Layer(output_size), activation_function_gpu_(activation_function_gpu) {
			activation_function_ = activation_function;
		}
		~ActivationLayer() {}

		const std::vector<double>* Forward(const std::vector<double>* input) override {
			input_ = input;
			for (size_t i = 0; i < output_.size(); i++) {
				output_[i] = activation_function_(input_->at(i));
			}
			return &output_;
		}
		const std::vector<double>* ForwardGpu(const std::vector<double>* input) override {
			input_ = input;
			activation_function_gpu_(*input, output_);
			return &output_;
		}

		LayerType GetType() const override {
			return LayerType::Activation;
		}
	private:
		double (*activation_function_)(double);
		void (*activation_function_gpu_)(const std::vector<double>&, std::vector<double>&);
	};
}