#pragma once

#include <vector>
#include <string>

#include "cuda utilities/cuda_ops.h"

#include "Serialization.h"

namespace obc {
	// Parent Class for all layers
	class Layer {
	public:
		Layer(int output_size) 
			: input_(nullptr), output_(output_size) {}
		~Layer() {}

		virtual const std::vector<double>* Forward(const std::vector<double>* input) = 0;
		virtual const std::vector<double>* ForwardGpu(const std::vector<double>* input) = 0;

		virtual const std::vector<std::vector<double>> Backward(std::vector<double> output_gradients) = 0;

		virtual const std::vector<double> Backward(const std::vector<double> output_gradients, double learning_rate) = 0;
		virtual const std::vector<double> BackwardGpu(const std::vector<double> output_gradients, double learning_rate) = 0;

		virtual std::vector<std::vector<double>*> GetTrainableParameters() = 0;

		virtual const ser::LayerType GetType() const = 0;
		virtual const ser::LayerData Serialize() const = 0;
	protected:
		// a pointer to a vector of inputs
		// will be retrieved from another layer
		const std::vector<double>* input_;
		// the result of the forward operation
		std::vector<double> output_;

	};
}