#pragma once

#include <random>

#include "Layer.h"

namespace obc {
	// Dense Layer
	// A fully connected layer.
	// Each input is connected to each output.
	// Each output has a bias.
	class DenseLayer : public Layer {
	public:
		DenseLayer(int input_size, int output_size)
			: Layer(output_size) {

			std::random_device rnd_device;
			std::mt19937 engine{ rnd_device() };
			std::normal_distribution<double> dist(0.0, 1.0/std::sqrt(input_size));

			// initialise weights and biases with random values
			weights_.resize(input_size * output_size);
			std::generate(weights_.begin(), weights_.end(), [&]() { return dist(engine); });

			biases_.resize(output_size, 0.0);
			//std::generate(biases_.begin(), biases_.end(), [&]() { return dist(engine); });
		}

		const std::vector<double>* Forward(const std::vector<double>* input) override;

		/*
		Output layout:
		0 - dE/dX
		1 - dE/dW
		2 - dE/dB
		*/
		const std::vector<std::vector<double>> Backward(std::vector<double> output_gradients) override;

		void SetWeights(const std::vector<double>& weights) {
			weights_ = weights;
		}
		void SetBiases(const std::vector<double>& biases) {
			biases_ = biases;
		}

		std::vector<std::vector<double>*> GetTrainableParameters() override {
			return { &weights_, &biases_ };
		}

		const ser::LayerData Serialize() const override {
			ser::LayerData data;	
			data.type = ser::LayerType::kDense;
			data.input_size = weights_.size() / biases_.size();
			data.output_size = biases_.size();
			data.trainable_parameters["weights"] = weights_;
			data.trainable_parameters["biases"] = biases_;
			return data;
		}

		const ser::LayerType GetType() const override {
			return ser::LayerType::kDense;
		}

	private:	
		double GetWeight(int input_index, int output_index) const {
			return weights_[input_index * biases_.size() + output_index];
		}

		std::vector<double> weights_;
		std::vector<double> biases_;
	};
}