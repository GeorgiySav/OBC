#pragma once

#include <memory>
#include <vector>

#include "Layer.h"
#include "DenseLayer.h"
#include "ActivationFunctions.h"

#include "crow_all.h"

namespace obc {
	class Network {
	public:
		Network() : gpu_enabled_(false) {}
		Network(std::initializer_list<Layer*> layers) : layers_(layers), gpu_enabled_(false) {}
		~Network() {
			for (auto& layer : layers_) {
				delete layer;
			}
		}

		void setGpuEnabled(bool on) { gpu_enabled_ = on; }

		std::vector<double> Predict(const std::vector<double>& input) {
			const std::vector<double>* output = &input;
			if (!gpu_enabled_) {
				for (auto& layer : layers_) {
					output = layer->Forward(output);
				}
			}
			else {
				for (auto& layer : layers_) {
					output = layer->ForwardGpu(output);
				}
			}
			return *output;
		}

		// gradient descent based training
		void Train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, size_t epochs, double learning_rate);
	
		// Returns a json which represents the structure of the network
		crow::json::wvalue ToJsonSpine();

	private:
		std::vector<Layer*> layers_;
		bool gpu_enabled_;
	};
}