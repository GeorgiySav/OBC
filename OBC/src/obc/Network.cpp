#include "Network.h"

#include <type_traits>

namespace obc {

	// template instantiation
	template void Network::Train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, int epochs, double learning_rate, ErrorFunction error);
	template void Network::Train(const std::vector<const std::vector<double>*>& X, const std::vector<const std::vector<double>*>& Y, int epochs, double learning_rate, ErrorFunction error);

	template <typename T>
	void Network::Train(
		const std::vector<T>& X, const std::vector<T>& Y,
		int epochs, double learning_rate, ErrorFunction error_type) {

		// set the error functions
		double (*error_function)(const std::vector<double>&, const std::vector<double>&) = nullptr;
		std::vector<double> (*error_prime)(const std::vector<double>&, const std::vector<double>&) = nullptr;
		switch (error_type)
		{
		case obc::ErrorFunction::kMSE:
			error_function = &Mse;
			error_prime = &MsePrime;
			break;
		case obc::ErrorFunction::kCrossEntropy:
			error_function = &CrossEntropy;
			error_prime = &CrossEntropyPrime;
			break;
		default:
			error_function = &Mse;
			error_prime = &MsePrime;
			break;
		}

		// Train the network
		for (int epoch = 0; epoch < epochs; ++epoch) {
			double error = 0;
			for (int i = 0; i < X.size(); i++) {
				// Forward propagation
				std::vector<double> output;
				if constexpr (std::same_as<T, std::vector<double>>)
					output = this->Predict(X[i]);
				else if constexpr (std::same_as<T, const std::vector<double>*>)
					output = this->Predict(*X[i]);

				// Error
				if constexpr (std::same_as<T, std::vector<double>>)
					error += error_function(Y[i], output);
				else if constexpr (std::same_as<T, const std::vector<double>*>)
					error += error_function(*Y[i], output);

				// Backward propagation
				std::vector<double> gradients;
				if constexpr (std::same_as<T, std::vector<double>>)
					gradients = error_prime(Y[i], output);
				else if constexpr (std::same_as<T, const std::vector<double>*>)
					gradients = error_prime(*Y[i], output);

				// if the last layer is a softmax layer, and cross entropy is used, then the gradients are already calculated
				// so we skip the last layer
				int last_layer = layers_.size() - 1;
				if (layers_[last_layer]->GetType() == ser::LayerType::kSoftmax && error_type == ErrorFunction::kCrossEntropy)
					last_layer--;
				for (int j = last_layer; j >= 0; j--) {
					if (gpu_enabled_)
						gradients = layers_[j]->BackwardGpu(gradients, learning_rate);
					else
						gradients = layers_[j]->Backward(gradients, learning_rate);
				}
			}
			error /= X.size();
			std::cout << "Epoch " << epoch << ", Error: " << error << std::endl;
		}
	}
}