#include "Network.h"

namespace obc {

	void Network::Train(
		const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y,
		size_t epochs, double learning_rate, ErrorFunction error) {

		// set the error functions
		double (*error_function)(const std::vector<double>&, const std::vector<double>&) = nullptr;
		std::vector<double> (*error_prime)(const std::vector<double>&, const std::vector<double>&) = nullptr;
		switch (error)
		{
		case obc::ErrorFunction::MSE:
			error_function = &Mse;
			error_prime = &MsePrime;
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
				std::vector<double> output = this->Predict(X[i]);

				// Error
				error += error_function(Y[i], output);

				// Backward propagation
				std::vector<double> gradients = error_prime(Y[i], output);
				for (int j = layers_.size() - 1; j >= 0; j--) {
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