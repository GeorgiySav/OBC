#include "xor.h"

#include <chrono>

namespace obc {

	extern void XorModel() {
		std::vector<std::vector<double>> inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
		std::vector<std::vector<double>> outputs = { {0}, {1}, {1}, {0} };

		Network cpu_xor_model = {
			new DenseLayer(2, 3),
		    new Sigmoid(3),
			new DenseLayer(3, 1),
			new Sigmoid(1)
		};
		std::cout << "CPU Model" << std::endl;
		std::cout << "Xor Model created" << std::endl;

		std::cout << "Xor Model training..." << std::endl;
		auto start = std::chrono::high_resolution_clock::now();

		// Train the model
		//cpu_xor_model.Train(inputs, outputs, 10000, 0.1, ErrorFunction::MSE);
		
		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> elapsed = end - start;
		std::cout << "Time to train: " << elapsed.count() << "s\n";

		for (int i = 0; i < inputs.size(); i++) {
			std::vector<double> prediction = cpu_xor_model.Predict(inputs[i]);
			std::cout << "Input: " << inputs[i][0] << " " << inputs[i][1] 
				<< ", Output: " << prediction[0] 
				<< ", Expected Output: " << outputs[i][0] << std::endl;
		}
		std::cout << std::endl;

		Network gpu_xor_model = {
			new DenseLayer(2, 3),
			new Sigmoid(3),
			new DenseLayer(3, 1),
			new Sigmoid(1)
		};
		//gpu_xor_model.setGpuEnabled(true);
		std::cout << "GPU Model" << std::endl;
		std::cout << "Xor Model created" << std::endl;

		std::cout << "Xor Model training..." << std::endl;
		start = std::chrono::high_resolution_clock::now();

		TrainingParameters params;
		/*params.learning_rate = 0.1;
		params.epochs = 10000;
		params.batch_size = 2;
		params.optimization = Optimization::kSGD;*/
		params.learning_rate = 0.01;
		params.epochs = 10000;
		params.batch_size = 4;
		params.optimization = Optimization::kAdam;
		params.beta1 =  0.9;
		params.beta2 = 0.999;
		params.epsilon = 1e-8;
		gpu_xor_model.Train(inputs, outputs, params);

		end = std::chrono::high_resolution_clock::now();

		elapsed = end - start;
		std::cout << "Time to train: " << elapsed.count() << "s\n";

		for (int i = 0; i < inputs.size(); i++) {
			std::vector<double> prediction = gpu_xor_model.Predict(inputs[i]);
			std::cout << "Input: " << inputs[i][0] << " " << inputs[i][1]
				<< ", Output: " << prediction[0]
				<< ", Expected Output: " << outputs[i][0] << std::endl;
		}
		std::cout << std::endl;
	}

}