#include "TrainMNIST.h" 

#include "MnistDataHandler.h"

#include "../../Network.h"

#include <chrono>

namespace obc {

	void TrainMNIST() {
		// Inspired by the LeNet-1 architecture
		/*
		Network network = {
			new ConvolutionalLayer(1, 28, 28, 5, 4),
			new ReLU(2304),
			new MaxPoolingLayer(4, 24, 24, 2),
			new ConvolutionalLayer(4, 12, 12, 5, 12),
			new ReLU(768),
			new MaxPoolingLayer(12, 8, 8, 2),
			new DenseLayer(192, 10),
			new Softmax(10)
		};
		Network network = {
			new ConvolutionalLayer(1, 28, 28, 5, 4),
			new ReLU(2304),
			new MaxPoolingLayer(4, 24, 24, 2),
			new DenseLayer(4 * 12 * 12, 10),
			new Softmax(10)
		};
		*/
		Network network = {
			new ConvolutionalLayer(1, 28, 28, 5, 6),
			new ReLU(24 * 24 * 6),
			new MaxPoolingLayer(6, 24, 24, 2),
			new ConvolutionalLayer(6, 12, 12, 5, 16),
			new ReLU(8 * 8 * 16),
			new MaxPoolingLayer(16, 8, 8, 2),
			new DenseLayer(16 * 4 * 4, 120),
			new ReLU(120),
			new DenseLayer(120, 84),
			new ReLU(84),
			new DenseLayer(84, 10),
			new Softmax(10)
		};
		network.Deserialize("mnist_model_lenet-5.obc", obc::ser::ArchiveType::Binary);

		MnistDataHandler data_handler;
		data_handler.LoadFeatureVector("./src/obc/example models/MNIST/train-images.idx3-ubyte");
		data_handler.LoadFeatureLabels("./src/obc/example models/MNIST/train-labels.idx1-ubyte");

		data_handler.SplitData();

		data_handler.PrintRandom(1);

		auto [X, Y] = data_handler.GetTrainingData();
		auto [X_test, Y_test] = data_handler.GetTestingData();
		auto [X_val, Y_val] = data_handler.GetValidationData();

		std::cout << "MNIST Model training..." << std::endl;
		auto start = std::chrono::high_resolution_clock::now();

		std::cout << "Begin training" << std::endl;

		TrainingParameters params;
		params.error = ErrorFunction::kCrossEntropy;
		params.optimiser = Optimizer::kAdam;
		params.beta1 = 0.9;
		params.beta2 = 0.999;
		params.epsilon = 1e-8;

		params.learning_rate = 0.00001;
		params.epochs = 3;
		network.Train(X, Y, params, X_val, Y_val);

		params.learning_rate = 0.00002;
		params.epochs = 3;
		//network.Train(X, Y, params, X_val, Y_val);

		params.learning_rate = 0.00001;
		params.epochs = 3;
		//network.Train(X, Y, params, X_val, Y_val);

		params.learning_rate = 0.000005;
		params.epochs = 4;
		//network.Train(X, Y, params, X_val, Y_val);

		params.learning_rate = 0.000001;
		params.epochs = 8;
		//network.Train(X, Y, params, X_val, Y_val);

		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Training complete" << std::endl;
		std::cout << "Time to train: " << std::chrono::duration<double>(end - start).count() << "s\n";

		std::vector<int> Y_labels;
		for (const auto& y : Y) {
			int max_index = 0;
			for (int i = 1; i < 10; i++) {
				if (y->at(max_index) < y->at(i))
					max_index = i;
			}
			Y_labels.push_back(max_index);
		}
		double accuracy = network.Test(X_test, Y_test);
		std::cout << "Accuracy: " << accuracy << "%" << std::endl;

		network.Serialize("mnist_model_lenet-5.obc", obc::ser::ArchiveType::Binary);
	}

}