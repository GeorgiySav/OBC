#include "TrainMNIST.h" 

#include "MnistDataHandler.h"

#include "../../Network.h"

#include <chrono>

namespace obc {

	void TrainMNIST() {
		// Inspired by the LeNet-1 architecture
		Network network = {
			new ConvolutionalLayer(1, 28, 28, 5, 4),
			new ReLU(2304),
			new MaxPoolingLayer(4, 24, 24, 2),
			new Sigmoid(576),
			new ConvolutionalLayer(4, 12, 12, 5, 12),
			new ReLU(768),
			new MaxPoolingLayer(12, 8, 8, 2),
			new Sigmoid(192),
			new DenseLayer(192, 10),
			new Softmax(10)
		};
		/*
		Network network = {
			new ConvolutionalLayer(1, 28, 28, 5, 4),
			new ReLU(2304),
			new MaxPoolingLayer(4, 24, 24, 2),
			new ReLU(576),
			new DenseLayer(576, 10),
			new Softmax(10)
		};*/

		MnistDataHandler data_handler;
		data_handler.LoadFeatureVector("./src/obc/example models/MNIST/train-images.idx3-ubyte");
		data_handler.LoadFeatureLabels("./src/obc/example models/MNIST/train-labels.idx1-ubyte");

		data_handler.SplitData();

		auto [X, Y] = data_handler.GetTrainingData();

		std::cout << "MNIST Model training..." << std::endl;
		auto start = std::chrono::high_resolution_clock::now();

		std::cout << "Begin training" << std::endl;
		network.Train(X, Y, 100, 0.1, ErrorFunction::kCrossEntropy);

		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Training complete" << std::endl;
		std::cout << "Time to train: " << std::chrono::duration<double>(end - start).count() << "s\n";

		network.Serialize("mnist_model.obc", obc::ser::ArchiveType::Binary);
	}

}