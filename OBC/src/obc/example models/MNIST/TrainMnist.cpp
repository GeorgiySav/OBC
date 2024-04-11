#include "TrainMNIST.h" 
#include "../../Network.h"

namespace obc {

	void TrainMNIST() {
		// Inspired by the LeNet-1 architecture
		Network network = {
			new ConvolutionalLayer(1, 28, 28, 5, 4),
			new ReLU(2304),
			new MaxPoolingLayer(4, 24, 24, 2, 2),
			new ReLU(576),
			new ConvolutionalLayer(4, 12, 12, 5, 12),
			new ReLU(768),
			new MaxPoolingLayer(12, 8, 8, 2, 2),
			new ReLU(192),
			new DenseLayer(192, 10),
			new Softmax(10)
		};
	}

}