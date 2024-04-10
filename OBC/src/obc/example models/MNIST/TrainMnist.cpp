#include "TrainMNIST.h" 
#include "../../Network.h"

namespace obc {

	void TrainMNIST() {
		// LeNet-1
		Network network = {
			new ConvolutionalLayer(1, 28, 28, 5, 4),
			new ReLU(),
		}
	}

}