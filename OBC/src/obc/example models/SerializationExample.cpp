#include "SerializationExample.h"

#include "../Network.h"

namespace obc {

	void SerializationExample() {
		Network nn = {
			new DenseLayer(2, 3),
			new Sigmoid(3),
			new DenseLayer(3, 1),
			new Sigmoid(1)
		};

		nn.Serialize("test_model.bin", ser::ArchiveType::Binary);

		Network nn2;

		nn2.Deserialize("test_model.bin", ser::ArchiveType::Binary);

		std::vector<double> input = { 1, 1 };

		std::vector<double> output0 = nn.Predict(input);
		std::vector<double> output1 = nn2.Predict(input);

		std::cout << "Output 0: " << output0[0] << std::endl;
		std::cout << "Output 1: " << output1[0] << std::endl;
	}

}