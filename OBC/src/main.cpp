#include <iostream>
#include <span>
#include <algorithm>

#include "obc/Network.h"

#include "obc/example models/Xor.h"

#include "obc/example models/SerializationExample.h"

#include "obc/example models/TestConvoModel.h"

#include "obc/example models/MNIST/TrainMnist.h"

int main() {

	//obc::XorModel();

	//obc::SerializationExample();

	//obc::TestConvoModel();

	obc::TrainMNIST();

	return 0;
}