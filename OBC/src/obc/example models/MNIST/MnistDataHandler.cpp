#include "MnistDataHandler.h"

#include <algorithm>
#include <random>

namespace obc {
	MnistDataHandler::MnistDataHandler() {
	}

	MnistDataHandler::~MnistDataHandler() {
	}

	void MnistDataHandler::LoadFeatureVector(std::string file_name) {
		uint32_t header[4]; // magic number, number of images, number of rows, number of columns
		unsigned char bytes[4];

		std::ifstream file(file_name, std::ios::binary);
		if (!file.is_open()) {
			throw std::runtime_error("Could not open file for reading");
		}

		for (int i = 0; i < 4; i++) {
			if (file.read((char*)bytes, sizeof(bytes))) {
				header[i] = ConvertToLittleEndian(bytes);
			}	
		}

		std::cout << "Got feature vector header" << std::endl;
		std::cout << "Number of images: " << header[1] << std::endl;
		std::cout << "Number of rows: " << header[2] << std::endl;
		std::cout << "Number of columns: " << header[3] << std::endl;

		int image_size = header[2] * header[3];
		for (int i = 0; i < header[1]; i++) {
			std::shared_ptr<MnistData> d = std::make_shared<MnistData>();
			uint8_t element = 0;
			for (int j = 0; j < image_size; j++) {
				if (file.read((char*)(&element), sizeof(element))) {
					// normalise pixel values
					double normalised = (double)element / 255.0;
					d->AppendImageData(normalised);
				}
				else {
					throw std::runtime_error("Could not read image data at index: " + j);
				}		
			}
			raw_data_.push_back(d);
		}

		file.close();

		std::cout << "Successfully loaded feature vectors" << std::endl;
		std::cout << "Number of feature vectors: " << raw_data_.size() << std::endl;
	}

	void MnistDataHandler::LoadFeatureLabels(std::string file_name) {
		uint32_t header[2]; // magic number, number of items
		unsigned char bytes[4];

		std::ifstream file(file_name, std::ios::binary);
		if (!file.is_open()) {
			throw std::runtime_error("Could not open file for reading");
		}

		for (int i = 0; i < 2; i++) {
			if (file.read((char*)bytes, sizeof(bytes))) {
				header[i] = ConvertToLittleEndian(bytes);
			}	
		}

		std::cout << "Got label file header" << std::endl;
		std::cout << "Number of items: " << header[1] << std::endl;

		for (int i = 0; i < header[1]; i++) {
			uint8_t element[1];
			if (file.read((char*)element, sizeof(element))) {
				raw_data_.at(i)->SetLabel(element[0]);
			}
			else {
				throw std::runtime_error("Could not read label data");
			}
		}

		file.close();

		std::cout << "Successfully loaded labels" << std::endl;
		std::cout << "Number of labels: " << raw_data_.size() << std::endl;
	}

	void MnistDataHandler::SplitData() {
	
		std::vector<int> indices;
		for (unsigned i = 0; i < raw_data_.size(); i++) {
			indices.push_back(i);
		}
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(indices.begin(), indices.end(), g);

		int training_size = raw_data_.size() * kTrainingPercentage;
		int testing_size = raw_data_.size() * kTestingPercentage;
		int validation_size = raw_data_.size() * kValidationPercentage;

		for (int i = 0; i < training_size; i++) {
			training_data_.push_back(raw_data_.at(indices[i]));
		}
		for (int i = training_size; i < training_size + testing_size; i++) {
			testing_data_.push_back(raw_data_.at(indices[i]));
		}
		for (int i = training_size + testing_size; i < training_size + testing_size + validation_size; i++) {
			validation_data_.push_back(raw_data_.at(indices[i]));
		}

		std::cout << "Successfully split data" << std::endl;
		std::cout << "Training data size: " << training_data_.size() << std::endl;
		std::cout << "Testing data size: " << testing_data_.size() << std::endl;
		std::cout << "Validation data size: " << validation_data_.size() << std::endl;
	}

	uint32_t MnistDataHandler::ConvertToLittleEndian(const unsigned char* bytes) {
		return (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
	}
}