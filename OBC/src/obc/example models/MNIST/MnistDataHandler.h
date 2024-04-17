#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include <random>

namespace obc {

	class MnistData {
	public:
		MnistData() {
			image_data_ = std::vector<double>();
		}
		~MnistData() {
		}

		void SetImageData(const std::vector<double>& feature_vector) {
			this->image_data_ = feature_vector;
		}
		void AppendImageData(double feature) {
			this->image_data_.push_back(feature);
		}
		void SetLabel(uint8_t label) {
			this->label_ = label;
		}	

		int GetImageSize() {
			return this->image_data_.size();
		}
		const std::vector<double>& GetImageData() {
			return this->image_data_;
		}
		uint8_t GetLabel() {
			return this->label_;
		}
		

	private:
		std::vector<double> image_data_;
		uint8_t label_;
	};

	class MnistDataHandler {
	public:
		MnistDataHandler();
		~MnistDataHandler();

		void LoadFeatureVector(std::string file_name);
		void LoadFeatureLabels(std::string file_name);

		void SplitData();

		uint32_t ConvertToLittleEndian(const unsigned char* bytes);

		std::vector<std::shared_ptr<MnistData>>& GetRawTrainingData() {
			return this->training_data_;
		}
		std::vector<std::shared_ptr<MnistData>>& GetRawTestingData() {
			return this->testing_data_;
		}
		std::vector<std::shared_ptr<MnistData>>& GetRawValidationData() {
			return this->validation_data_;
		}

		std::tuple<std::vector<const std::vector<double>*>, std::vector<const std::vector<double>*>> GetTrainingData() {
			std::vector<const std::vector<double>*> X;
			std::vector<const std::vector<double>*> Y;

			for (auto& data : training_data_) {
				X.push_back(&data->GetImageData());
				std::vector<double>* y = new std::vector<double>(10, 0.0);
				y->at(data->GetLabel()) = 1.0;
				Y.push_back(y);
			}

			return std::make_tuple(X, Y);	
		}

		std::tuple<std::vector<const std::vector<double>*>, std::vector<int>> GetTestingData() {
			std::vector<const std::vector<double>*> X;
			std::vector<int> Y;

			for (auto& data : testing_data_) {
				X.push_back(&data->GetImageData());
				Y.push_back(data->GetLabel());
			}

			return std::make_tuple(X, Y);
		}

		std::tuple<std::vector<const std::vector<double>*>, std::vector<int>> GetValidationData() {
			std::vector<const std::vector<double>*> X;
			std::vector<int> Y;

			for (auto& data : validation_data_) {
				X.push_back(&data->GetImageData());
				Y.push_back(data->GetLabel());
			}

			return std::make_tuple(X, Y);
		}

		void PrintRandom(int n) {
			std::random_device rnd_device;
			std::mt19937 engine{ rnd_device() };
			std::uniform_int_distribution<int> dist{ 0, (int)training_data_.size()};

			for (int i = 0; i < n; i++) {
				int index = dist(engine);
				auto image_data = training_data_[index]->GetImageData();
				int label = training_data_[index]->GetLabel();

				for (int y = 0; y < 28; y++) {
					for (int x = 0; x < 28; x++) {

						if (image_data[y * 28 + x] == 0)
							std::cout << ". ";
						else
							std::cout << "# ";
					}
					std::cout << "\n";
				}
				std::cout << "Label: " << label << "\n\n";
			}
		}
	private:
		std::vector<std::shared_ptr<MnistData>> raw_data_; // data pre-split
		std::vector<std::shared_ptr<MnistData>> training_data_;
		std::vector<std::shared_ptr<MnistData>> testing_data_;
		std::vector<std::shared_ptr<MnistData>> validation_data_;

		const double kTrainingPercentage = 0.75;
		const double kTestingPercentage = 0.2;
		const double kValidationPercentage = 0.05;
	};

}