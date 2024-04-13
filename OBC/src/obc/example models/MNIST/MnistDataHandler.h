#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>

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

			for (auto data : training_data_) {
				X.push_back(&data->GetImageData());
				std::vector<double>* y = new std::vector<double>(10, 0.0);
				y->at(data->GetLabel()) = 1.0;
				Y.push_back(y);
			}

			return std::make_tuple(X, Y);	
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