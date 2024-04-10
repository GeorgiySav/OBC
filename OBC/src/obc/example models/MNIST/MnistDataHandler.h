#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_set>

namespace obc {

	class MnistData {
	public:
		MnistData() {
			image_data_ = new std::vector<double>();
		}
		~MnistData() {
			delete image_data_;
		}

		void SetImageData(std::vector<double>* feature_vector) {
			this->image_data_ = feature_vector;
		}
		void AppendImageData(double feature) {
			this->image_data_->push_back(feature);
		}
		void SetLabel(uint8_t label) {
			this->label_ = label;
		}
		void SetEnumLabel(int enum_label) {
			this->enum_label_ = enum_label;
		}

		int GetImageSize() {
			return this->image_data_->size();
		}
		std::vector<double>* GetImageData() {
			return this->image_data_;
		}
		uint8_t GetLabel() {
			return this->label_;
		}
		int GetEnumLabel() {
			return this->enum_label_;
		}

	private:
		std::vector<double>* image_data_;
		uint8_t label_;
		int enum_label_; 
	};

	class MnistDataHandler {
	public:
		MnistDataHandler();
		~MnistDataHandler();

		void LoadFeatureVector(std::string file_name);
		void LoadFeatureLabels(std::string file_name);

		void SplitData();

		void CountClasses();

		uint32_t ConvertToLittleEndian(const unsigned char* bytes);

		std::vector<MnistData*>* GetTrainingData() {
			return this->training_data_;
		}
		std::vector<MnistData*>* GetTestingData() {
			return this->testing_data_;
		}
		std::vector<MnistData*>* GetValidationData() {
			return this->validation_data_;
		}
	private:
		std::vector<MnistData*>* raw_data_; // data pre-split
		std::vector<MnistData*>* training_data_;
		std::vector<MnistData*>* testing_data_;
		std::vector<MnistData*>* validation_data_;

		int num_classes_;
		int image_data_size_;
		std::map<int8_t, int> class_counts_;

		const double kTrainingPercentage = 0.75;
		const double kTestingPercentage = 0.2;
		const double kValidationPercentage = 0.05;
	};

}