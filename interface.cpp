#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

int main()
{ 
	// Create a env to create sessions
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mycustommodel");

	// Configure session options
	Ort::SessionOptions options;
	// Set optimization level
	options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	
	// Create a session
	const char* model_path = "models/yolo11s.onnx";
	Ort::Session session(env, model_path, options);

	// Use allocator to get IO names
	Ort::AllocatorWithDefaultOptions allocator;
	std::vector<std::string> input_names = session.GetInputNames();
	std::vector<std::string> output_names = session.GetOutputNames();
	const char* input_name = input_names[0].c_str();
	const char* output_name = output_names[0].c_str();

	// Create Input Vector
	cv::Mat img = cv::imread("images/image1.jpg");
	cv::Mat blob;

	cv::dnn::blobFromImage(
		img,
		blob,
		1.0 / 255.0,
		cv::Size(640, 640), //resize
		cv::Scalar(),
		true, //swap RB
		false //crop
	);

	float* data = reinterpret_cast<float*>(blob.data);
	size_t size = blob.total();
	std::vector<int64_t> input_shape = {1, 3, 640, 640}; // Set input vector dimension i.e 3 channel rgb image 640*640
	
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memory_info,
		data,
		size,
		input_shape.data(),
		input_shape.size()
	);


	// Run Inference
	std::vector<Ort::Value> inputs;
	inputs.push_back(std::move(input_tensor));

	std::vector<Ort::Value> outputs = session.Run(
		Ort::RunOptions(nullptr),
		&input_name,
		inputs.data(),
		inputs.size(),
		&output_name,
		1);

	// outputs[0] is the first output tensor
	Ort::Value output_tensor = std::move(outputs[0]);

	// Read output data from Ort::Value
	float* output_data = output_tensor.GetTensorMutableData<float>();

	// Get shape and type information
	auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
	auto shape = type_info.GetShape();

	size_t output_size = 1;
	for (int64_t dim : shape) output_size *= dim;

	// Example: print first 10 values
	for (size_t i = 0; i < std::min<size_t>(10, output_size); i++)
	{
		std::cout << output_data[i] << " ";
	}
	std::cout << std::endl;

	return 0;
}
