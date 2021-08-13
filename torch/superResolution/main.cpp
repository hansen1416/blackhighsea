#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

// install opencv 4.5.1, https://github.com/opencv/opencv/archive/4.5.1.zip
// opencv 头文件
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>
#include <typeinfo>
#include <stdio.h>
#include <string.h> //strlen
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>    //close
#include <arpa/inet.h> //close
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h> //FD_SET, FD_ISSET, FD_ZERO macros

cv::Mat TensorToCVMat(torch::Tensor tensor);

int srImage(const std::string &model_path, const std::string &input_image_path,
            const std::string &output_image_path);

int main(int argc, const char *argv[])
{
    std::string model_path = "/home/hlz/Downloads/srGeneratorTrace.pt";
    std::string input_image_path = "/home/hlz/Pictures/lr.jpg";
    std::string output_image_path = "/home/hlz/Pictures/sr.jpg";
    std::string output_path;

    // std::cout << "have paths "<< std::endl;

    output_path = srImage(model_path, input_image_path, output_image_path);

    std::cout << "output image " << output_path << std::endl;

    return 0;
}

cv::Mat TensorToCVMat(torch::Tensor tensor)
{
    // torch.squeeze(input, dim=None, *, out=None) → Tensor
    // Returns a tensor with all the dimensions of input of size 1 removed.
    // tensor.detach
    // Returns a new Tensor, detached from the current graph.
    // permute dimension, 3x700x700 => 700x700x3
    tensor = tensor.detach().permute({1, 2, 0});
    // float to 255 range
    tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
    // GPU to CPU?, may not needed
    tensor = tensor.to(torch::kCPU);
    // shape of tensor
    int64_t height = tensor.size(0);
    int64_t width = tensor.size(1);

    // Mat takes data form like {0,0,255,0,0,255,...} ({B,G,R,B,G,R,...})
    // so we must reshape tensor, otherwise we get a 3x3 grid
    tensor = tensor.reshape({width * height * 3});
    // CV_8UC3 is an 8-bit unsigned integer matrix/image with 3 channels
    cv::Mat imgbin(cv::Size(width, height), CV_8UC3, tensor.data_ptr());

    return imgbin;
}

int srImage(const std::string &model_path, const std::string &input_image_path,
            const std::string &output_image_path)
{
    std::cout << "start SR image" << std::endl;

    cv::Mat input_image;
    // handle (-215:Assertion failed) !_src.empty() in function 'cvtColor'
    cv::Mat read_image = cv::imread(input_image_path);
    if (read_image.empty() || !read_image.data)
        std::cout << "read image fail" << input_image_path << std::endl;

    cv::cvtColor(read_image, input_image, cv::COLOR_BGR2RGB);

    // 转换 [unsigned int] to [float]
    input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);
    torch::Tensor tensor_image = torch::from_blob(input_image.data, {1, input_image.rows, input_image.cols, 3});
    tensor_image = tensor_image.permute({0, 3, 1, 2});

    // transforms.Normalize(mean=[0.485, 0.456, 0.406],
    //                    std=[0.229, 0.224, 0.225])
    tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
    tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
    tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);

    std::cout << "get image tensor" << tensor_image.sizes() << std::endl;

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        // module = torch::jit::load(argv[1]);
        module = torch::jit::load(model_path);
        std::cout << "model loaded " << typeid(module).name() << "\n";
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model " << e.what() << std::endl;
        return -1;
    }

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image);

    std::cout << "create a vector of inputs " << std::endl;

    // Execute the model and turn its output into a tensor.
    torch::Tensor output = module.forward(inputs).toTensor();

    std::cout << "get output from model " << output.sizes() << std::endl;

    // # Adding 0.1 to all normalization values since the model is trained (erroneously)
    // without correct de-normalization
    output[0][0] = output[0][0].mul_(0.229).add_(0.485 + 0.1);
    output[0][1] = output[0][1].mul_(0.224).add_(0.456 + 0.1);
    output[0][2] = output[0][2].mul_(0.225).add_(0.406 + 0.1);

    torch::Tensor temp_tensor = torch::rand({3, output.sizes()[2], output.sizes()[3]});
    // swap channel, from RGB => BGR
    temp_tensor[0] = output[0][2];
    temp_tensor[1] = output[0][1];
    temp_tensor[2] = output[0][0];

    cv::Mat output_mat = TensorToCVMat(temp_tensor);

    if (output_mat.empty())
    {
        std::cout << "TensorToCVMat return empty " << std::endl;
        return 1;
    }

    cv::imwrite(output_image_path, output_mat);

    return 0;
}
