#include <iostream>
#include <cstdint>
#include <memory>
#include <typeinfo>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> //strlen
#include <errno.h>
#include <unistd.h>    //close
#include <arpa/inet.h> //close
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h> //FD_SET, FD_ISSET, FD_ZERO macros
#include <random>
// opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
// pytorch
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

std::vector<std::string> split(const std::string &s, char delim);

std::string random_string(int len);

cv::Mat TensorToCVMat(torch::Tensor tensor);

cv::Mat stylizeImage(const char* model_path, const char* input_image_path, const char* output_image_path);

int main(int argc, const char *argv[])
{
    if (argc < 4){
        std::cout << "wrong arguments, model_path, input_image_path output_image_path" << std::endl;
        return -1;
    }
    // ./cartoonGan /opt/gan-generator.pt /home/trip2.jpg /home/cg_trip2.jpg
    stylizeImage(argv[1], argv[2], argv[3]);

    return 0;
}

template <typename Out>
void split(const std::string &s, char delim, Out result)
{
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim))
    {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

std::string random_string(int len)
{
    std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::shuffle(str.begin(), str.end(), generator);

    return str.substr(0, len); // assumes 32 < number of characters in str
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

cv::Mat stylizeImage(const char* model_path, const char* input_image_path, const char* output_image_path)
{
    cv::Mat input_image;
    cv::Mat output_image;
    // handle (-215:Assertion failed) !_src.empty() in function 'cvtColor'
    cv::Mat read_image = cv::imread(input_image_path, cv::IMREAD_COLOR);
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

    // std::cout << tensor_image << "\n";

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        // module = torch::jit::load(argv[1]);
        module = torch::jit::load(model_path);
        // std::cout << typeid(module).name() << "\n";
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return output_image;
    }

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image);

    // Execute the model and turn its output into a tensor.
    torch::Tensor output = module.forward(inputs).toTensor();

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
        return output_image;
    }

    std::cout << "stylize finished" << std::endl;

    cv::imwrite(output_image_path, output_mat);

    return output_mat;
}