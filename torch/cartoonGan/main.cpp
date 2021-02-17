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

torch::Tensor image_to_tensor();

int main(int argc, const char *argv[])
{
    // if (argc != 3)
    // {
    //     std::cerr << "usage: gan-model <path-to-exported-model> <path-to-image-tensor>"
    //               << std::endl;
    //     return -1;
    // }

    cv::Mat input_image;
    cv::Mat read_image = cv::imread("/home/hlz/Pictures/p1.jpg");
    if (read_image.empty() || !read_image.data)
        std::cout << "read image fail" << std::endl;

    cv::cvtColor(read_image, input_image, cv::COLOR_BGR2RGB);

     // 转换 [unsigned int] to [float]
    input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);
    torch::Tensor tensor_image = torch::from_blob(input_image.data, {1, input_image.rows, input_image.cols,3});
    tensor_image = tensor_image.permute({0,3,1,2});

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
        module = torch::jit::load("/home/hlz/blackhighsea/torch/cartoonGan/model-trace/gan-generator.pt");
        std::cout << typeid(module).name() << "\n";
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    // torch::Tensor image_tensor;
    // try
    // {
    //     torch::jit::script::Module image_list = torch::jit::load(argv[2]);

    //     image_tensor = image_list.attr("image_list").toTensor();

    //     // Load values by name
    //     // std::cout << image_tensor << "\n";
    // }
    // catch (const std::exception &e)
    // {
    //     std::cerr << e.what() << '\n';
    //     return -1;
    // }

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image);

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();

    torch::save(output, "oi.pt");
    
    return 0;
}

// torch::Tensor image_to_tensor() {

//     cv::Mat input_image;
//     cv::Mat read_image = cv::imread("/home/hlz/Pictures/p1.jpg");
//     if (read_image.empty() || !read_image.data)
//         std::cout << "read image fail" << std::endl;

//     cv::cvtColor(read_image, input_image, cv::COLOR_BGR2RGB);

//      // 转换 [unsigned int] to [float]
//     input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);
//     torch::Tensor tensor_image = torch::from_blob(input_image.data, {1, input_image.rows, input_image.cols,3});
//     tensor_image = tensor_image.permute({0,3,1,2});

//     // transforms.Normalize(mean=[0.485, 0.456, 0.406],
//     //                    std=[0.229, 0.224, 0.225])
//     tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
//     tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
//     tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);
 
//     return tensor_image;
// }