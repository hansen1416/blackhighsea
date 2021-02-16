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

int main(int argc, const char *argv[])
{
    cv::Mat input_image;

    std::cout << "has open cv" << typeid(cv::Mat).name() << "\n";
    
    // if (argc != 3)
    // {
    //     std::cerr << "usage: gan-model <path-to-exported-model> <path-to-image-tensor>"
    //               << std::endl;
    //     return -1;
    // }

    // torch::jit::script::Module module;
    // try
    // {
    //     // Deserialize the ScriptModule from a file using torch::jit::load().
    //     // module = torch::jit::load(argv[1]);
    //     module = torch::jit::load("/home/hlz/blackhighsea/torch/cartoonGan/model-trace/gan-generator.pt");
    //     std::cout << typeid(module).name() << "\n";
    // }
    // catch (const c10::Error &e)
    // {
    //     std::cerr << "error loading the model\n";
    //     return -1;
    // }

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

    // // Create a vector of inputs.
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(image_tensor);

    // // Execute the model and turn its output into a tensor.
    // at::Tensor output = module.forward(inputs).toTensor();

    // torch::save(output, "oi.pt");
    
    return 0;
}