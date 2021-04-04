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

cv::Mat TensorToCVMat(torch::Tensor tensor);

int main(int argc, const char *argv[])
{
    // if (argc != 3)
    // {
    //     std::cerr << "usage: gan-model <path-to-exported-model> <path-to-image-tensor>"
    //               << std::endl;
    //     return -1;
    // }

    cv::Mat input_image;
    cv::Mat read_image = cv::imread("/home/hlz/blackhighsea/torch/cartoonGan/input/p2.jpg");
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
    torch::Tensor output = module.forward(inputs).toTensor();

    // # Adding 0.1 to all normalization values since the model is trained (erroneously) 
    // without correct de-normalization
    output[0][0] = output[0][0].mul_(0.229).add_(0.485+0.1);
    output[0][1] = output[0][1].mul_(0.224).add_(0.456+0.1);
    output[0][2] = output[0][2].mul_(0.225).add_(0.406+0.1);

    torch::Tensor temp_tensor = torch::rand({3, output.sizes()[2], output.sizes()[3]});

    temp_tensor[0] = output[0][2];
    temp_tensor[1] = output[0][1];
    temp_tensor[2] = output[0][0];

    // print tensor shape
    // std::cout << tempTtemp_tensorensor.sizes() << "\n";

    // std::cout << output[0][1][2][3] << output[0][1][3][3] << output[0][2][200][3] <<
    //       output[0][2][20][630] << output[0][1][200][300] << "\n";

    cv::Mat output_mat = TensorToCVMat(temp_tensor);

    if(output_mat.empty())
    {
        std::cout << "TensorToCVMat return empty " << std::endl;
        return 1;
    }

    cv::imwrite("/home/hlz/blackhighsea/torch/cartoonGan/output/o1.jpg", output_mat);
    
    return 0;
}


cv::Mat TensorToCVMat(torch::Tensor tensor)
{
    // torch.squeeze(input, dim=None, *, out=None) → Tensor
    // Returns a tensor with all the dimensions of input of size 1 removed.
    // tensor.detach
    // Returns a new Tensor, detached from the current graph.
    // permute dimension, 3x700x700 => 700x700x3
    tensor = tensor.squeeze(0).detach().permute({1, 2, 0});
    // float to 255 range
    tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
    // GPU to CPU?, may be not needed
    tensor = tensor.to(torch::kCPU);
    // shape of tensor
    int64_t height = tensor.size(0);
    int64_t width = tensor.size(1);
    // 700 x 700
    // std::cout << width << height << "\n";

    // std::cout << tensor.sizes() << "\n";

    // std::cout << tensor[0][0][0] << tensor[0][0][1] << tensor[0][0][2] << "\n";

    // Mat takes data form like {0,0,255,0,0,255,...} ({B,G,R,B,G,R,...})
    // so we must reshape tensor, otherwise we get a 3x3 grid
    auto new_tensor = tensor.reshape({width * height * 3, 1});

    // std::cout << new_tensor.sizes() << "\n";

    // std::cout << new_tensor[0] << new_tensor[1] << new_tensor[2] << "\n";

    cv::Mat imgbin(cv::Size(width, height), CV_8UC3, new_tensor.data_ptr());

    return imgbin;

    // cv::Mat resultImg(height, width, CV_8UC3);
    // //copy the data from out_tensor to resultImg
    // std::memcpy((void *) resultImg.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());

    // return resultImg;

    // tensor = tensor.to(torch::kU8).to(torch::kCPU);
    // auto sizes = tensor.sizes();
    // cv::Mat mat{cv::Size{static_cast<int>(sizes[1]) ,
    //                     static_cast<int>(sizes[0]) },
    //             CV_8UC(static_cast<int>(sizes[2])),
    //             tensor.data_ptr()};

    // //  cv::imwrite("/tmp/mat.png", mat);
    // return mat;
}

