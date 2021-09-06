#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

// opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <alibabacloud/oss/OssClient.h>

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

cv::Mat stylizeImage(const std::string &model_path, const cv::Mat input_mat);

std::vector<std::string> split(const std::string &s, char delim);

#define TRUE 1
#define FALSE 0
#define PORT 8888

int main(int argc, const char *argv[])
{
    std::string params[3] = {"/opt/gan-generator.pt",
                             "https://bhs-media.oss-cn-hongkong.aliyuncs.com/imgs/trip2.jpg",
                             "https://bhs-media.oss-cn-hongkong.aliyuncs.com/imgs/trip2_out.jpg"};

    /* 初始化OSS账号信息 */
    std::string AccessKeyId = "LTAI5tLwV38wLDsnsxKEdX3f";
    std::string AccessKeySecret = "vC8Uv3jophlVnRSkNBWkqTkp9fL9F7";
    std::string Endpoint = "oss-cn-hongkong.aliyuncs.com";
    /* 填写Bucket名称，例如examplebucket */
    std::string BucketName = "bhs-media";
    /* 填写文件完整路径，例如exampledir/exampleobject.txt。文件完整路径中不能包含Bucket名称 */
    std::string ObjectName = "imgs/trip2.jpg";

    // Initialize the SDK
    AlibabaCloud::OSS::InitializeSdk();
    AlibabaCloud::OSS::ClientConfiguration conf;

    /* 设置连接池数，默认为16个。*/
    conf.maxConnections = 20;

    /* 设置请求超时时间，超时没有收到数据就关闭连接，默认为10000ms。*/
    conf.requestTimeoutMs = 8000;

    /* 设置建立连接的超时时间，默认为5000ms。*/
    conf.connectTimeoutMs = 8000;

    AlibabaCloud::OSS::OssClient client(Endpoint, AccessKeyId, AccessKeySecret, conf);

    /*获取文件到本地内存。*/
    AlibabaCloud::OSS::GetObjectRequest getrequest(BucketName, ObjectName);

    std::cout << "starting get object" << std::endl;

    auto getOutcome = client.GetObject(getrequest);

    if (!getOutcome.isSuccess())
    {
        /*异常处理。*/
        std::cout << "getObjectToBuffer fail"
                  << ",code:" << getOutcome.error().Code() << ",message:" << getOutcome.error().Message() << ",requestId:" << getOutcome.error().RequestId() << std::endl;
        AlibabaCloud::OSS::ShutdownSdk();
        return -1;
    }

    std::cout << "getObjectToBuffer"
              << " success, Content-Length:" << getOutcome.result().Metadata().ContentLength() << std::endl;
    /*通过read接口读取数据。*/
    // auto& stream = getOutcome.result().Content();

    int size = getOutcome.result().Metadata().ContentLength();

    // auto &stream = getOutcome.result().Content();
    auto stream = getOutcome.result().Content();

    char buffer[size];
    while (stream->good())
    {
        stream->read(buffer, size);
        // auto count = stream->gcount();
        /*根据实际情况处理数据。*/
    }

    // char str[size];
    // *(getOutcome.result().Content()) >> str;

    std::cout << sizeof(buffer) << std::endl;

    cv::Mat my_mat = cv::Mat(400, 400, CV_8UC3, &buffer[0]);
    // cv::Mat my_mat = cv::imread(&str[0]);

    std::cout << my_mat.size() << std::endl;

    std::cout << "start stylizeImage" << std::endl;

    // starting stylize image

    std::string model_path = "/opt/gan-generator.pt";

    cv::Mat output_mat = stylizeImage(model_path, my_mat);

    // end stylizing image

    std::cout << "something ends" << output_mat.size() << std::endl;

    return 0;

    std::shared_ptr<std::iostream> content = std::make_shared<std::stringstream>();
    *content << "Thank you for using Alibaba Cloud Object Storage Service!";
    AlibabaCloud::OSS::PutObjectRequest putRequest(BucketName, ObjectName, content);

    /* 上传文件 */
    auto putOutcome = client.PutObject(putRequest);

    if (!putOutcome.isSuccess())
    {
        /* 异常处理 */
        std::cout << "PutObject fail"
                  << ",code:" << putOutcome.error().Code() << ",message:" << putOutcome.error().Message() << ",requestId:" << putOutcome.error().RequestId() << std::endl;
        /* 释放网络等资源。*/
        AlibabaCloud::OSS::ShutdownSdk();
        return -1;
    }

    std::string hostName = "https://bhs-media.oss-cn-hongkong.aliyuncs.com/";

    /* 释放网络等资源 */
    AlibabaCloud::OSS::ShutdownSdk();

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

cv::Mat stylizeImage(const std::string &model_path, const cv::Mat input_mat)
{
    cv::Mat input_image;
    cv::Mat output_image;
    // // handle (-215:Assertion failed) !_src.empty() in function 'cvtColor'
    // cv::Mat read_image = cv::imread(input_image_path);
    // if (read_image.empty() || !read_image.data)
    //     std::cout << "read image fail" << input_image_path << std::endl;

    cv::cvtColor(input_mat, input_image, cv::COLOR_BGR2RGB);

    std::cout << "22" << std::endl;

    // 转换 [unsigned int] to [float]
    input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);
    torch::Tensor tensor_image = torch::from_blob(input_image.data, {1, input_image.rows, input_image.cols, 3});
    tensor_image = tensor_image.permute({0, 3, 1, 2});

    std::cout << "22222" << std::endl;

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

    std::cout << "333333" << std::endl;

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

    std::cout << "544444" << std::endl;

    cv::Mat output_mat = TensorToCVMat(temp_tensor);

    std::cout << "55555" << std::endl;

    if (output_mat.empty())
    {
        std::cout << "TensorToCVMat return empty " << std::endl;
        return output_image;
    }

    std::cout << "styliz finished" << std::endl;

    // cv::imwrite(output_image_path, output_mat);

    return output_image;
}