#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

// install opencv 4.5.1, https://github.com/opencv/opencv/archive/4.5.1.zip
// opencv 头文件
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

// #include <cossdk/cos_api.h>
// #include <cossdk/os_sys_config.h>
// #include <cossdk/cos_defines.h>

#include "cos_api.h"
#include "cos_sys_config.h"
#include "cos_defines.h"

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

int stylizeImage(const std::string &model_path, const std::string &input_image_path,
                 const std::string &output_image_path);

std::vector<std::string> split(const std::string &s, char delim);

#define TRUE 1
#define FALSE 0
#define PORT 8888

int main(int argc, const char *argv[])
{
    std::string params[3] = {"/opt/gan-generator.pt", 
    "https://bhs-1254289668.cos.ap-shanghai.myqcloud.com/imgs/8dfaa8d9-c006-40cd-beb0-160b5ef51538.jpg", 
    "https://bhs-1254289668.cos.ap-shanghai.myqcloud.com/imgs/8dfaa8d9-c006-40cd-beb0-160b5ef51538_out.jpg"};

    // stylizeImage(params[0], params[1], params[2]);

    qcloud_cos::CosConfig config("./config.json");

    std::cout << "something" << std::endl;

    return 0;

    int opt = TRUE;
    int master_socket, addrlen, new_socket, client_socket[30],
        max_clients = 30, activity, i, valread, sd;
    int max_sd;
    struct sockaddr_in address;

    char buffer[1025]; //data buffer of 1K

    //set of socket descriptors
    fd_set readfds;

    //a message
    char const *message = "greeting";

    //initialise all client_socket[] to 0 so not checked
    for (i = 0; i < max_clients; i++)
    {
        client_socket[i] = 0;
    }

    //create a master socket
    if ((master_socket = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    //set master socket to allow multiple connections ,
    //this is just a good habit, it will work without this
    if (setsockopt(master_socket, SOL_SOCKET, SO_REUSEADDR, (char *)&opt,
                   sizeof(opt)) < 0)
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    //type of socket created
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    //bind the socket to localhost port 8888
    if (bind(master_socket, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    printf("Listener on port %d \n", PORT);

    //try to specify maximum of 3 pending connections for the master socket
    if (listen(master_socket, 3) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    //accept the incoming connection
    addrlen = sizeof(address);
    puts("Waiting for connections ...");

    while (TRUE)
    {
        //clear the socket set
        FD_ZERO(&readfds);

        //add master socket to set
        FD_SET(master_socket, &readfds);
        max_sd = master_socket;

        //add child sockets to set
        for (i = 0; i < max_clients; i++)
        {
            //socket descriptor
            sd = client_socket[i];

            //if valid socket descriptor then add to read list
            if (sd > 0)
                FD_SET(sd, &readfds);

            //highest file descriptor number, need it for the select function
            if (sd > max_sd)
                max_sd = sd;
        }

        //wait for an activity on one of the sockets , timeout is NULL ,
        //so wait indefinitely
        activity = select(max_sd + 1, &readfds, NULL, NULL, NULL);

        if ((activity < 0) && (errno != EINTR))
        {
            printf("select error");
        }

        //If something happened on the master socket ,
        //then its an incoming connection
        if (FD_ISSET(master_socket, &readfds))
        {
            if ((new_socket = accept(master_socket,
                                     (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0)
            {
                perror("accept");
                exit(EXIT_FAILURE);
            }

            //inform user of socket number - used in send and receive commands
            printf("New connection , socket fd is %d , ip is : %s , port : %d \n",
                   new_socket, inet_ntoa(address.sin_addr), ntohs(address.sin_port));

            //send new connection greeting message
            if (send(new_socket, message, strlen(message), 0) != strlen(message))
            {
                perror("send");
            }

            puts("Welcome message sent successfully");

            //add new socket to array of sockets
            for (i = 0; i < max_clients; i++)
            {
                //if position is empty
                if (client_socket[i] == 0)
                {
                    client_socket[i] = new_socket;
                    printf("Adding to list of sockets as %d\n", i);

                    break;
                }
            }
        }

        //else its some IO operation on some other socket
        for (i = 0; i < max_clients; i++)
        {
            sd = client_socket[i];

            if (FD_ISSET(sd, &readfds))
            {
                //Check if it was for closing , and also read the
                //incoming message
                if ((valread = read(sd, buffer, 1024)) == 0)
                {
                    //Somebody disconnected , get his details and print
                    getpeername(sd, (struct sockaddr *)&address,
                                (socklen_t *)&addrlen);
                    printf("Host disconnected , ip %s , port %d \n",
                           inet_ntoa(address.sin_addr), ntohs(address.sin_port));

                    //Close the socket and mark as 0 in list for reuse
                    close(sd);
                    client_socket[i] = 0;
                }

                //Echo back the message that came in
                else
                {
                    //set the string terminating NULL byte on the end
                    //of the data read
                    buffer[valread] = '\0';
                    // params is "model_path input_image_path output_image_path"
                    std::vector<std::string> params = split(buffer, ' ');

                    if (params.empty() || params.size() != 3)
                    {
                        char const *error_msg = "Wrong input";

                        send(sd, error_msg, strlen(error_msg), 0);
                    }
                    else
                    {

                        stylizeImage(params.at(0), params.at(1), params.at(2));

                        std::cout << "stylize finished" << std::endl;

                        char const *output = params.at(2).c_str();

                        send(sd, output, strlen(output), 0);
                    }

                    // char const *model_path = "/home/hlz/blackhighsea/torch/cartoonGan/model-trace/gan-generator.pt";
                    // char const *input_image_path = "/home/hlz/Pictures/2.jpg";
                    // char const *output_image_path = "/home/hlz/Pictures/o2.jpg";
                }
            }
        }
    }

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

int stylizeImage(const std::string &model_path, const std::string &input_image_path,
                 const std::string &output_image_path)
{
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
        return -1;
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
        return 1;
    }

    cv::imwrite(output_image_path, output_mat);

    return 0;
}
