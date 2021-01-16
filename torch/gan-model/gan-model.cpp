#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "usage: gan-model <path-to-exported-model> <path-to-image-tensor>"
                  << std::endl;
        return -1;
    }

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    try
    {
        torch::jit::script::Module image_list = torch::jit::load(argv[2]);

        torch::Tensor a = image_list.attr("image_list").toTensor();

        // Load values by name
        std::cout << a << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return -1;
    }

    // // Create a vector of inputs.
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(torch::ones({1, 3, 224, 224}));

    // // Execute the model and turn its output into a tensor.
    // at::Tensor output = module.forward(inputs).toTensor();
    // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}