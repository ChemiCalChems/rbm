#include <iostream>
#include <cstdint>
#include <csignal>

#include "rbm.hpp"

namespace {
auto parseMNISTCSV(std::string_view filename)
{
    std::vector<std::array<bool, 784>> result;
    std::ifstream in{filename.data()};

    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::string str;
    while (std::getline(in, str))
    {
        std::array<bool, 784> line;

        std::size_t readValues{0};
        std::istringstream is{str};
        is.ignore(std::numeric_limits<std::streamsize>::max(), ',');
        for (; readValues < 784 && !is.eof(); readValues++)
        {
            std::string value;
            std::getline(is, value, ',');
            int numValue = std::stoi(value);
            if (numValue >= 128) line[readValues] = true;
            else line[readValues] = false;
        }

        if (readValues != 784) throw readValues;
        result.emplace_back(std::move(line));
    }

    return result;
}
}

bool interrupted = false;
bool usr1 = false;

int main()
{
    std::signal(SIGINT, [](int){interrupted = true;});
    std::signal(SIGUSR1, [](int){usr1 = true;});

    auto MNISTcases{parseMNISTCSV("mnist_train.csv")};
    RBM<784, 500> rbm;

    std::size_t roundsElapsed = 0;
    for (; !interrupted; ++roundsElapsed)
    {
        std::cout << "Round #" << roundsElapsed + 1 << "\n";
        RDM<50>(rbm, 0.001, MNISTcases);

        if (usr1)
        {
            rbm.dumpParametersToFile("parameters.dat");
            usr1 = false;
        }
    }

    std::cout << "Caught sigint\n";
    rbm.dumpParametersToFile("parameters.dat");
}
