#include "rbm.hpp"
#include <array>
#include <cstdlib>
#include <ctime>
#include <execution>
#include <algorithm>
#include <ranges>

#include "BitmapPlusPlus.hpp"

int main()
{
    constexpr std::size_t V = 784;
    constexpr std::size_t H = 500;

    constexpr unsigned char graphsInX = 10;
    constexpr unsigned char graphsInY = 10;

    constexpr std::size_t mcmcSteps = 50;
    constexpr std::size_t pixelSize = 5;

    RBM<784, 500> rbm;
    rbm.loadParametersFromFile("parameters.dat");

    bmp::Bitmap image{28 * pixelSize * graphsInX, 28 * pixelSize * graphsInY};

    std::srand(std::time(nullptr));

    auto view{std::views::cartesian_product(std::views::iota(static_cast<unsigned char>(0), graphsInX), std::views::iota(static_cast<unsigned char>(0), graphsInY))};

#pragma omp parallel for
    for (const auto& [graphX, graphY] : view)
    {
        std::array<bool, V> visibleState;
        std::array<bool, H> hiddenState;

        std::generate(std::execution::par, visibleState.begin(), visibleState.end(), []{return std::rand() % 2 == 0;});
        std::generate(std::execution::par, hiddenState.begin(), hiddenState.end(), []{return std::rand() % 2 == 0;});

        for (std::size_t i = 0; i < mcmcSteps; ++i)
        {
            AlternatingGibbsSingleStep(std::span{visibleState}, std::span{hiddenState}, rbm);
        }

        for (unsigned int i = 0; i < V; ++i)
        {
            auto pixelX{graphX * 28 + i%28};
            auto pixelY{graphY * 28 + i/28};

            image.fill_rect(pixelX * pixelSize, pixelY * pixelSize, pixelSize, pixelSize, visibleState[i] ? bmp::White : bmp::Black);
        }
    }

    image.save("output.bmp");
}
