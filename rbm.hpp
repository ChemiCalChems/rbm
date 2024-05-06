#pragma once

#include <iostream>
#include <cstdint>
#include <array>
#include <span>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <string_view>
#include <vector>
#include <sstream>
#include <string>
#include <cstdlib>
#include <random>
#include <ctime>
#include <execution>
#include <iomanip>
#include <ranges>
#include <csignal>

template<std::size_t V, std::size_t H>
struct RBM
{
    std::array<float, V> visibleBiases{};
    std::array<float, H> hiddenBiases{};
    std::array<std::array<float, H>, V> weights{};

    float energy(std::span<bool, V> visibleData, std::span<bool, H> hiddenData) const
    {
        float result{0.f};
        result -= std::inner_product(visibleBiases.begin(), visibleBiases.end(), visibleData.begin(), 0.f);
        result -= std::inner_product(hiddenBiases.begin(), hiddenBiases.end(), hiddenData.begin(), 0.f);
        for (std::size_t i = 0; i < V; i++)
        {
            result -= visibleData[i] * std::inner_product(weights[i].begin(), weights[i].end(), hiddenData.begin(), 0.f);
        }

        return result;
    }

    void dumpParametersToFile(std::string_view filename) const
    {
        std::ofstream out{filename.data()};
        for (std::size_t i = 0; i < V; i++)
        {
            out << visibleBiases[i] << " ";
        }
        out << "\n";
        for (std::size_t j = 0; j < H; j++)
        {
            out << hiddenBiases[j] << " ";
        }
        out << "\n";
        for (std::size_t i = 0; i < V; i++)
        {
            for (std::size_t j = 0; j < H; j++)
            {
                out << weights[i][j] << " ";
            }
        }
        out << "\n";
    }
    void loadParametersFromFile(std::string_view filename)
    {
        std::ifstream in{filename.data()};

        for (std::size_t i = 0; i < V; i++)
        {
            in >> visibleBiases[i];
        }

        for (std::size_t j = 0; j < H; j++)
        {
            in >> hiddenBiases[j];
        }

        for (std::size_t i = 0; i < V; i++)
        {
            for (std::size_t j = 0; j < H; j++)
            {
                in >> weights[i][j];
            }
        }
    }
};

template<typename T>
T sigmoid(T x)
{
    return static_cast<T>(1)/(static_cast<T>(1) + std::exp(-x));
}

template<std::size_t V, std::size_t H>
void AlternatingGibbsSingleStep(std::span<bool, V> visibleState, std::span<bool, H> hiddenState, const RBM<V,H>& rbm)
{
    std::random_device rd;

    auto hiddenStateEnumerated = hiddenState | std::views::enumerate;
#pragma omp parallel for
    for (auto&& [j, value] : hiddenStateEnumerated)
    {
        float sumVWs{0};
        for (std::size_t i = 0; i < V; ++i)
        {
            sumVWs += visibleState[i] * rbm.weights[i][j];
        }
        value = (std::uniform_real_distribution{}(rd) < sigmoid(rbm.hiddenBiases[j] + sumVWs));
    }

    auto visibleStateEnumerated = visibleState | std::views::enumerate;
#pragma omp parallel for
    for (auto&& [i, value] : visibleStateEnumerated)
    {
        float sumWHs{0};
        for (std::size_t j = 0; j < H; ++j)
        {
            sumWHs += hiddenState[j] * rbm.weights[i][j];
        }
        value = (std::uniform_real_distribution{}(rd) < sigmoid(rbm.visibleBiases[i] + sumWHs));
    }
}

template<std::size_t K, std::size_t MiniBatchSize = 10, std::size_t V, std::size_t H>
void RDM(RBM<V, H>& rbm, float learningParameter, const std::vector<std::array<bool, V>>& trainingSet)
{
    std::array<bool, V> visibleState;
    std::array<bool, H> hiddenState;

    std::random_device rd;
    std::srand(std::time(nullptr));
    std::generate(std::execution::par, visibleState.begin(), visibleState.end(), []{return std::rand() % 2 == 0;});
    std::generate(std::execution::par, hiddenState.begin(), hiddenState.end(), []{return std::rand() % 2 == 0;});

    for (std::size_t step = 0; step < K; ++step)
    {
        AlternatingGibbsSingleStep(std::span{visibleState}, std::span{hiddenState}, rbm);
    }

    std::array<std::size_t, MiniBatchSize> miniBatchIndeces; 
    std::generate(std::execution::par, miniBatchIndeces.begin(), miniBatchIndeces.end(), [&rd, n=trainingSet.size()]{return std::uniform_int_distribution<std::size_t>{0, n - 1}(rd);});

#pragma omp parallel for
    for (std::size_t j = 0; j < H; ++j)
    {
        std::array<bool, MiniBatchSize> reconstructedHJs;
        for (std::size_t k = 0; k < MiniBatchSize; ++k)
        {
            float sumVWs{0};
            for (std::size_t i = 0; i < V; ++i)
            {
                sumVWs += trainingSet[miniBatchIndeces[k]][i]*rbm.weights[i][j];
            }
            reconstructedHJs[k] = (std::uniform_real_distribution{}(rd) < sigmoid(rbm.hiddenBiases[j] + sumVWs));
        }
        rbm.hiddenBiases[j] += learningParameter * (std::accumulate(reconstructedHJs.begin(), reconstructedHJs.end(), 0.f) / MiniBatchSize - hiddenState[j]);

        for (std::size_t i = 0; i < V; ++i)
        {
            float positiveTerm{0};
            for (std::size_t k = 0; k < MiniBatchSize; ++k)
            {
                positiveTerm += trainingSet[miniBatchIndeces[k]][i] * reconstructedHJs[k];
            }
            positiveTerm /= MiniBatchSize;

            rbm.weights[i][j] += learningParameter * (positiveTerm - static_cast<float>(visibleState[i] * hiddenState[j]));
        }
    }
#pragma omp parallel for
    for (std::size_t i = 0; i < V; ++i)
    {
        float positiveTerm{0};
        for (std::size_t k = 0; k < MiniBatchSize; ++k)
        {
            positiveTerm += trainingSet[miniBatchIndeces[k]][i];
        }
        positiveTerm /= MiniBatchSize;

        rbm.visibleBiases[i] += learningParameter * (positiveTerm - visibleState[i]);
    }
    for (std::size_t k = 0; k < MiniBatchSize; ++k)
    {
        std::size_t bitsWrong{0};
        for (std::size_t i = 0; i < V; i++)
        {
            if (visibleState[i] != trainingSet[miniBatchIndeces[k]][i])
            {
                ++bitsWrong;
            }
        }
        std::cout << "Minibatch sample #" << std::setw(1) << k << ": " << std::setw(4) << bitsWrong << " bits wrong, " << static_cast<float>(bitsWrong) / V << "\n";
    }
}
