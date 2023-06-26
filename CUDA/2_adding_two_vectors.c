#include <iostream>
#include <vector>

// Function to add two vectors
std::vector<int> addVectors(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    std::vector<int> result;

    // Check if the sizes of the vectors are equal
    if (vec1.size() != vec2.size()) {
        std::cerr << "Error: Vectors must have the same size." << std::endl;
        return result;
    }

    // Add corresponding elements from the two vectors
    for (size_t i = 0; i < vec1.size(); ++i) {
        result.push_back(vec1[i] + vec2[i]);
    }

    return result;
}

int main() {
    std::vector<int> vector1 = {1, 2, 3, 4, 5};
    std::vector<int> vector2 = {6, 7, 8, 9, 10};

    std::vector<int> sum = addVectors(vector1, vector2);

    std::cout << "Result: ";
    for (const auto& value : sum) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}

