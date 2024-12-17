#include <faiss/IndexHNSW.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

// Function to read vectors from a file
std::vector<float> read_vectors(const std::string& filename, int& num_vectors, int& dimension) {
    std::ifstream file(filename);
    std::string line;
    std::vector<float> vectors;
    num_vectors = 0;
    dimension = 0;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string value;
        int current_dim = 0;

        while (std::getline(ss, value, ',')) {
            vectors.push_back(std::stof(value));
            current_dim++;
        }

        if (dimension == 0) {
            dimension = current_dim; // Set the dimension based on the first line
        }
        num_vectors++;
    }

    file.close();
    return vectors;
}

int main() {
    // Filenames for dataset and query vectors
    const std::string dataset_file = "/mnt/nfs/home/ppesaladinne/work1/datasets/msspacev/msspacev_data_1M.csv";
    const std::string query_file = "/mnt/nfs/home/ppesaladinne/work1/datasets/msspacev/msspacev_query.csv";

    // Read dataset vectors
    int nb, d;
    std::vector<float> xb = read_vectors(dataset_file, nb, d);

    // Read query vectors
    int nq, q_dim;
    std::vector<float> xq = read_vectors(query_file, nq, q_dim);

    // Ensure the dimensions match
    if (d != q_dim) {
        std::cerr << "Error: Dataset and query dimensions do not match!" << std::endl;
        return 1;
    }

    // HNSW parameters
    int M = 16;  // Number of neighbors per layer
    int efSearch = 32;  // Search parameter for HNSW
    int k = 10;   // Number of nearest neighbors to retrieve

    // Create and initialize the HNSW index
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efSearch = efSearch;
    std::cout << "HNSW index created with dimension: " << d << ", M: " << M << ", efSearch: " << efSearch << std::endl;

    // Add dataset vectors to the index
    index.add(nb, xb.data());
    std::cout << "Added " << nb << " database vectors to the index." << std::endl;

    // Perform a search for each query
    std::vector<faiss::idx_t> labels(nq * k);
    std::vector<float> distances(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    // Output results
    std::cout << "Search Results:" << std::endl;
    for (int i = 0; i < nq; i++) {
        std::cout << "Query " << i << ":" << std::endl;
        for (int j = 0; j < k; j++) {
            int idx = i * k + j;
            std::cout << "  Neighbor " << j << ": Index = " << labels[idx]
                      << ", Distance = " << distances[idx] << std::endl;
        }
    }

    return 0;
}

