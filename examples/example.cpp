#include <faiss/IndexHNSW.h>
#include <iostream>
#include <vector>
#include <random>

int main() {
    int d = 128;      // Dimension of the vectors
    int M = 32;       // Number of neighbors per layer in HNSW
    int nb = 10000;   // Number of database vectors
    int nq = 10;      // Number of queries
    int k = 5;        // Number of nearest neighbors to retrieve

    // Create an HNSW Flat index
    faiss::IndexHNSWFlat index(d, M);
    std::cout << "HNSW index created with dimension: " << d << " and M: " << M << std::endl;

    // Randomly generate database vectors
    std::vector<float> xb(nb * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < xb.size(); i++) {
        xb[i] = dis(gen);  // Random float in [0, 1]
    }

    // Add vectors to the index
    index.add(nb, xb.data());
    std::cout << nb << " database vectors added." << std::endl;

    // Randomly generate query vectors
    std::vector<float> xq(nq * d);
    for (size_t i = 0; i < xq.size(); i++) {
        xq[i] = dis(gen);  // Random float in [0, 1]
    }

    // Perform a search for each query
    std::vector<faiss::idx_t> labels(nq * k);
    std::vector<float> distances(nq * k);

    index.search(nq, xq.data(), k, distances.data(), labels.data());

    // Output the results
    for (int i = 0; i < nq; i++) {
        std::cout << "Query " << i << " results:" << std::endl;
        for (int j = 0; j < k; j++) {
            std::cout << "  Neighbor " << j << ": Index = " << labels[i * k + j]
                      << ", Distance = " << distances[i * k + j] << std::endl;
        }
    }

    return 0;
}

