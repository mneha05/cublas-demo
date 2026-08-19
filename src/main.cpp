#include <iostream>
#include <vector>

int main() {
    std::cout << "cuBLAS demo scaffold (CPU fallback)" << std::endl;
    const int N = 5;
    std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N);
    for (int i = 0; i < N; ++i) c[i] = a[i] + b[i];
    std::cout << "result:";
    for (auto v : c) std::cout << " " << v;
    std::cout << std::endl;
    std::cout << "To integrate cuBLAS: link against cublas and replace with cublasSaxpy/cublasSdot etc." << std::endl;
    return 0;
}
