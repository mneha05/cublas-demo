#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define CUDA_OK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
#define CUBLAS_OK(x) do { auto e=(x); if(e!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error"); } while(0)
#define CUDNN_OK(x) do { auto e=(x); if(e!=CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(e)); } while(0)

struct EventTimer {
    cudaEvent_t start{}, stop{};
    EventTimer() { CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop)); }
    ~EventTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { CUDA_OK(cudaEventRecord(start)); }
    float end() {
        CUDA_OK(cudaEventRecord(stop));
        CUDA_OK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B,
              std::vector<float>& C, int M, int N, int K) {
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[r*K+k] * B[k*N+c];
            C[r*N+c] = sum;
        }
}

void run_cublas() {
    constexpr int M = 256, N = 256, K = 256;
    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(M*K), B(K*N), C(M*N), ref(M*N);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);
    cpu_gemm(A, B, ref, M, N, K);

    float *dA=nullptr,*dB=nullptr,*dC=nullptr;
    CUDA_OK(cudaMalloc(&dA, A.size()*sizeof(float)));
    CUDA_OK(cudaMalloc(&dB, B.size()*sizeof(float)));
    CUDA_OK(cudaMalloc(&dC, C.size()*sizeof(float)));
    CUDA_OK(cudaMemcpy(dA, A.data(), A.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, B.data(), B.size()*sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t h{};
    CUBLAS_OK(cublasCreate(&h));
    const float alpha=1.0f,beta=0.0f;

    // cuBLAS uses column-major layout; swapping operands gives row-major C = A*B.
    CUBLAS_OK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                         N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
    CUDA_OK(cudaDeviceSynchronize());

    EventTimer timer;
    constexpr int iters=50;
    timer.begin();
    for(int i=0;i<iters;++i)
        CUBLAS_OK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
    float ms = timer.end()/iters;

    CUDA_OK(cudaMemcpy(C.data(), dC, C.size()*sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "cuBLAS SGEMM " << M << "x" << K << " * " << K << "x" << N
              << " avg_ms=" << ms << " max_abs_error=" << max_abs_diff(C, ref) << "\n";

    cublasDestroy(h); cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

void run_cudnn() {
    const int N=1,C=3,H=64,W=64,K=16,R=3,S=3,pad=1,stride=1;
    cudnnHandle_t h{}; CUDNN_OK(cudnnCreate(&h));

    cudnnTensorDescriptor_t xdesc{}, ydesc{};
    cudnnFilterDescriptor_t fdesc{};
    cudnnConvolutionDescriptor_t cdesc{};
    CUDNN_OK(cudnnCreateTensorDescriptor(&xdesc));
    CUDNN_OK(cudnnCreateTensorDescriptor(&ydesc));
    CUDNN_OK(cudnnCreateFilterDescriptor(&fdesc));
    CUDNN_OK(cudnnCreateConvolutionDescriptor(&cdesc));

    CUDNN_OK(cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W));
    CUDNN_OK(cudnnSetFilter4dDescriptor(fdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K,C,R,S));
    CUDNN_OK(cudnnSetConvolution2dDescriptor(cdesc,pad,pad,stride,stride,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));

    int on,oc,oh,ow;
    CUDNN_OK(cudnnGetConvolution2dForwardOutputDim(cdesc,xdesc,fdesc,&on,&oc,&oh,&ow));
    CUDNN_OK(cudnnSetTensor4dDescriptor(ydesc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,on,oc,oh,ow));

    size_t xs=N*C*H*W, fs=K*C*R*S, ys=on*oc*oh*ow;
    float *dx=nullptr,*df=nullptr,*dy=nullptr;
    CUDA_OK(cudaMalloc(&dx,xs*sizeof(float))); CUDA_OK(cudaMalloc(&df,fs*sizeof(float))); CUDA_OK(cudaMalloc(&dy,ys*sizeof(float)));
    std::vector<float> x(xs,0.01f), f(fs,0.02f);
    CUDA_OK(cudaMemcpy(dx,x.data(),xs*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(df,f.data(),fs*sizeof(float),cudaMemcpyHostToDevice));

    cudnnConvolutionFwdAlgoPerf_t perf{};
    int returned=0;
    CUDNN_OK(cudnnGetConvolutionForwardAlgorithm_v7(h,xdesc,fdesc,cdesc,ydesc,1,&returned,&perf));
    size_t workspace_bytes=0;
    CUDNN_OK(cudnnGetConvolutionForwardWorkspaceSize(h,xdesc,fdesc,cdesc,ydesc,perf.algo,&workspace_bytes));
    void* workspace=nullptr;
    if(workspace_bytes) CUDA_OK(cudaMalloc(&workspace,workspace_bytes));

    const float alpha=1.0f,beta=0.0f;
    EventTimer timer;
    constexpr int iters=50;
    timer.begin();
    for(int i=0;i<iters;++i)
        CUDNN_OK(cudnnConvolutionForward(h,&alpha,xdesc,dx,fdesc,df,cdesc,perf.algo,
                                        workspace,workspace_bytes,&beta,ydesc,dy));
    float ms=timer.end()/iters;
    std::cout << "cuDNN Conv2D output=" << on << "x" << oc << "x" << oh << "x" << ow
              << " avg_ms=" << ms << " workspace_bytes=" << workspace_bytes << "\n";

    if(workspace) cudaFree(workspace); cudaFree(dx); cudaFree(df); cudaFree(dy);
    cudnnDestroyTensorDescriptor(xdesc); cudnnDestroyTensorDescriptor(ydesc);
    cudnnDestroyFilterDescriptor(fdesc); cudnnDestroyConvolutionDescriptor(cdesc); cudnnDestroy(h);
}

int main() {
    try {
        run_cublas();
        run_cudnn();
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
