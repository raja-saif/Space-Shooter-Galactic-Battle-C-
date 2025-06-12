#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h> // Keep for potential profiling needs
#include <cublas_v2.h>         // Added for cuBLAS (Tensor Cores)
#include <cuda_fp16.h>         // Added for half precision types

// Network parameters (from V3)
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 1 // Same as V1/V2/V3
#define EPOCHS 3
#define NUM_CLASSES 10

// Batch processing (from V3)
#define MAX_BATCH_SIZE 256 // Keep batching
#define NUM_STREAMS 4      // Keep streams

// Thread block configuration (Keep optimized V3 sizes)
#define BLOCK_SIZE_1D 256
#define BLOCK_SIZE_X 16 // For 2D kernels
#define BLOCK_SIZE_Y 16

// Tensor Core alignment not explicitly enforced here by padding, relying on cuBLAS/GPU to handle.
// Modern cuBLAS is often efficient even without strict padding for these modest sizes,
// but for max performance, padding dimensions to multiples of 8 (or larger for newer archs)
// might be considered. For simplicity and matching the input V4 structure, we omit explicit padding.

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS error in %s:%d: Status %d\n", __FILE__, __LINE__, \
              status);                                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)


// Timer function (Unchanged)
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// GPU timer functions using CUDA events (Unchanged from V3)
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} GpuTimer;

void startGpuTimer(GpuTimer *timer) {
    CUDA_CHECK(cudaEventCreate(&timer->start));
    CUDA_CHECK(cudaEventCreate(&timer->stop));
    CUDA_CHECK(cudaEventRecord(timer->start));
}

float stopGpuTimer(GpuTimer *timer) {
    float milliseconds = 0;
    CUDA_CHECK(cudaEventRecord(timer->stop));
    CUDA_CHECK(cudaEventSynchronize(timer->stop)); // Essential for accuracy
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, timer->start, timer->stop));
    CUDA_CHECK(cudaEventDestroy(timer->start));
    CUDA_CHECK(cudaEventDestroy(timer->stop));
    return milliseconds / 1000.0f; // Convert to seconds
}

// Host memory allocation helpers (Unchanged from V3)
double *allocateHostVector(size_t size) {
    double *vec = (double *)malloc(size * sizeof(double));
    if (!vec) {
        perror("Failed to allocate host vector");
        exit(EXIT_FAILURE);
    }
    return vec;
}

double** allocateHostMatrixPtrs(int rows) {
     double** mat = (double**)malloc(rows * sizeof(double*));
     if (!mat) {
         perror("Failed to allocate host matrix rows");
         exit(EXIT_FAILURE);
     }
     return mat;
}

void freeHostMatrixPtrs(double** mat, int rows) {
     if(mat) {
        free(mat);
     }
}

double** allocateStandardMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    if (!mat) {
        perror("Failed to allocate standard matrix rows");
        exit(EXIT_FAILURE);
    }
    double *storage = (double *)malloc((size_t)rows * cols * sizeof(double));
    if (!storage) {
        perror("Failed to allocate standard matrix storage");
        free(mat);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++) {
        mat[i] = storage + i * cols; // Point row pointers into the contiguous block
    }
    return mat;
}

void freeStandardMatrix(double **mat) {
    if (mat) {
        if (mat[0]) {
            free(mat[0]); // Free the contiguous block
        }
        free(mat); // Free the row pointers
    }
}

// --- Kernels for Precision Conversion ---
// --- Kernels for Precision Conversion ---

// Convert double to half precision
__global__ void doubleToHalfKernel(const double* src, half* dst, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Corrected: Convert double -> float -> half
        dst[idx] = __float2half((float)src[idx]);
    }
}

// Convert half to double precision
__global__ void halfToDoubleKernel(const half* src, double* dst, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Corrected: Convert half -> float -> double
        dst[idx] = (double)__half2float(src[idx]);
    }
}

// Convert float to half precision (This one is usually correct)
__global__ void floatToHalfKernel(const float* src, half* dst, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __float2half(src[idx]);
    }
}

// Convert half to float precision (This one is usually correct)
__global__ void halfToFloatKernel(const half* src, float* dst, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __half2float(src[idx]);
    }
}

// Convert float to double precision (Unchanged, usually fine)
__global__ void floatToDoubleKernel(const float* src, double* dst, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = (double)src[idx]; // Simple cast
    }
}

// Convert double to float precision (Unchanged, usually fine)
__global__ void doubleToFloatKernel(const double* src, float* dst, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = (float)src[idx]; // Simple cast
    }
}


// Neural network structure - V4 modified for Tensor Cores
typedef struct {
    // Master Device matrices (double precision for updates)
    double *d_W1;
    double *d_W2;
    double *d_b1;
    double *d_b2;

    // Half precision copies for Tensor Core GEMM input
    half *d_W1_half;
    half *d_W2_half;
    half *d_inputs_half; // Input batch converted to half
    half *d_hiddens_half; // Hidden activations converted back to half for layer 2 GEMM

    // Float precision buffers for intermediate GEMM results and activations
    float *d_gemm1_output_float; // Stores result of W1*inputs (FP32)
    float *d_hiddens_float;      // Stores hidden activations after bias+ReLU (FP32)
    float *d_gemm2_output_float; // Stores result of W2*hiddens (FP32)
    float *d_pre_softmax_float; // Stores output pre-softmax after bias (FP32)

    // Double precision buffers (from V3) for data I/O and host softmax compatibility
    double *d_inputs;           // Current batch input (double, temporary for copy)
    double *d_outputs;          // Final layer output post-softmax (double)
    double *d_pre_softmax_outputs; // Stores pre-softmax result for host transfer (double)
    double *d_targets;          // Current batch target labels (double)

    // Float precision buffers for gradients
    float *d_d_outputs_float;   // Gradient w.r.t. output layer (post-softmax, pre-activation for layer 2?) (FP32) - Error (y_pred - y_true)
    float *d_d_hiddens_float;   // Gradient w.r.t. hidden layer (post-ReLU activation) (FP32)

    // cuBLAS handle
    cublasHandle_t cublas_handle;

    // CUDA streams for overlapping operations
    cudaStream_t streams[NUM_STREAMS];

} NeuralNetwork;


// --- V4 Kernels Modified for Mixed Precision ---

// Apply bias and ReLU activation: GEMM output (FP32) + bias (double) -> Activation output (FP32)
__global__ void applyBiasReluFloatKernel(float *output_activation, const float *gemm_output, const double *bias,
                                     int features, int batch_size) {
    int f_idx = blockIdx.x * blockDim.x + threadIdx.x; // Feature index
    int b_idx = blockIdx.y;                            // Batch index

    if (f_idx < features && b_idx < batch_size) {
        size_t offset = (size_t)b_idx * features + f_idx;
        // Add bias (convert double bias to float) and apply ReLU
        float biased_val = gemm_output[offset] + (float)bias[f_idx];
        output_activation[offset] = fmaxf(biased_val, 0.0f); // Apply ReLU
    }
}

// Apply bias (no activation): Used before softmax transfer
__global__ void applyBiasFloatKernel(float *output, const float *gemm_output, const double *bias,
                                     int features, int batch_size) {
    int f_idx = blockIdx.x * blockDim.x + threadIdx.x; // Feature index
    int b_idx = blockIdx.y;                            // Batch index

    if (f_idx < features && b_idx < batch_size) {
        size_t offset = (size_t)b_idx * features + f_idx;
        // Add bias (convert double bias to float)
        output[offset] = gemm_output[offset] + (float)bias[f_idx];
    }
}


// Compute output layer error (post-softmax(double) - target(double)) -> d_d_outputs (float)
// This corrects the logic from the provided V4 snippet.
__global__ void computeOutputErrorFloatKernel(const double *d_outputs_post_softmax, const double *d_targets,
                                           float *d_d_outputs_float, // Output gradient is float
                                           int output_size, int batch_size) {
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x; // Index within output vector
    int batch_idx = blockIdx.y;                      // Index for sample in batch

    if (o_idx < output_size && batch_idx < batch_size) {
        size_t offset = (size_t)batch_idx * output_size + o_idx;
        // Calculate output error (output - target) and store as float
        d_d_outputs_float[offset] = (float)(d_outputs_post_softmax[offset] - d_targets[offset]);
    }
}


// Compute hidden layer error gradient using FP32/FP16
// d_d_hiddens (float) = (W2_half^T * d_d_outputs_float) * relu_deriv(hiddens_float)
__global__ void computeHiddenErrorFloatKernel(const half *d_W2_half, const float *d_d_outputs_float,
                                          const float *d_hiddens_float, float *d_d_hiddens_float, // Output gradient is float
                                          int hidden_size, int output_size, int batch_size) {
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x; // Index for hidden unit
    int batch_idx = blockIdx.y;                        // Index for sample in batch

    if (h_idx < hidden_size && batch_idx < batch_size) {
        size_t hidden_offset = (size_t)batch_idx * hidden_size;
        size_t output_offset = (size_t)batch_idx * output_size;

        float error_sum = 0.0f;

        // Compute W2^T * d_d_outputs_float for column h_idx
        // W2 is output_size x hidden_size (row major storage assumed from init)
        // W2_half accessed W2[o_idx * hidden_size + h_idx] corresponds to W2[o_idx][h_idx]
        for (int o_idx = 0; o_idx < output_size; o_idx++) {
            // Access W2^T element W2[o_idx][h_idx] which is stored at W2[o_idx * hidden_size + h_idx]
            // Convert half weight to float for calculation
            float weight_ho = __half2float(d_W2_half[(size_t)o_idx * hidden_size + h_idx]);
            error_sum += weight_ho * d_d_outputs_float[output_offset + o_idx];
        }

        // Apply ReLU derivative using the FP32 hidden activations
        float activation = d_hiddens_float[hidden_offset + h_idx];
        d_d_hiddens_float[hidden_offset + h_idx] = error_sum * (activation > 0.0f ? 1.0f : 0.0f);
    }
}

// Update weights W2 (Double) using FP32/FP16 gradients/activations
// d_W2 (double) -= learning_rate * averaged_gradient(d_d_outputs_float * d_hiddens_float^T)
__global__ void updateWeightsW2FloatKernel(double *d_W2, const float *d_d_outputs_float, const float *d_hiddens_float,
                                     double learning_rate, int hidden_size, int output_size,
                                     int batch_size) {
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x; // Output index (row)
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y; // Hidden index (col)

    if (o_idx < output_size && h_idx < hidden_size) {
        float weight_update_sum = 0.0f; // Accumulate update in float

        // Sum gradient over the batch: d_output[b][o] * hidden[b][h] (All float inputs)
        for (int b = 0; b < batch_size; b++) {
            size_t output_offset = (size_t)b * output_size;
            size_t hidden_offset = (size_t)b * hidden_size;
            weight_update_sum += d_d_outputs_float[output_offset + o_idx] * d_hiddens_float[hidden_offset + h_idx];
        }

        // Average gradient over the batch
        float averaged_gradient = weight_update_sum / (float)batch_size;

        // Apply update to the double precision master weight
        size_t weight_idx = (size_t)o_idx * hidden_size + h_idx;
        d_W2[weight_idx] -= learning_rate * (double)averaged_gradient; // Update double weight
    }
}

// Update weights W1 (Double) using FP32/FP16 gradients/inputs
// d_W1 (double) -= learning_rate * averaged_gradient(d_d_hiddens_float * d_inputs_half^T)
__global__ void updateWeightsW1FloatKernel(double *d_W1, const float *d_d_hiddens_float, const half *d_inputs_half,
                                     double learning_rate, int input_size, int hidden_size,
                                     int batch_size) {
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x; // Hidden index (row)
    int i_idx = blockIdx.y * blockDim.y + threadIdx.y; // Input index (col)

    if (h_idx < hidden_size && i_idx < input_size) {
        float weight_update_sum = 0.0f; // Accumulate update in float

        // Sum gradient over the batch: d_hidden[b][h] * input[b][i]
        for (int b = 0; b < batch_size; b++) {
            size_t hidden_offset = (size_t)b * hidden_size;
            size_t input_offset = (size_t)b * input_size;
            // Convert half input to float for multiplication
            float input_val = __half2float(d_inputs_half[input_offset + i_idx]);
            weight_update_sum += d_d_hiddens_float[hidden_offset + h_idx] * input_val;
        }

        // Average gradient over the batch
        float averaged_gradient = weight_update_sum / (float)batch_size;

        // Apply update to the double precision master weight
        size_t weight_idx = (size_t)h_idx * input_size + i_idx;
        d_W1[weight_idx] -= learning_rate * (double)averaged_gradient; // Update double weight
    }
}

// Update biases (Double) using FP32 gradients
__global__ void updateBiasesFloatKernel(double *d_b1, double *d_b2, const float *d_d_hiddens_float, const float *d_d_outputs_float,
                                     double learning_rate, int hidden_size, int output_size,
                                     int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Update hidden layer biases b1[idx]
    if (idx < hidden_size) {
        float bias1_update_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            bias1_update_sum += d_d_hiddens_float[(size_t)b * hidden_size + idx];
        }
        float averaged_gradient = bias1_update_sum / (float)batch_size;
        d_b1[idx] -= learning_rate * (double)averaged_gradient; // Update double bias
    }

    // Update output layer biases b2[idx]
    if (idx < output_size) {
        float bias2_update_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            bias2_update_sum += d_d_outputs_float[(size_t)b * output_size + idx];
        }
        float averaged_gradient = bias2_update_sum / (float)batch_size;
        d_b2[idx] -= learning_rate * (double)averaged_gradient; // Update double bias
    }
}


// Host softmax implementation (Unchanged from V3)
void softmax_host(double *x, int size) {
    if (size <= 0) return;
    double max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val)
            max_val = x[i];
    }

    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val); // Improve numerical stability
        sum += x[i];
    }
    // Use a small epsilon to avoid division by zero
    if (sum < 1e-9) sum = 1e-9;

    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Initialize neural network - V4 for Tensor Cores
NeuralNetwork *createNetwork() {
    NeuralNetwork *net = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));

    // --- Initialize cuBLAS ---
    CUBLAS_CHECK(cublasCreate(&net->cublas_handle));
    // Attempt to enable Tensor Core operations. May implicitly use them if HW supports & conditions met.
    // Volta+ GPUs with FP16 compute: CUBLAS_TENSOR_OP_MATH
    // Ampere+ GPUs with FP16, BF16, TF32: CUBLAS_DEFAULT_MATH allows TF32 for FP32 GEMM, CUBLAS_TENSOR_OP_MATH enforces FP16/BF16 use.
    // We target FP16 inputs, so TENSOR_OP_MATH is appropriate.
    CUBLAS_CHECK(cublasSetMathMode(net->cublas_handle, CUBLAS_TENSOR_OP_MATH));
    printf("cuBLAS handle created and Tensor Core math mode requested.\n");

    // --- Calculate sizes ---
    size_t w1_elements = (size_t)HIDDEN_SIZE * INPUT_SIZE;
    size_t w2_elements = (size_t)OUTPUT_SIZE * HIDDEN_SIZE;
    size_t b1_elements = HIDDEN_SIZE;
    size_t b2_elements = OUTPUT_SIZE;
    size_t batch_input_elements = (size_t)MAX_BATCH_SIZE * INPUT_SIZE;
    size_t batch_hidden_elements = (size_t)MAX_BATCH_SIZE * HIDDEN_SIZE;
    size_t batch_output_elements = (size_t)MAX_BATCH_SIZE * OUTPUT_SIZE;

    // --- Allocate Device Memory ---
    // Double precision master weights/biases
    CUDA_CHECK(cudaMalloc(&net->d_W1, w1_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_W2, w2_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_b1, b1_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_b2, b2_elements * sizeof(double)));

    // Half precision copies for GEMM
    CUDA_CHECK(cudaMalloc(&net->d_W1_half, w1_elements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&net->d_W2_half, w2_elements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&net->d_inputs_half, batch_input_elements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&net->d_hiddens_half, batch_hidden_elements * sizeof(half)));

    // Float precision intermediates & activations
    CUDA_CHECK(cudaMalloc(&net->d_gemm1_output_float, batch_hidden_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->d_hiddens_float, batch_hidden_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->d_gemm2_output_float, batch_output_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->d_pre_softmax_float, batch_output_elements * sizeof(float)));


    // Double precision inputs/outputs/targets (from V3)
    CUDA_CHECK(cudaMalloc(&net->d_inputs, batch_input_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_outputs, batch_output_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_pre_softmax_outputs, batch_output_elements * sizeof(double))); // For Host softmax D->H transfer
    CUDA_CHECK(cudaMalloc(&net->d_targets, batch_output_elements * sizeof(double)));

    // Float precision gradients
    CUDA_CHECK(cudaMalloc(&net->d_d_outputs_float, batch_output_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->d_d_hiddens_float, batch_hidden_elements * sizeof(float)));

    // --- Initialize Weights/Biases on Host (Using Xavier Init from V3) ---
    double *h_W1 = allocateHostVector(w1_elements);
    double *h_W2 = allocateHostVector(w2_elements);
    double *h_b1 = allocateHostVector(b1_elements);
    double *h_b2 = allocateHostVector(b2_elements);

    srand(time(NULL));
    double w1_bound = sqrt(6.0 / (INPUT_SIZE + HIDDEN_SIZE));
    double w2_bound = sqrt(6.0 / (HIDDEN_SIZE + OUTPUT_SIZE));
    for (size_t i = 0; i < w1_elements; i++) h_W1[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * w1_bound;
    for (size_t i = 0; i < w2_elements; i++) h_W2[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * w2_bound;
    memset(h_b1, 0, b1_elements * sizeof(double));
    memset(h_b2, 0, b2_elements * sizeof(double));

    // --- Copy Initial Weights/Biases Host -> Device (Double Precision) ---
    CUDA_CHECK(cudaMemcpy(net->d_W1, h_W1, w1_elements * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_W2, h_W2, w2_elements * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b1, h_b1, b1_elements * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b2, h_b2, b2_elements * sizeof(double), cudaMemcpyHostToDevice));

    // --- Convert Initial Double Weights -> Half Weights on Device ---
    dim3 convBlock(BLOCK_SIZE_1D);
    dim3 convGridW1((w1_elements + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);
    dim3 convGridW2((w2_elements + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);
    doubleToHalfKernel<<<convGridW1, convBlock>>>(net->d_W1, net->d_W1_half, w1_elements);
    doubleToHalfKernel<<<convGridW2, convBlock>>>(net->d_W2, net->d_W2_half, w2_elements);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for conversion to finish before freeing host arrays

    // --- Clean up temporary host arrays ---
    free(h_W1); free(h_W2); free(h_b1); free(h_b2);

    // --- Create CUDA Streams ---
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&net->streams[i], cudaStreamNonBlocking));
    }

    return net;
}

// Forward pass for a batch using Tensor Cores - V4
void forwardBatch(NeuralNetwork *net, double **batch_images, double **batch_outputs_final, int batch_size, int stream_idx) {
    cudaStream_t stream = net->streams[stream_idx];

    // Set cuBLAS to use the current stream
    CUBLAS_CHECK(cublasSetStream(net->cublas_handle, stream));

    // === Host Side Prep ===
    // 1. Prepare flat input batch on HOST (double)
    size_t input_batch_elements = (size_t)batch_size * INPUT_SIZE;
    size_t input_batch_bytes_double = input_batch_elements * sizeof(double);
    double *h_input_batch = allocateHostVector(input_batch_elements);
    for (int i = 0; i < batch_size; i++) {
        memcpy(h_input_batch + (size_t)i * INPUT_SIZE, batch_images[i], INPUT_SIZE * sizeof(double));
    }

    // === Device Operations ===
    // 2. Copy flat input batch HOST -> DEVICE (double, Async) -> net->d_inputs
    CUDA_CHECK(cudaMemcpyAsync(net->d_inputs, h_input_batch, input_batch_bytes_double, cudaMemcpyHostToDevice, stream));

    // 3. Convert inputs: Device double -> half (Async) -> net->d_inputs_half
    dim3 convGridInput((input_batch_elements + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);
    dim3 convBlockInput(BLOCK_SIZE_1D);
    doubleToHalfKernel<<<convGridInput, convBlockInput, 0, stream>>>(
        net->d_inputs, net->d_inputs_half, input_batch_elements
    );

    // 4. Layer 1 GEMM: W1_half * Inputs_half -> gemm1_output_float (FP16*FP16 -> FP32)
    // Use cublasGemmEx for mixed precision with Tensor Core support
    const float alpha_fp32 = 1.0f;
    const float beta_fp32 = 0.0f;

    // C[MxN] = op(A)[MxK] * op(B)[KxN]
    // Our calculation: Hidden_out[Hidden x Batch] = W1[Hidden x Input] * Inputs[Input x Batch]
    // V4 Code Interpretation:
    // M = HIDDEN_SIZE, N = batch_size, K = INPUT_SIZE
    // op(A) = W1_half^T (lda is input_size) -> Original A assumed KxM (Input x Hidden)
    // op(B) = Inputs_half^N (ldb is input_size) -> Original B assumed KxN (Input x Batch)
    // C = gemm1_output_float (ldc is hidden_size) -> Result is M x N (Hidden x Batch) - CORRECT

    // We assume W1_half is HxI (row-major) and Inputs_half is BxI (row-major)
    // We compute C = W1 * Inputs^T. So A=W1 (N), B=Inputs (T) ? No, C is H x B
    // Let's trust the V4 parameters which compute C = (W1^T)^T * Inputs ? Still seems off.
    // Alternative: C^T = Inputs^T * W1^T
    // M=Batch, N=Hidden, K=Input. A=Inputs, B=W1. op(A)=T, op(B)=T
    // Call: cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, B, H, I, ..., A=Inputs(B*I, lda=I), ..., B=W1(H*I, ldb=I), ..., C=Out(B*H, ldc=H)
    // Let's stick to the provided V4's interpretation/call structure for now.
    CUBLAS_CHECK(cublasGemmEx(net->cublas_handle,
                      CUBLAS_OP_T,                          // Transpose A = W1_half
                      CUBLAS_OP_N,                          // No Transpose B = Inputs_half
                      HIDDEN_SIZE,                          // M: rows of C and op(A) = HIDDEN_SIZE
                      batch_size,                           // N: cols of C and op(B) = batch_size
                      INPUT_SIZE,                           // K: cols of op(A) and rows of op(B) = INPUT_SIZE
                      &alpha_fp32,                          // alpha (use float version)
                      net->d_W1_half,                       // A matrix (W1) on device
                      CUDA_R_16F,                           // Data type of A (Half)
                      INPUT_SIZE,                           // Leading dimension of A (assuming KxM storage? No, should be Input for HxI)
                      net->d_inputs_half,                   // B matrix (Inputs) on device
                      CUDA_R_16F,                           // Data type of B (Half)
                      INPUT_SIZE,                           // Leading dimension of B (assuming KxN storage? No, should be Input for BxI)
                      &beta_fp32,                           // beta (use float version)
                      net->d_gemm1_output_float,            // C matrix (Output=HiddenPreAct) on device
                      CUDA_R_32F,                           // Data type of C (Float)
                      HIDDEN_SIZE,                          // Leading dimension of C (M = Hidden size)
                      CUDA_R_32F,                           // Compute type (Use FP32 for accumulation)
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));      // Algo selection hinting Tensor Cores

    // 5. Apply Bias and ReLU (FP32) -> net->d_hiddens_float
    dim3 gridBiasRelu1((HIDDEN_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockBiasRelu1(BLOCK_SIZE_1D);
    applyBiasReluFloatKernel<<<gridBiasRelu1, blockBiasRelu1, 0, stream>>>(
        net->d_hiddens_float, net->d_gemm1_output_float, net->d_b1, // Output=FP32, Input=FP32, Bias=double
        HIDDEN_SIZE, batch_size
    );

    // 6. Convert Hidden Activations: Device float -> half (Async) -> net->d_hiddens_half
    size_t hidden_batch_elements = (size_t)batch_size * HIDDEN_SIZE;
    dim3 convGridHidden((hidden_batch_elements + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);
    floatToHalfKernel<<<convGridHidden, convBlockInput, 0, stream>>>(
        net->d_hiddens_float, net->d_hiddens_half, hidden_batch_elements
    );

    // 7. Layer 2 GEMM: W2_half * Hiddens_half -> gemm2_output_float (FP16*FP16 -> FP32)
    // C[MxN] = op(A)[MxK] * op(B)[KxN]
    // Calculation: Output_PreAct[Output x Batch] = W2[Output x Hidden] * Hiddens[Hidden x Batch]
    // V4 Code Interpretation:
    // M = OUTPUT_SIZE, N = batch_size, K = HIDDEN_SIZE
    // op(A) = W2_half^T (lda=HIDDEN) -> Original A = KxM (Hidden x Output)?
    // op(B) = Hiddens_half^N (ldb=HIDDEN) -> Original B = KxN (Hidden x Batch)? -> Correct shape & ldb
    // C = gemm2_output_float (ldc=OUTPUT) -> Result M x N (Output x Batch) -> Correct shape & ldc
    CUBLAS_CHECK(cublasGemmEx(net->cublas_handle,
                      CUBLAS_OP_T,                          // Transpose A = W2_half
                      CUBLAS_OP_N,                          // No Transpose B = Hiddens_half
                      OUTPUT_SIZE,                          // M: rows of C and op(A) = OUTPUT_SIZE
                      batch_size,                           // N: cols of C and op(B) = batch_size
                      HIDDEN_SIZE,                          // K: cols of op(A) and rows of op(B) = HIDDEN_SIZE
                      &alpha_fp32,                          // alpha
                      net->d_W2_half,                       // A matrix (W2) on device
                      CUDA_R_16F,                           // Data type of A (Half)
                      HIDDEN_SIZE,                          // Leading dimension of A (K = Hidden size?)
                      net->d_hiddens_half,                  // B matrix (Hidden Activations) on device
                      CUDA_R_16F,                           // Data type of B (Half)
                      HIDDEN_SIZE,                          // Leading dimension of B (K = Hidden size)
                      &beta_fp32,                           // beta
                      net->d_gemm2_output_float,            // C matrix (Output PreAct) on device
                      CUDA_R_32F,                           // Data type of C (Float)
                      OUTPUT_SIZE,                          // Leading dimension of C (M = Output size)
                      CUDA_R_32F,                           // Compute type (Use FP32 for accumulation)
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));      // Algo selection

    // 8. Apply Bias to Layer 2 Output (FP32) -> net->d_pre_softmax_float
    dim3 gridBias2((OUTPUT_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockBias2(BLOCK_SIZE_1D);
     applyBiasFloatKernel<<<gridBias2, blockBias2, 0, stream>>>(
        net->d_pre_softmax_float, net->d_gemm2_output_float, net->d_b2,
        OUTPUT_SIZE, batch_size
    );

    // 9. Convert Pre-Softmax Output: Device float -> double (Async) -> net->d_pre_softmax_outputs
    size_t output_batch_elements = (size_t)batch_size * OUTPUT_SIZE;
    size_t output_batch_bytes_double = output_batch_elements * sizeof(double);
    dim3 convGridOutput((output_batch_elements + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);
    floatToDoubleKernel<<<convGridOutput, convBlockInput, 0, stream>>>(
        net->d_pre_softmax_float, net->d_pre_softmax_outputs, output_batch_elements
    );

    // === Softmax on Host (as per V3) ===
    // 10. Copy pre-softmax output DEVICE -> HOST (double, Async)
    double *h_output_batch = allocateHostVector(output_batch_elements); // Host buffer for calculation
    CUDA_CHECK(cudaMemcpyAsync(h_output_batch, net->d_pre_softmax_outputs, output_batch_bytes_double, cudaMemcpyDeviceToHost, stream));

    // 11. Synchronize this stream to ensure host copy is complete BEFORE host softmax
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 12. Compute softmax on HOST for each sample in the batch
    for (int i = 0; i < batch_size; i++) {
        softmax_host(h_output_batch + (size_t)i * OUTPUT_SIZE, OUTPUT_SIZE);
    }

    // === Post Softmax Operations ===
    // 13. Copy final softmax result HOST -> DEVICE (double, into net->d_outputs for backprop and reference) (Async)
    CUDA_CHECK(cudaMemcpyAsync(net->d_outputs, h_output_batch, output_batch_bytes_double, cudaMemcpyHostToDevice, stream));

    // 14. Populate the output parameter C array if provided (for evaluation / loss calc on host)
    if (batch_outputs_final != NULL) {
        for (int i = 0; i < batch_size; i++) {
             memcpy(batch_outputs_final[i], h_output_batch + (size_t)i * OUTPUT_SIZE, OUTPUT_SIZE * sizeof(double));
        }
    }

    // 15. Clean up host temporary buffers for this batch
    free(h_input_batch);
    free(h_output_batch);

    // Note: Backprop kernels will be launched later on the same stream,
    // implicitly waiting for the step 13 memcpy to complete.
}


// Backpropagation for a batch using V4 Mixed Precision approach
void backwardBatch(NeuralNetwork *net, double **batch_labels, int batch_size, int stream_idx) {
    cudaStream_t stream = net->streams[stream_idx];

    // === Host Side Prep ===
    // 1. Prepare flat target batch on HOST (double)
    size_t output_batch_elements = (size_t)batch_size * OUTPUT_SIZE;
    size_t output_batch_bytes_double = output_batch_elements * sizeof(double);
    double *h_target_batch = allocateHostVector(output_batch_elements);
    for (int i = 0; i < batch_size; i++) {
        memcpy(h_target_batch + (size_t)i * OUTPUT_SIZE, batch_labels[i], OUTPUT_SIZE * sizeof(double));
    }

    // === Device Operations ===
    // 2. Copy flat target batch HOST -> DEVICE (double, Async) -> net->d_targets
    CUDA_CHECK(cudaMemcpyAsync(net->d_targets, h_target_batch, output_batch_bytes_double, cudaMemcpyHostToDevice, stream));

    // 3. Compute Output Error Kernel: (post_softmax(double) - targets(double)) -> d_d_outputs_float (Async)
    // Uses net->d_outputs which contains post-softmax values from forward pass H->D copy
    dim3 gridErr1((OUTPUT_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockErr1(BLOCK_SIZE_1D);
    computeOutputErrorFloatKernel<<<gridErr1, blockErr1, 0, stream>>>(
        net->d_outputs, net->d_targets, net->d_d_outputs_float,
        OUTPUT_SIZE, batch_size
    );

    // 4. Compute Hidden Error Kernel: (W2_half^T * d_d_outputs_float) .* relu_deriv(hiddens_float) -> d_d_hiddens_float (Async)
    dim3 gridErr2((HIDDEN_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockErr2(BLOCK_SIZE_1D);
    computeHiddenErrorFloatKernel<<<gridErr2, blockErr2, 0, stream>>>(
        net->d_W2_half, net->d_d_outputs_float, net->d_hiddens_float, net->d_d_hiddens_float,
        HIDDEN_SIZE, OUTPUT_SIZE, batch_size
    );

    // --- Update Kernels (using FP32 gradients/activations, updating double weights/biases) ---
    // 5. Update Weights W2 Kernel (Async)
    dim3 blockW2(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridW2((OUTPUT_SIZE + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                (HIDDEN_SIZE + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    updateWeightsW2FloatKernel<<<gridW2, blockW2, 0, stream>>>(
        net->d_W2, net->d_d_outputs_float, net->d_hiddens_float, // Use float activations
        LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE, batch_size
    );

    // 6. Update Weights W1 Kernel (Async)
    dim3 blockW1(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridW1((HIDDEN_SIZE + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                (INPUT_SIZE + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    updateWeightsW1FloatKernel<<<gridW1, blockW1, 0, stream>>>(
        net->d_W1, net->d_d_hiddens_float, net->d_inputs_half, // Use half inputs
        LEARNING_RATE, INPUT_SIZE, HIDDEN_SIZE, batch_size
    );

    // 7. Update Biases Kernel (Async)
    // Uses fmax for grid size as in V3
    dim3 blockBias(BLOCK_SIZE_1D);
    dim3 gridBias((fmax(HIDDEN_SIZE, OUTPUT_SIZE) + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);
    updateBiasesFloatKernel<<<gridBias, blockBias, 0, stream>>>(
        net->d_b1, net->d_b2, net->d_d_hiddens_float, net->d_d_outputs_float,
        LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE, batch_size
    );

    // === Convert Updated Weights for Next Iteration ===
    // 8. Convert updated double weights -> half weights (Async)
    // These must complete before the *next* forward pass uses them.
    size_t w1_elements = (size_t)HIDDEN_SIZE * INPUT_SIZE;
    size_t w2_elements = (size_t)OUTPUT_SIZE * HIDDEN_SIZE;
    dim3 convGridW1((w1_elements + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);
    dim3 convGridW2((w2_elements + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);
    dim3 convBlockCommon(BLOCK_SIZE_1D);

    doubleToHalfKernel<<<convGridW1, convBlockCommon, 0, stream>>>(net->d_W1, net->d_W1_half, w1_elements);
    doubleToHalfKernel<<<convGridW2, convBlockCommon, 0, stream>>>(net->d_W2, net->d_W2_half, w2_elements);


    // === Cleanup ===
    // 9. Clean up host temporary buffer for targets
    free(h_target_batch);

    // Synchronization will happen after the batch loop in train() or before next use.
}


// Train function (Mostly unchanged from V3 structure, just calls V4 forward/backward)
void train(NeuralNetwork *net, double **images, double **labels, int numImages) {
    printf("Starting training (V4 Tensor Core Enabled)...\n");
    clock_t total_start_cpu = clock();
    GpuTimer gpu_timer;
    startGpuTimer(&gpu_timer); // Start GPU timer

    int batch_size = MAX_BATCH_SIZE;

    // Allocate buffer for batch outputs on host ONCE (for loss/accuracy calculation)
    // Uses standard C matrix helper which allocates contiguous block
    double **batch_outputs_host = allocateStandardMatrix(batch_size, OUTPUT_SIZE);


    // Allocate host pointers for image/label batches ONCE
    double **batch_images_ptrs = allocateHostMatrixPtrs(batch_size);
    double **batch_labels_ptrs = allocateHostMatrixPtrs(batch_size);


    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start_cpu = clock();
        double epoch_loss = 0.0;
        long long correct_count = 0; // Use long long for potentially large sums

        // Shuffle indices (Fisher-Yates) - unchanged
        int *indices = (int *)malloc(numImages * sizeof(int));
        for (int i = 0; i < numImages; i++) indices[i] = i;
        for (int i = numImages - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        // Process in batches
        for (int batch_start = 0; batch_start < numImages; batch_start += batch_size) {
            int current_batch_size = fmin(batch_size, numImages - batch_start);
             if (current_batch_size <= 0) continue;

             // Resize host output buffer if last batch is smaller (memcpy needs exact size later)
             // Not strictly necessary if allocateStandardMatrix gives MAX_BATCH_SIZE, but safer.
             // We just need to use current_batch_size in loops below.

            // Prepare pointers to the current batch data
            for(int i=0; i < current_batch_size; ++i) {
                int idx = indices[batch_start + i];
                batch_images_ptrs[i] = images[idx];
                batch_labels_ptrs[i] = labels[idx];
            }

            // Determine stream
            int stream_idx = (batch_start / batch_size) % NUM_STREAMS;

            // --- Forward Pass (V4) ---
            // Resulting softmax probabilities will be in batch_outputs_host
            forwardBatch(net, batch_images_ptrs, batch_outputs_host, current_batch_size, stream_idx);
            // forwardBatch SYNCHRONIZES internally before host softmax

            // --- Loss and Accuracy Calculation (on Host using batch_outputs_host) --- Unchanged
            for (int i = 0; i < current_batch_size; i++) {
                double sample_loss = 0.0;
                int actual_label_idx = -1;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                     if (batch_labels_ptrs[i][j] > 0.5) {
                        actual_label_idx = j;
                        double prob = batch_outputs_host[i][j];
                         if (prob < 1e-9) prob = 1e-9; // Avoid log(0)
                         sample_loss = -log(prob);
                         break;
                    }
                }
                 // Handle case where label not found (should not happen with one-hot)
                 if (actual_label_idx == -1) {
                     fprintf(stderr, "Warning: Actual label not found for sample index %d in batch starting %d\n", i, batch_start);
                 } else {
                     epoch_loss += sample_loss;
                 }


                // Find predicted label for sample i
                int pred_label_idx = 0;
                for (int j = 1; j < OUTPUT_SIZE; j++) {
                    if (batch_outputs_host[i][j] > batch_outputs_host[i][pred_label_idx]) {
                        pred_label_idx = j;
                    }
                }
                if (pred_label_idx == actual_label_idx) {
                    correct_count++;
                }
            }

            // --- Backward Pass (V4) ---
            // Launched on the same stream. Waits for forward pass ops on that stream.
            backwardBatch(net, batch_labels_ptrs, current_batch_size, stream_idx);

             // Print progress (less frequently) - Unchanged
            if ((batch_start / batch_size + 1) % 50 == 0) {
                printf("Epoch %d: %d / %d samples processed.\r",
                        epoch + 1, batch_start + current_batch_size, numImages);
                 fflush(stdout);
             }

        } // End batch loop

        // Synchronize all streams at the end of the epoch
        for (int i = 0; i < NUM_STREAMS; i++) {
            CUDA_CHECK(cudaStreamSynchronize(net->streams[i]));
        }

        // Print epoch summary - Unchanged
        printf("\nEpoch %d - Avg Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, epoch_loss / numImages,
               (double)correct_count * 100.0 / numImages,
               get_time(epoch_start_cpu));

        free(indices); // Free shuffled indices for the epoch

    } // End epoch loop


    // Free host buffers allocated once
    freeStandardMatrix(batch_outputs_host); // Use the correct free function
    freeHostMatrixPtrs(batch_images_ptrs, batch_size); // These only hold pointers
    freeHostMatrixPtrs(batch_labels_ptrs, batch_size); // These only hold pointers


    float gpu_time_sec = stopGpuTimer(&gpu_timer);
    printf("\nTotal GPU processing time (measured by CUDA events): %.3fs\n", gpu_time_sec);
    printf("Total training wall time (CPU measured): %.3fs\n", get_time(total_start_cpu));

}


// Evaluate accuracy on test data (Calls V4 forwardBatch)
void evaluate(NeuralNetwork *net, double **images, double **labels, int numImages) {
    printf("\nEvaluating model on test data (V4)...\n");
    clock_t start_cpu = clock();
    GpuTimer gpu_timer;
    startGpuTimer(&gpu_timer);

    int batch_size = MAX_BATCH_SIZE;
    long long correct_count = 0;

    // Allocate buffer for batch outputs on host ONCE
    double **batch_outputs_host = allocateStandardMatrix(batch_size, OUTPUT_SIZE);

    // Allocate host pointers for image batches ONCE
    double **batch_images_ptrs = allocateHostMatrixPtrs(batch_size); // Only needs image ptrs

    for (int batch_start = 0; batch_start < numImages; batch_start += batch_size) {
        int current_batch_size = fmin(batch_size, numImages - batch_start);
        if (current_batch_size <= 0) continue;

        // Prepare pointers to the current batch data
        for(int i=0; i < current_batch_size; ++i) {
             batch_images_ptrs[i] = images[batch_start + i];
        }

        // Forward Pass (V4) - Use stream 0 for evaluation simplicity
        // Populates batch_outputs_host with softmax probabilities
        forwardBatch(net, batch_images_ptrs, batch_outputs_host, current_batch_size, 0);
        // forwardBatch synchronizes stream 0 internally before returning results

        // Compare predictions with actual labels (on Host) - Unchanged from V3 logic
        for (int i = 0; i < current_batch_size; i++) {
            int pred_label_idx = 0;
            int actual_label_idx = -1;
            // Find actual label from original labels array
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (labels[batch_start + i][j] > 0.5) {
                     actual_label_idx = j;
                     break;
                 }
            }
            // Find predicted label from host results
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (batch_outputs_host[i][j] > batch_outputs_host[i][pred_label_idx]) {
                    pred_label_idx = j;
                }
            }

             if (actual_label_idx == -1) {
                 fprintf(stderr, "Warning: Actual label not found during evaluation for index %d\n", batch_start + i);
             } else if (pred_label_idx == actual_label_idx) {
                correct_count++;
            }
        }
    } // End batch loop

    // No need to sync streams explicitly here as forwardBatch syncs internally for host softmax.

    // Free host buffers allocated once
    freeStandardMatrix(batch_outputs_host);
    freeHostMatrixPtrs(batch_images_ptrs, batch_size);

    float gpu_time_sec = stopGpuTimer(&gpu_timer); // Stop GPU timer

    printf("\nTest Accuracy: %.2f%% (%lld / %d)\n",
           (double)correct_count * 100.0 / numImages, correct_count, numImages);
    printf("GPU evaluation time (measured by CUDA events): %.3fs\n", gpu_time_sec);
    printf("Total evaluation wall time (CPU measured): %.3fs\n", get_time(start_cpu));
}

// --- Load MNIST dataset functions --- (Unchanged from V3 - standard matrix allocation)
// Assuming MNIST files are in ../data/ relative to executable
const char* TRAIN_IMAGES_PATH = "../data/train-images.idx3-ubyte";
const char* TRAIN_LABELS_PATH = "../data/train-labels.idx1-ubyte";
const char* TEST_IMAGES_PATH = "../data/t10k-images.idx3-ubyte";
const char* TEST_LABELS_PATH = "../data/t10k-labels.idx1-ubyte";

double **loadMNISTImages(const char *filename, int numImages, int *imgSize) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening %s\n", filename);
        perror("fopen"); exit(1);
    }

    unsigned char header[16];
    if(fread(header, 1, 16, file) != 16) {
        fprintf(stderr, "Error reading header from %s\n", filename); fclose(file); exit(1);
    }
    // Bytes 4-7 contain number of images (big endian) - optional check
    // Bytes 8-11 contain rows, 12-15 contain columns
    int rows = (header[8] << 24) | (header[9] << 16) | (header[10] << 8) | header[11];
    int cols = (header[12] << 24) | (header[13] << 16) | (header[14] << 8) | header[15];
    *imgSize = rows * cols;
    if(*imgSize != INPUT_SIZE) {
         fprintf(stderr, "Error: Image size mismatch in %s (%d*%d != %d)\n", filename, rows, cols, INPUT_SIZE);
         fclose(file); exit(1);
    }

    // Allocate standard C 2D array
    double **images = allocateStandardMatrix(numImages, INPUT_SIZE);

    // Read image data
    unsigned char *buffer = (unsigned char *)malloc(INPUT_SIZE * sizeof(unsigned char));
     if (!buffer) {
        perror("Failed to allocate image buffer");
        fclose(file); exit(1);
     }

    printf("Loading %d images from %s...\n", numImages, filename);
    for (int i = 0; i < numImages; i++) {
         if (fread(buffer, sizeof(unsigned char), INPUT_SIZE, file) != INPUT_SIZE) {
            fprintf(stderr, "Error reading image data for index %d from %s\n", i, filename);
            fclose(file); free(buffer); freeStandardMatrix(images); exit(1);
         }
        // Normalize and store
        for (int j = 0; j < INPUT_SIZE; j++) {
            images[i][j] = buffer[j] / 255.0;
        }
         if ((i + 1) % 10000 == 0) {
             printf("  Loaded %d images...\n", i+1); fflush(stdout);
         }
    }
     printf("Finished loading images.\n");


    free(buffer);
    fclose(file);
    return images;
}

double **loadMNISTLabels(const char *filename, int numLabels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening %s\n", filename);
         perror("fopen"); exit(1);
    }
     // Bytes 0-3 magic number, bytes 4-7 number of items
    fseek(file, 8, SEEK_SET); // Skip MNIST header

    double **labels = allocateStandardMatrix(numLabels, OUTPUT_SIZE);
    unsigned char label_byte;

    printf("Loading %d labels from %s...\n", numLabels, filename);
    for (int i = 0; i < numLabels; i++) {
         if (fread(&label_byte, sizeof(unsigned char), 1, file) != 1) {
             fprintf(stderr, "Error reading label data for index %d from %s\n", i, filename);
             fclose(file); freeStandardMatrix(labels); exit(1);
         }
          if(label_byte >= OUTPUT_SIZE) {
             fprintf(stderr, "Error: Invalid label %d encountered at index %d in %s\n", label_byte, i, filename);
             fclose(file); freeStandardMatrix(labels); exit(1);
         }

        // One-hot encode
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label_byte) ? 1.0 : 0.0;
        }
         if ((i + 1) % 10000 == 0) {
             printf("  Loaded %d labels...\n", i+1); fflush(stdout);
         }
    }
    printf("Finished loading labels.\n");

    fclose(file);
    return labels;
}

// Free network memory - V4 with mixed precision buffers
void freeNetwork(NeuralNetwork *net) {
    if (!net) return;

    printf("Freeing network resources...\n");

    // Free device memory
    // Doubles
    if(net->d_W1) CUDA_CHECK(cudaFree(net->d_W1));
    if(net->d_W2) CUDA_CHECK(cudaFree(net->d_W2));
    if(net->d_b1) CUDA_CHECK(cudaFree(net->d_b1));
    if(net->d_b2) CUDA_CHECK(cudaFree(net->d_b2));
    if(net->d_inputs) CUDA_CHECK(cudaFree(net->d_inputs));
    if(net->d_outputs) CUDA_CHECK(cudaFree(net->d_outputs));
    if(net->d_pre_softmax_outputs) CUDA_CHECK(cudaFree(net->d_pre_softmax_outputs));
    if(net->d_targets) CUDA_CHECK(cudaFree(net->d_targets));

    // Halfs
    if(net->d_W1_half) CUDA_CHECK(cudaFree(net->d_W1_half));
    if(net->d_W2_half) CUDA_CHECK(cudaFree(net->d_W2_half));
    if(net->d_inputs_half) CUDA_CHECK(cudaFree(net->d_inputs_half));
    if(net->d_hiddens_half) CUDA_CHECK(cudaFree(net->d_hiddens_half));

    // Floats
    if(net->d_gemm1_output_float) CUDA_CHECK(cudaFree(net->d_gemm1_output_float));
    if(net->d_hiddens_float) CUDA_CHECK(cudaFree(net->d_hiddens_float));
    if(net->d_gemm2_output_float) CUDA_CHECK(cudaFree(net->d_gemm2_output_float));
    if(net->d_pre_softmax_float) CUDA_CHECK(cudaFree(net->d_pre_softmax_float));
    if(net->d_d_outputs_float) CUDA_CHECK(cudaFree(net->d_d_outputs_float));
    if(net->d_d_hiddens_float) CUDA_CHECK(cudaFree(net->d_d_hiddens_float));


    // Destroy cuBLAS handle
    if(net->cublas_handle) CUBLAS_CHECK(cublasDestroy(net->cublas_handle));

    // Destroy CUDA streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        if(net->streams[i]) CUDA_CHECK(cudaStreamDestroy(net->streams[i]));
    }

    // Free the struct itself
    free(net);
     printf("Network resources freed.\n");
}

// Main function (adapted from V3, using V4 components)
int main() {
    printf("--- MNIST Neural Network - V4: Tensor Core Accelerated GPU Implementation ---\n");
    printf("Network: Input=%d, Hidden=%d, Output=%d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Config: BatchSize=%d, LearningRate=%.2f, Epochs=%d, Streams=%d\n",
            MAX_BATCH_SIZE, (double)LEARNING_RATE, EPOCHS, NUM_STREAMS);


    // --- Device Query and Setup ---
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found.\n"); return EXIT_FAILURE;
    }
    cudaDeviceProp props;
    CUDA_CHECK(cudaSetDevice(0)); // Use device 0
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("\nUsing GPU: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    // Check Tensor Core support (Volta+, Compute Capability >= 7.0)
    if (props.major < 7) {
         printf("WARNING: This GPU (Compute Capability %d.%d) does NOT have Tensor Cores. Performance will be lower.\n",
                 props.major, props.minor);
    } else {
        printf("GPU supports Tensor Cores.\n");
    }
    // Consider setting cache config if beneficial, e.g., L1 preference
    // CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    CUDA_CHECK(cudaProfilerStart()); // Start profiler if needed


    // --- Load MNIST dataset ---
    printf("\nLoading MNIST dataset...\n");
    int trainImgSize, testImgSize;
    double** train_images = loadMNISTImages(TRAIN_IMAGES_PATH, 60000, &trainImgSize);
    double** train_labels = loadMNISTLabels(TRAIN_LABELS_PATH, 60000);
    double** test_images = loadMNISTImages(TEST_IMAGES_PATH, 10000, &testImgSize);
    double** test_labels = loadMNISTLabels(TEST_LABELS_PATH, 10000);
    if(trainImgSize != INPUT_SIZE || testImgSize != INPUT_SIZE){
        fprintf(stderr, "Input size mismatch during data loading.\n"); exit(1);
    }
    printf("Dataset loaded successfully.\n");


    // --- Create Neural Network (V4 version) ---
    NeuralNetwork *net = createNetwork();


    // --- Train Network ---
    train(net, train_images, train_labels, 60000);


    // --- Evaluate on Test Set ---
    evaluate(net, test_images, test_labels, 10000);


    // --- Free Resources ---
    printf("\nCleaning up host and device resources...\n");
    freeNetwork(net); // Frees device memory, cuBLAS handle, streams

    // Free host MNIST data
    freeStandardMatrix(train_images);
    freeStandardMatrix(train_labels);
    freeStandardMatrix(test_images);
    freeStandardMatrix(test_labels);
     printf("Host data freed.\n");

    // --- Profiler and Device Reset ---
    CUDA_CHECK(cudaProfilerStop()); // Stop profiler
    // Optional: Reset device state
    // CUDA_CHECK(cudaDeviceReset());

    printf("\nExecution Finished Successfully.\n");
    return 0;
}