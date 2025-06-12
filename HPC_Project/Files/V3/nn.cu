#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h> // Keep for potential profiling needs

// Network parameters (from V3/V2)
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 1 // Same as V1/V2
#define EPOCHS 3
#define NUM_CLASSES 10

// Batch processing (from V3)
#define MAX_BATCH_SIZE 256 // Keep batching
#define NUM_STREAMS 4      // Keep streams

// Thread block configuration (from V3)
#define BLOCK_SIZE_1D 256
#define BLOCK_SIZE_X 16 // Keep 2D config where used
#define BLOCK_SIZE_Y 16

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

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// GPU timer functions using CUDA events (Keep from V3)
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

// Allocate/Free host matrix memory (Simplified version from V2 for flat host arrays needed)
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
        // Assuming the underlying data for rows was allocated elsewhere or managed differently
        free(mat);
     }
}

// Helper for allocating a standard C 2D array if needed (like in original load)
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

// Free allocated standard matrix memory
void freeStandardMatrix(double **mat) {
    if (mat) {
        if (mat[0]) {
            free(mat[0]); // Free the contiguous block
        }
        free(mat); // Free the row pointers
    }
}

// Neural network structure - optimized for GPU (Keep V3 Structure)
typedef struct {
    // Host weight/bias arrays (Only used for initialization/final sync if needed)
    // Keep copies to easily revert init or compare later if needed.
    // double** W1;
    // double** W2;
    // double* b1;
    // double* b2;

    // Device matrices (flattened for GPU - Primary storage)
    double *d_W1;
    double *d_W2;
    double *d_b1;
    double *d_b2;

    // Batch processing device arrays
    double *d_inputs;   // Stores current batch input
    double *d_hiddens;  // Stores hidden layer output (post-ReLU)
    double *d_outputs;  // Stores final layer output (post-Softmax)
    double *d_pre_softmax_outputs; // Temporary store for pre-softmax values
    double *d_targets;  // Stores current batch target labels

    // Batch processing device arrays for gradients
    double *d_d_hiddens; // Gradient wrt hidden layer output (post-ReLU)
    double *d_d_outputs; // Gradient wrt output layer output (post-Softmax)

    // CUDA streams for overlapping operations
    cudaStream_t streams[NUM_STREAMS];

} NeuralNetwork;

// --- Kernels (Keep V3 optimized kernels except Softmax) ---

// Forward layer 1 (Matrix-vector + bias)
__global__ void forwardLayer1Kernel(double *W1, double *b1, double *inputs, double *hiddens,
                                  int input_size, int hidden_size, int batch_size) {
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x; // Index for hidden unit
    int batch_idx = blockIdx.y;                       // Index for sample in batch

    if (h_idx < hidden_size && batch_idx < batch_size) {
        int input_offset = batch_idx * input_size;   // Offset for start of input vector in batch
        int hidden_offset = batch_idx * hidden_size; // Offset for start of hidden vector in batch

        double sum = b1[h_idx];
        const double *input_vec = inputs + input_offset; // Pointer to current input sample
        const double *W1_row = W1 + h_idx * input_size;  // Pointer to current weight row

        // Compute dot product: W1[h_idx] . inputs[batch_idx]
        for (int i = 0; i < input_size; i++) {
            sum += W1_row[i] * input_vec[i];
        }

        // Store result (pre-activation) - V3 used d_hiddens directly, assumed ReLU after
        hiddens[hidden_offset + h_idx] = sum;
    }
}


// Optimized ReLU activation for batch processing
__global__ void batchReluKernel(double *x, int size_per_sample, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Index within a sample's vector
    int batch_idx = blockIdx.y;                     // Index for sample in batch

    if (idx < size_per_sample && batch_idx < batch_size) {
        int offset = batch_idx * size_per_sample + idx;
        x[offset] = (x[offset] > 0.0) ? x[offset] : 0.0;
    }
}

// Forward layer 2 (Matrix-vector + bias)
__global__ void forwardLayer2Kernel(double *W2, double *b2, double *hiddens, double *outputs, // Output here is PRE-SOFTMAX
                                  int hidden_size, int output_size, int batch_size) {
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x; // Index for output unit
    int batch_idx = blockIdx.y;                       // Index for sample in batch

    if (o_idx < output_size && batch_idx < batch_size) {
        int hidden_offset = batch_idx * hidden_size;  // Offset for start of hidden vector in batch
        int output_offset = batch_idx * output_size; // Offset for start of output vector in batch

        double sum = b2[o_idx];
        const double *hidden_vec = hiddens + hidden_offset; // Pointer to current hidden sample (post-ReLU)
        const double *W2_row = W2 + o_idx * hidden_size;  // Pointer to current weight row

        // Compute dot product: W2[o_idx] . hiddens[batch_idx]
        for (int i = 0; i < hidden_size; i++) {
            sum += W2_row[i] * hidden_vec[i];
        }

        // Store result (pre-softmax)
        outputs[output_offset + o_idx] = sum;
    }
}

// *** REMOVED V3 softmaxKernel ***

// Compute output layer error (d_output = output - target) for backpropagation
__global__ void computeOutputErrorKernel(double *d_outputs, double *d_targets, double *d_d_outputs, // Gradients stored here
                                      int output_size, int batch_size) {
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x; // Index within output vector
    int batch_idx = blockIdx.y;                      // Index for sample in batch

    if (o_idx < output_size && batch_idx < batch_size) {
        int offset = batch_idx * output_size; // Offset for the start of the sample in arrays
        // Assumes d_outputs contains POST-softmax values
        d_d_outputs[offset + o_idx] = d_outputs[offset + o_idx] - d_targets[offset + o_idx];
    }
}


// Compute hidden layer error gradient, optimized from V3 (W2^T * d_output) * relu_deriv(hidden)
__global__ void computeHiddenErrorKernel(double *d_W2, double *d_d_outputs, double *d_hiddens, double *d_d_hiddens, // Gradients stored here
                                       int hidden_size, int output_size, int batch_size) {
    // __shared__ double shared_d_outputs[BLOCK_SIZE_1D]; // Optimization from V3, let's keep it simple first? Maybe remove.
    // Simple non-shared version first, matching V2 logic more directly
    // A thread computes the gradient for one hidden unit for one batch sample.

    int h_idx = blockIdx.x * blockDim.x + threadIdx.x; // Index for hidden unit
    int batch_idx = blockIdx.y;                        // Index for sample in batch

    if (h_idx < hidden_size && batch_idx < batch_size) {
        int hidden_offset = batch_idx * hidden_size;
        int output_offset = batch_idx * output_size;

        double error_sum = 0.0;

        // Compute W2^T * d_output for column h_idx
        // W2 is output_size x hidden_size
        for (int o_idx = 0; o_idx < output_size; o_idx++) {
            // Accessing W2[o_idx][h_idx] which is d_W2[o_idx * hidden_size + h_idx]
            // Accessing d_d_outputs[batch_idx][o_idx] which is d_d_outputs[output_offset + o_idx]
            error_sum += d_W2[o_idx * hidden_size + h_idx] * d_d_outputs[output_offset + o_idx];
        }

        // Apply ReLU derivative: multiply by 1 if hidden activation > 0, else 0
        // d_hiddens contains the activations *after* ReLU
        double activation = d_hiddens[hidden_offset + h_idx];
        d_d_hiddens[hidden_offset + h_idx] = error_sum * (activation > 0.0 ? 1.0 : 0.0);
    }
}

// Update weights W2, optimized with 2D grid (Kept from V3)
// d_W2 -= learning_rate * averaged_gradient(d_d_outputs * d_hiddens^T)
__global__ void updateWeightsW2Kernel(double *d_W2, double *d_d_outputs, double *d_hiddens,
                                    double learning_rate, int hidden_size, int output_size,
                                    int batch_size) {
    // Each thread updates one weight W2[o_idx][h_idx]
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x; // Output index (row)
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y; // Hidden index (col)

    if (o_idx < output_size && h_idx < hidden_size) {
        double weight_update_sum = 0.0;

        // Sum gradient over the batch: d_output[b][o] * hidden[b][h]
        for (int b = 0; b < batch_size; b++) {
            int output_offset = b * output_size;
            int hidden_offset = b * hidden_size;
            weight_update_sum += d_d_outputs[output_offset + o_idx] * d_hiddens[hidden_offset + h_idx];
        }

        // Average gradient over the batch and apply update
        double averaged_gradient = weight_update_sum / batch_size;
        d_W2[o_idx * hidden_size + h_idx] -= learning_rate * averaged_gradient;
    }
}


// Update weights W1, optimized with 2D grid (Kept from V3)
// d_W1 -= learning_rate * averaged_gradient(d_d_hiddens * d_inputs^T)
__global__ void updateWeightsW1Kernel(double *d_W1, double *d_d_hiddens, double *d_inputs,
                                    double learning_rate, int input_size, int hidden_size,
                                    int batch_size) {
    // Each thread updates one weight W1[h_idx][i_idx]
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x; // Hidden index (row)
    int i_idx = blockIdx.y * blockDim.y + threadIdx.y; // Input index (col)

    if (h_idx < hidden_size && i_idx < input_size) {
        double weight_update_sum = 0.0;

        // Sum gradient over the batch: d_hidden[b][h] * input[b][i]
        for (int b = 0; b < batch_size; b++) {
            int hidden_offset = b * hidden_size;
            int input_offset = b * input_size;
            weight_update_sum += d_d_hiddens[hidden_offset + h_idx] * d_inputs[input_offset + i_idx];
        }

        // Average gradient over the batch and apply update
        double averaged_gradient = weight_update_sum / batch_size;
        d_W1[h_idx * input_size + i_idx] -= learning_rate * averaged_gradient;
    }
}

// Update biases for both layers (Kept from V3)
__global__ void updateBiasesKernel(double *d_b1, double *d_b2, double *d_d_hiddens, double *d_d_outputs,
                                 double learning_rate, int hidden_size, int output_size,
                                 int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Update hidden layer biases b1[idx]
    if (idx < hidden_size) {
        double bias1_update_sum = 0.0;
        for (int b = 0; b < batch_size; b++) {
            bias1_update_sum += d_d_hiddens[b * hidden_size + idx];
        }
        double averaged_gradient = bias1_update_sum / batch_size;
        d_b1[idx] -= learning_rate * averaged_gradient;
    }

    // Update output layer biases b2[idx]
    // Note: This assumes idx < output_size calculation runs in parallel for different range
    if (idx < output_size) {
        double bias2_update_sum = 0.0;
        for (int b = 0; b < batch_size; b++) {
            bias2_update_sum += d_d_outputs[b * output_size + idx];
        }
        double averaged_gradient = bias2_update_sum / batch_size;
        d_b2[idx] -= learning_rate * averaged_gradient;
    }
}


// *** Add back softmax_host function from V2 ***
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



// Initialize neural network - USING XAVIER INIT from Original V3
NeuralNetwork *createNetwork() {
    NeuralNetwork *net = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));

    // --- Allocate Device Memory FIRST ---
    size_t w1_size = (size_t)HIDDEN_SIZE * INPUT_SIZE;
    size_t w2_size = (size_t)OUTPUT_SIZE * HIDDEN_SIZE;
    size_t b1_size = HIDDEN_SIZE;
    size_t b2_size = OUTPUT_SIZE;
    size_t batch_input_size = (size_t)MAX_BATCH_SIZE * INPUT_SIZE;
    size_t batch_hidden_size = (size_t)MAX_BATCH_SIZE * HIDDEN_SIZE;
    size_t batch_output_size = (size_t)MAX_BATCH_SIZE * OUTPUT_SIZE;

    CUDA_CHECK(cudaMalloc(&net->d_W1, w1_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_W2, w2_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_b1, b1_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_b2, b2_size * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&net->d_inputs, batch_input_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_hiddens, batch_hidden_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_outputs, batch_output_size * sizeof(double))); // Will hold final post-softmax output
    CUDA_CHECK(cudaMalloc(&net->d_pre_softmax_outputs, batch_output_size * sizeof(double))); // TEMP store for pre-softmax
    CUDA_CHECK(cudaMalloc(&net->d_targets, batch_output_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_d_hiddens, batch_hidden_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_d_outputs, batch_output_size * sizeof(double)));


    // --- Initialize on HOST (Using Xavier/Glorot initialization - From Original V3) ---
    double *h_W1 = allocateHostVector(w1_size);
    double *h_W2 = allocateHostVector(w2_size);
    double *h_b1 = allocateHostVector(b1_size); // Will be calloc'd effectively later
    double *h_b2 = allocateHostVector(b2_size); // Will be calloc'd effectively later

    srand(time(NULL));
    // Xavier bounds
    double w1_bound = sqrt(6.0 / (INPUT_SIZE + HIDDEN_SIZE));
    double w2_bound = sqrt(6.0 / (HIDDEN_SIZE + OUTPUT_SIZE));

    // Initialize W1 with Xavier uniform distribution
    for (size_t i = 0; i < w1_size; i++) {
        h_W1[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * w1_bound;
    }
    // Initialize W2 with Xavier uniform distribution
    for (size_t i = 0; i < w2_size; i++) {
         h_W2[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * w2_bound;
    }
    // Initialize biases to zero
    memset(h_b1, 0, b1_size * sizeof(double)); // Use memset for zeroing
    memset(h_b2, 0, b2_size * sizeof(double));


    // --- Copy from Host to Device ---
    CUDA_CHECK(cudaMemcpy(net->d_W1, h_W1, w1_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_W2, h_W2, w2_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b1, h_b1, b1_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b2, h_b2, b2_size * sizeof(double), cudaMemcpyHostToDevice));

    // --- Clean up temporary host arrays ---
    free(h_W1);
    free(h_W2);
    free(h_b1);
    free(h_b2);

    // Create CUDA streams (from V3)
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&net->streams[i], cudaStreamNonBlocking));
    }

    return net;
}

// Forward pass for a batch of inputs - Modified for host softmax
// Takes HOST input (batch_images), produces HOST output (batch_outputs_final)
void forwardBatch(NeuralNetwork *net, double **batch_images, double **batch_outputs_final, int batch_size, int stream_idx) {
    cudaStream_t stream = net->streams[stream_idx];

    // 1. Prepare flat input batch on HOST
    size_t input_batch_bytes = (size_t)batch_size * INPUT_SIZE * sizeof(double);
    double *h_input_batch = allocateHostVector(batch_size * INPUT_SIZE);
    for (int i = 0; i < batch_size; i++) {
        // batch_images[i] points to the row in the original dataset memory
        memcpy(h_input_batch + (size_t)i * INPUT_SIZE, batch_images[i], INPUT_SIZE * sizeof(double));
    }

    // 2. Copy flat input batch HOST -> DEVICE (Async)
    CUDA_CHECK(cudaMemcpyAsync(net->d_inputs, h_input_batch, input_batch_bytes, cudaMemcpyHostToDevice, stream));

    // 3. Layer 1 Kernel: W1 * inputs + b1 -> d_hiddens (Pre-ReLU)
    dim3 grid1((HIDDEN_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 block1(BLOCK_SIZE_1D);
    forwardLayer1Kernel<<<grid1, block1, 0, stream>>>(
        net->d_W1, net->d_b1, net->d_inputs, net->d_hiddens, // Output is pre-ReLU hidden
        INPUT_SIZE, HIDDEN_SIZE, batch_size
    );

    // 4. ReLU Kernel: Applies ReLU inplace to d_hiddens
    dim3 gridRelu((HIDDEN_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockRelu(BLOCK_SIZE_1D);
    batchReluKernel<<<gridRelu, blockRelu, 0, stream>>>(net->d_hiddens, HIDDEN_SIZE, batch_size);

    // 5. Layer 2 Kernel: W2 * hiddens + b2 -> d_pre_softmax_outputs
    dim3 grid2((OUTPUT_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 block2(BLOCK_SIZE_1D);
    forwardLayer2Kernel<<<grid2, block2, 0, stream>>>(
        net->d_W2, net->d_b2, net->d_hiddens, net->d_pre_softmax_outputs, // Output to temp pre-softmax buffer
        HIDDEN_SIZE, OUTPUT_SIZE, batch_size
    );

    // --- Softmax on Host ---
    // 6. Copy pre-softmax output DEVICE -> HOST (Async)
    size_t output_batch_bytes = (size_t)batch_size * OUTPUT_SIZE * sizeof(double);
    double *h_output_batch = allocateHostVector(batch_size * OUTPUT_SIZE); // Host buffer for calculation
    CUDA_CHECK(cudaMemcpyAsync(h_output_batch, net->d_pre_softmax_outputs, output_batch_bytes, cudaMemcpyDeviceToHost, stream));

    // 7. Synchronize ONLY this stream to ensure h_output_batch is ready
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 8. Compute softmax on HOST for each sample in the batch
    for (int i = 0; i < batch_size; i++) {
        softmax_host(h_output_batch + (size_t)i * OUTPUT_SIZE, OUTPUT_SIZE);
    }

    // 9. Copy final softmax result HOST -> DEVICE (into net->d_outputs for backprop) (Async)
    CUDA_CHECK(cudaMemcpyAsync(net->d_outputs, h_output_batch, output_batch_bytes, cudaMemcpyHostToDevice, stream));

    // 10. Populate the output parameter if provided (for evaluation / loss calc)
    if (batch_outputs_final != NULL) {
        // We already have the final result in h_output_batch
        for (int i = 0; i < batch_size; i++) {
             memcpy(batch_outputs_final[i], h_output_batch + (size_t)i * OUTPUT_SIZE, OUTPUT_SIZE * sizeof(double));
        }
    }

    // 11. Clean up host temporary buffers
    free(h_input_batch);
    free(h_output_batch);

    // Note: Kernels for backprop are launched later using the same stream.
    // cudaStreamSynchronize(stream) will be called at the end of the batch loop in train().
}

// Backpropagation for a batch - Kept largely same as V3
void backwardBatch(NeuralNetwork *net, double **batch_labels, int batch_size, int stream_idx) {
    cudaStream_t stream = net->streams[stream_idx];

    // 1. Prepare flat target batch on HOST
    size_t target_batch_bytes = (size_t)batch_size * OUTPUT_SIZE * sizeof(double);
    double *h_target_batch = allocateHostVector(batch_size * OUTPUT_SIZE);
    for (int i = 0; i < batch_size; i++) {
        memcpy(h_target_batch + (size_t)i * OUTPUT_SIZE, batch_labels[i], OUTPUT_SIZE * sizeof(double));
    }

    // 2. Copy flat target batch HOST -> DEVICE (Async)
    CUDA_CHECK(cudaMemcpyAsync(net->d_targets, h_target_batch, target_batch_bytes, cudaMemcpyHostToDevice, stream));

    // 3. Compute Output Error Kernel: d_outputs - d_targets -> d_d_outputs
    dim3 gridErr1((OUTPUT_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockErr1(BLOCK_SIZE_1D);
    computeOutputErrorKernel<<<gridErr1, blockErr1, 0, stream>>>(
        net->d_outputs, net->d_targets, net->d_d_outputs, // Use post-softmax d_outputs
        OUTPUT_SIZE, batch_size
    );

    // 4. Compute Hidden Error Kernel: (W2^T * d_d_outputs) .* relu_deriv(hiddens) -> d_d_hiddens
    dim3 gridErr2((HIDDEN_SIZE + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D, batch_size);
    dim3 blockErr2(BLOCK_SIZE_1D);
    computeHiddenErrorKernel<<<gridErr2, blockErr2, 0, stream>>>(
        net->d_W2, net->d_d_outputs, net->d_hiddens, net->d_d_hiddens,
        HIDDEN_SIZE, OUTPUT_SIZE, batch_size
    );

    // 5. Update Weights W2 Kernel
    dim3 blockW2(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridW2((OUTPUT_SIZE + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                (HIDDEN_SIZE + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    updateWeightsW2Kernel<<<gridW2, blockW2, 0, stream>>>(
        net->d_W2, net->d_d_outputs, net->d_hiddens,
        LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE, batch_size
    );

    // 6. Update Weights W1 Kernel
    // Note: d_inputs must still contain the input for the current batch
    dim3 blockW1(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridW1((HIDDEN_SIZE + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                (INPUT_SIZE + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    updateWeightsW1Kernel<<<gridW1, blockW1, 0, stream>>>(
        net->d_W1, net->d_d_hiddens, net->d_inputs, // Ensure d_inputs hasn't been overwritten
        LEARNING_RATE, INPUT_SIZE, HIDDEN_SIZE, batch_size
    );

    // 7. Update Biases Kernel
    dim3 blockBias(BLOCK_SIZE_1D);
    dim3 gridBias((fmax(HIDDEN_SIZE, OUTPUT_SIZE) + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);
    updateBiasesKernel<<<gridBias, blockBias, 0, stream>>>(
        net->d_b1, net->d_b2, net->d_d_hiddens, net->d_d_outputs,
        LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE, batch_size
    );

    // 8. Clean up host temporary buffer
    free(h_target_batch);

    // Synchronization will happen after the batch loop in train().
}


// Train function (modified from V3)
void train(NeuralNetwork *net, double **images, double **labels, int numImages) {
    printf("Starting training (V3 modified for accuracy)...\n");
    clock_t total_start_cpu = clock();
    GpuTimer gpu_timer;
    startGpuTimer(&gpu_timer); // Start GPU timer

    int batch_size = MAX_BATCH_SIZE;

    // Allocate buffer for batch outputs on host ONCE (for loss/accuracy calculation)
    double **batch_outputs_host = allocateHostMatrixPtrs(batch_size);
     // Allocate underlying storage for batch outputs ONCE
    double *batch_outputs_storage = allocateHostVector((size_t)batch_size * OUTPUT_SIZE);
    for(int i = 0; i < batch_size; ++i) {
        batch_outputs_host[i] = batch_outputs_storage + i * OUTPUT_SIZE;
    }


    // Allocate host pointers for image/label batches ONCE
    double **batch_images_ptrs = allocateHostMatrixPtrs(batch_size);
    double **batch_labels_ptrs = allocateHostMatrixPtrs(batch_size);


    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start_cpu = clock();
        double epoch_loss = 0.0;
        long long correct_count = 0; // Use long long for potentially large sums

        // Shuffle indices (Fisher-Yates)
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

            // Prepare pointers to the current batch data
            for(int i=0; i < current_batch_size; ++i) {
                int idx = indices[batch_start + i];
                batch_images_ptrs[i] = images[idx];
                batch_labels_ptrs[i] = labels[idx];
            }

            // Determine stream
            int stream_idx = (batch_start / batch_size) % NUM_STREAMS;

            // --- Forward Pass ---
            // Resulting softmax probabilities will be in batch_outputs_host
            forwardBatch(net, batch_images_ptrs, batch_outputs_host, current_batch_size, stream_idx);

             // At this point, forwardBatch has sync'd the stream to get results for softmax.
             // The H->D copy of final softmax results back to net->d_outputs has been *issued* on the stream.

            // --- Loss and Accuracy Calculation (on Host using batch_outputs_host) ---
            for (int i = 0; i < current_batch_size; i++) {
                // Cross-entropy loss for sample i
                double sample_loss = 0.0;
                 int actual_label_idx = -1;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                     if (batch_labels_ptrs[i][j] > 0.5) { // Find actual label
                        actual_label_idx = j;
                        double prob = batch_outputs_host[i][j];
                         if (prob < 1e-9) prob = 1e-9; // Avoid log(0)
                         sample_loss = -log(prob);
                         break; // Found the target label
                    }
                }
                epoch_loss += sample_loss;


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

            // --- Backward Pass ---
            // Uses net->d_outputs which contains the results of forward pass copied back to device.
             // Needs to wait for the H->D copy in forwardBatch to complete *before* launching backprop kernels on the *same stream*.
             // Because kernels are launched on the same stream after the memcpyAsync, they will wait automatically.
            backwardBatch(net, batch_labels_ptrs, current_batch_size, stream_idx);

             // Print progress (less frequently)
            if ((batch_start / batch_size + 1) % 50 == 0) { // Print every 50 batches
                printf("Epoch %d: %d / %d samples processed.\r",
                        epoch + 1, batch_start + current_batch_size, numImages);
                 fflush(stdout);
             }

        } // End batch loop

        // Synchronize all streams at the end of the epoch
        for (int i = 0; i < NUM_STREAMS; i++) {
            CUDA_CHECK(cudaStreamSynchronize(net->streams[i]));
        }

        // Print epoch summary
        printf("Epoch %d - Avg Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, epoch_loss / numImages,
               (double)correct_count * 100.0 / numImages,
               get_time(epoch_start_cpu));

        free(indices); // Free shuffled indices for the epoch

    } // End epoch loop


    // Free host buffers allocated once
    free(batch_outputs_storage);
    freeHostMatrixPtrs(batch_outputs_host, batch_size);
    freeHostMatrixPtrs(batch_images_ptrs, batch_size);
    freeHostMatrixPtrs(batch_labels_ptrs, batch_size);


    float gpu_time_sec = stopGpuTimer(&gpu_timer);
    printf("Total GPU processing time: %.3fs\n", gpu_time_sec);
    printf("Total training time (CPU): %.3fs\n", get_time(total_start_cpu));

    // Optional: Sync weights back to host if needed for some reason (not typically required)
    // syncWeightsToHost(net);
}


// Evaluate accuracy on test data (modified V3 with host softmax)
void evaluate(NeuralNetwork *net, double **images, double **labels, int numImages) {
    printf("\nEvaluating model...\n");
    clock_t start_cpu = clock();
    GpuTimer gpu_timer;
    startGpuTimer(&gpu_timer); // Start GPU timer

    int batch_size = MAX_BATCH_SIZE;
    long long correct_count = 0;

    // Allocate buffer for batch outputs on host ONCE
    double **batch_outputs_host = allocateHostMatrixPtrs(batch_size);
    double *batch_outputs_storage = allocateHostVector((size_t)batch_size * OUTPUT_SIZE);
     for(int i = 0; i < batch_size; ++i) {
        batch_outputs_host[i] = batch_outputs_storage + i * OUTPUT_SIZE;
     }

    // Allocate host pointers for image batches ONCE
    double **batch_images_ptrs = allocateHostMatrixPtrs(batch_size);


    for (int batch_start = 0; batch_start < numImages; batch_start += batch_size) {
        int current_batch_size = fmin(batch_size, numImages - batch_start);
        if (current_batch_size <= 0) continue;

        // Prepare pointers to the current batch data
        for(int i=0; i < current_batch_size; ++i) {
             batch_images_ptrs[i] = images[batch_start + i]; // Point to rows in original test data
        }

        // Forward Pass (using stream 0 for evaluation)
        // Populates batch_outputs_host with softmax probabilities
        forwardBatch(net, batch_images_ptrs, batch_outputs_host, current_batch_size, 0);

        // Make sure the results are ready after the forward pass stream execution
        CUDA_CHECK(cudaStreamSynchronize(net->streams[0]));

        // Compare predictions with actual labels (on Host)
        for (int i = 0; i < current_batch_size; i++) {
            int pred_label_idx = 0;
             int actual_label_idx = -1;
            // Find actual label
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (labels[batch_start + i][j] > 0.5) {
                     actual_label_idx = j;
                     break;
                 }
            }
            // Find predicted label
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (batch_outputs_host[i][j] > batch_outputs_host[i][pred_label_idx]) {
                    pred_label_idx = j;
                }
            }

            if (pred_label_idx == actual_label_idx) {
                correct_count++;
            }
        }
    } // End batch loop

    // Free host buffers allocated once
    free(batch_outputs_storage);
    freeHostMatrixPtrs(batch_outputs_host, batch_size);
    freeHostMatrixPtrs(batch_images_ptrs, batch_size);

    float gpu_time_sec = stopGpuTimer(&gpu_timer); // Stop GPU timer

    printf("Test Accuracy: %.2f%% (%lld / %d)\n",
           (double)correct_count * 100.0 / numImages, correct_count, numImages);
    printf("GPU evaluation time: %.3fs\n", gpu_time_sec);
    printf("Total evaluation time (CPU): %.3fs\n", get_time(start_cpu));
}

// --- Load MNIST dataset functions --- (Using standard allocation now)
double **loadMNISTImages(const char *filename, int numImages) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET); // Skip MNIST header

    // Allocate standard C 2D array
    double **images = allocateStandardMatrix(numImages, INPUT_SIZE);

    // Read image data directly into allocated block for better locality
    unsigned char *buffer = (unsigned char *)malloc(INPUT_SIZE * sizeof(unsigned char));
     if (!buffer) {
        perror("Failed to allocate image buffer");
        fclose(file); exit(1);
     }


    printf("Loading %d images from %s...\n", numImages, filename);
    for (int i = 0; i < numImages; i++) {
         if (fread(buffer, sizeof(unsigned char), INPUT_SIZE, file) != INPUT_SIZE) {
            fprintf(stderr, "Error reading image data for index %d\n", i);
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
        exit(1);
    }
    fseek(file, 8, SEEK_SET); // Skip MNIST header

    double **labels = allocateStandardMatrix(numLabels, OUTPUT_SIZE); // Allocate storage
    unsigned char label_byte;

    printf("Loading %d labels from %s...\n", numLabels, filename);
    for (int i = 0; i < numLabels; i++) {
         if (fread(&label_byte, sizeof(unsigned char), 1, file) != 1) {
             fprintf(stderr, "Error reading label data for index %d\n", i);
             fclose(file); freeStandardMatrix(labels); exit(1);
         }

        // One-hot encode directly into allocated memory
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

// Free network memory
void freeNetwork(NeuralNetwork *net) {
    if (!net) return;

    // Free device memory
    CUDA_CHECK(cudaFree(net->d_W1));
    CUDA_CHECK(cudaFree(net->d_W2));
    CUDA_CHECK(cudaFree(net->d_b1));
    CUDA_CHECK(cudaFree(net->d_b2));
    CUDA_CHECK(cudaFree(net->d_inputs));
    CUDA_CHECK(cudaFree(net->d_hiddens));
    CUDA_CHECK(cudaFree(net->d_outputs));
    CUDA_CHECK(cudaFree(net->d_pre_softmax_outputs));
    CUDA_CHECK(cudaFree(net->d_targets));
    CUDA_CHECK(cudaFree(net->d_d_hiddens));
    CUDA_CHECK(cudaFree(net->d_d_outputs));

    // Destroy CUDA streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamDestroy(net->streams[i]));
    }

    // Free the struct itself
    free(net);
}

// Main function (adapted from V3)
int main() {
    printf("MNIST Neural Network - GPU Implementation (V3 Modified for Accuracy)\n\n");

    // Optional: Device Query / Config
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found.\n"); return EXIT_FAILURE;
    }
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0)); // Using device 0
    printf("Using GPU: %s (Compute Capability %d.%d)\n", props.name, props.major, props.minor);
    // CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)); // Optional: Set cache config

    // Load MNIST dataset (Update paths if necessary)
    printf("\nLoading MNIST dataset...\n");
    // Make sure these paths are correct relative to where you execute the program
    double** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);
    printf("Dataset loaded successfully.\n\n");


    // Create neural network (uses device memory, V2 init)
    NeuralNetwork *net = createNetwork();

    // Train network
    train(net, train_images, train_labels, 60000);

    // Evaluate on test set
    evaluate(net, test_images, test_labels, 10000);

    // Free resources
    printf("\nCleaning up...\n");
    freeNetwork(net);
    freeStandardMatrix(train_images);
    freeStandardMatrix(train_labels);
    freeStandardMatrix(test_images);
    freeStandardMatrix(test_labels);

    // Optional: Reset device
    // CUDA_CHECK(cudaDeviceReset());

    printf("Done.\n");
    return 0;
}