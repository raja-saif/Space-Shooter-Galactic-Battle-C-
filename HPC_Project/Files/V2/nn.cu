#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

// CUDA error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// ReLU activation kernel
__global__ void reluKernel(double* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = (x[idx] > 0) ? x[idx] : 0;
    }
}

// Softmax activation kernel
__global__ void softmaxKernel(double* x, int size) {
    // Find maximum value for numerical stability
    double max_val = -INFINITY;
    for (int i = 0; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// CPU-based activation functions (used for initialization and validation)
void relu(double* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax(double* x, int size) {
    double max_val = -INFINITY;
    for (int i = 0; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Neural network structure
typedef struct {
    // Host matrices
    double** W1;
    double** W2;
    double* b1;
    double* b2;
    
    // Device matrices (flattened for GPU)
    double* d_W1;
    double* d_W2;
    double* d_b1;
    double* d_b2;
    double* d_input;
    double* d_hidden;
    double* d_output;
    double* d_target;
    double* d_d_hidden;
    double* d_d_output;
} NeuralNetwork;

// Forward pass kernel
__global__ void forwardKernel1(double* input, double* W1, double* b1, double* hidden, 
                             int input_size, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < hidden_size) {
        hidden[i] = b1[i];
        for (int j = 0; j < input_size; j++) {
            hidden[i] += W1[i * input_size + j] * input[j];
        }
    }
}

__global__ void forwardKernel2(double* hidden, double* W2, double* b2, double* output, 
                             int hidden_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < output_size) {
        output[i] = b2[i];
        for (int j = 0; j < hidden_size; j++) {
            output[i] += W2[i * hidden_size + j] * hidden[j];
        }
    }
}

// Backward pass kernels
__global__ void backwardKernel1(double* output, double* target, double* d_output, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < output_size) {
        d_output[i] = output[i] - target[i];
    }
}

__global__ void backwardKernel2(double* W2, double* d_output, double* hidden, double* d_hidden, 
                              int hidden_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < hidden_size) {
        d_hidden[i] = 0;
        for (int j = 0; j < output_size; j++) {
            d_hidden[i] += W2[j * hidden_size + i] * d_output[j];
        }
        d_hidden[i] *= (hidden[i] > 0); // ReLU derivative
    }
}

__global__ void backwardKernel3(double* W2, double* d_output, double* hidden, double* d_hidden, 
                              double learning_rate, int hidden_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < output_size && j < hidden_size) {
        W2[i * hidden_size + j] -= learning_rate * d_output[i] * hidden[j];
    }
}

__global__ void backwardKernel4(double* W1, double* d_hidden, double* input, double learning_rate,
                              int input_size, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < hidden_size && j < input_size) {
        W1[i * input_size + j] -= learning_rate * d_hidden[i] * input[j];
    }
}

__global__ void backwardKernel5(double* b2, double* d_output, double learning_rate, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < output_size) {
        b2[i] -= learning_rate * d_output[i];
    }
}

__global__ void backwardKernel6(double* b1, double* d_hidden, double learning_rate, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < hidden_size) {
        b1[i] -= learning_rate * d_hidden[i];
    }
}

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Allocate host memory
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    
    // Initialize weights with small random values
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_input, INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_hidden, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_output, OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_target, OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_d_hidden, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_d_output, OUTPUT_SIZE * sizeof(double)));
    
    // Copy initial weights to device
    double* W1_flat = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* W2_flat = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    
    // Flatten 2D arrays for GPU
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            W1_flat[i * INPUT_SIZE + j] = net->W1[i][j];
        }
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            W2_flat[i * HIDDEN_SIZE + j] = net->W2[i][j];
        }
    }
    
    CUDA_CHECK(cudaMemcpy(net->d_W1, W1_flat, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_W2, W2_flat, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    free(W1_flat);
    free(W2_flat);
    
    return net;
}

// Forward pass using GPU
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch kernels
    int blockSize = 128;
    int gridSize1 = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    int gridSize2 = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    
    forwardKernel1<<<gridSize1, blockSize>>>(net->d_input, net->d_W1, net->d_b1, net->d_hidden, 
                                         INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Apply ReLU activation
    reluKernel<<<gridSize1, blockSize>>>(net->d_hidden, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    forwardKernel2<<<gridSize2, blockSize>>>(net->d_hidden, net->d_W2, net->d_b2, net->d_output, 
                                         HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Apply softmax activation
    softmaxKernel<<<1, 1>>>(net->d_output, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(hidden, net->d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
}

// Backpropagation using GPU
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    // Copy target to device
    CUDA_CHECK(cudaMemcpy(net->d_target, target, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch kernels
    int blockSize = 128;
    int gridSize1 = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    int gridSize2 = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    
    // Compute output layer gradient
    backwardKernel1<<<gridSize1, blockSize>>>(net->d_output, net->d_target, net->d_d_output, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Compute hidden layer gradient
    backwardKernel2<<<gridSize2, blockSize>>>(net->d_W2, net->d_d_output, net->d_hidden, net->d_d_hidden, 
                                          HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Update weights and biases
    dim3 blockSize3(16, 16);
    dim3 gridSize3((OUTPUT_SIZE + blockSize3.x - 1) / blockSize3.x, 
                  (HIDDEN_SIZE + blockSize3.y - 1) / blockSize3.y);
    
    backwardKernel3<<<gridSize3, blockSize3>>>(net->d_W2, net->d_d_output, net->d_hidden, net->d_d_hidden, 
                                           LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    dim3 gridSize4((HIDDEN_SIZE + blockSize3.x - 1) / blockSize3.x, 
                  (INPUT_SIZE + blockSize3.y - 1) / blockSize3.y);
    
    backwardKernel4<<<gridSize4, blockSize3>>>(net->d_W1, net->d_d_hidden, net->d_input, LEARNING_RATE,
                                           INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    backwardKernel5<<<gridSize1, blockSize>>>(net->d_b2, net->d_d_output, LEARNING_RATE, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    backwardKernel6<<<gridSize2, blockSize>>>(net->d_b1, net->d_d_hidden, LEARNING_RATE, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Sync weights from device to host
void syncWeightsToHost(NeuralNetwork* net) {
    double* W1_flat = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* W2_flat = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    
    CUDA_CHECK(cudaMemcpy(W1_flat, net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(W2_flat, net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(net->b1, net->d_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(net->b2, net->d_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Unflatten arrays
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i][j] = W1_flat[i * INPUT_SIZE + j];
        }
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i][j] = W2_flat[i * HIDDEN_SIZE + j];
        }
    }
    
    free(W1_flat);
    free(W2_flat);
}

// Train network
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        
        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            forward(net, images[i], hidden, output);
            backward(net, images[i], hidden, output, labels[i]);
            
            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k] + 1e-10);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
            
            // Print progress
            if (i % 1000 == 0) {
                printf("Epoch %d: %d/%d images processed\r", epoch + 1, i, numImages);
                fflush(stdout);
            }
        }
        
        // Sync weights back to host for evaluation
        syncWeightsToHost(net);
        
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward(net, images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Read MNIST dataset
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// Free network memory
void freeNetwork(NeuralNetwork* net) {
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    
    // Free device memory
    CUDA_CHECK(cudaFree(net->d_W1));
    CUDA_CHECK(cudaFree(net->d_W2));
    CUDA_CHECK(cudaFree(net->d_b1));
    CUDA_CHECK(cudaFree(net->d_b2));
    CUDA_CHECK(cudaFree(net->d_input));
    CUDA_CHECK(cudaFree(net->d_hidden));
    CUDA_CHECK(cudaFree(net->d_output));
    CUDA_CHECK(cudaFree(net->d_target));
    CUDA_CHECK(cudaFree(net->d_d_hidden));
    CUDA_CHECK(cudaFree(net->d_d_output));
    
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network - GPU Implementation (V2)\n\n");
    
    // Check if CUDA device is available
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found\n");
        return EXIT_FAILURE;
    }
    
    // Print device info
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Using GPU: %s\n", deviceProp.name);
    
    // Load MNIST dataset
    double** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);
    
    // Create and train network
    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);
    
    // Free memory
    freeNetwork(net);
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);
    
    return 0;
}