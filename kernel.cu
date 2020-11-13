
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<cstdlib>
#include<ctime>
#include <iostream>
#include<fstream>
#include <cstringt.h>
#include<vector>
#include<windows.h>
#include<thread>
#include <stdio.h>
#include <iomanip>
#define RED     "\033[31m"      
#define GREEN   "\033[32m"      
#define YELLOW  "\033[33m"      
#define BLUE    "\033[34m"  
#define RESET   "\033[0m"
int GPU_CLOCK_RATE;
using namespace std;
clock_t start, v1, v2, v3, finish;
int readIn(float* matrixA,float* matrixB);                      //从指定路径读取指定二进制文件，必须提前准备好
float* matrix_product_v1(float* matrixA, float* matrixB, int ma, int na, int mb, int nb);
float** matrix_product_v2(float** matrixA, float** matrixB);
float** matrix_product_CUDA(float** matrixA, float** matrixB);
float** matrixCut(float* matrixA);
const int THREAD_NUM = thread::hardware_concurrency();
const int DATA_LENGTH = 200000000, DATA_HALF_LENGTH = 100000000, DATA_WIDTH = 10000, DATA_WIDTH_S = 5000;
int active_thread = 768 * 2;


struct matrix {
    int m, n;
    float* value;
};




__global__ void dot_product_Kernel(float* input, float* answer)
{
    int active_thread = 8 * 256;
    int x = threadIdx.x;
    int y = blockIdx.x;
    int pos = x * y;
    int left = pos; int right = DATA_HALF_LENGTH - 1 + pos;
    float ans = 0;
    while (right < DATA_LENGTH) {
        ans += input[left] * input[right];
        left += active_thread; right += active_thread;
    }
    answer[pos] = ans;

}
__global__ void matrix_minus_Kernel(float* ans,float* matrixA,float* matrixB) {
    int active_thread = 8 * 256;
    int x = threadIdx.x;
    int y = blockIdx.x;
    int pos = y * 256+x;
    int tmp = DATA_WIDTH_S * DATA_WIDTH_S;
    while (pos < tmp) {
        ans[pos] = matrixA[pos] - matrixB[pos];
        pos += active_thread;
    }

}
__global__ void matrix_add_Kernel(float* ans, float* matrixA, float* matrixB) {
    int active_thread = 8 * 256;
    int x = threadIdx.x;
    int y = blockIdx.x;
    int pos = y * 256 + x;
    int tmp = DATA_WIDTH_S * DATA_WIDTH_S;
    while (pos < tmp) {
        ans[pos] = matrixA[pos] + matrixB[pos];
        pos += active_thread;
    }

}
__global__ void matrix_product_Kernel(float* ans, float* matrixA, float* matrixB) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    int pos = y * 256 + x;
    int active_thread = 8 * 256;
    int ma = DATA_WIDTH_S;
    for (int i = 0; i < ma; i++) {
        pos = pos % (ma);
        for (pos; pos < ma; pos+=active_thread) {
            int a = i * ma + pos;
            ans[a] = 0;
            for (int k = 0; k + 9 < ma; k += 10) {
                 int b = i * ma + k; int c = k * ma + pos;
                ans[a] += matrixA[b++] * matrixB[c];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
            }
        }

    }
}
__global__ void smatrix_product_Kernel(float* ans, float* matrixA, float* matrixB) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    int ma = DATA_WIDTH_S;
    for (int i = y; i+8 < ma; i+=8) {
        __shared__ float row[DATA_WIDTH_S];
        int tmp = x;
        while (tmp < DATA_WIDTH_S) {
            row[tmp] = matrixA[i * ma + tmp];
            tmp += 256;
        }
        __syncthreads();
        for (int j = x; j< ma;j += 256) {
            int a = i * ma + j;
            ans[a] = 0;
            for (int k = 0; k + 9 < ma; k += 10) {
                int b =  k; int c = k * ma + j;
                float answer = 0;
                answer += row[b++] * matrixB[c];
                answer += row[b++] * matrixB[c += ma];
                answer += row[b++] * matrixB[c += ma];
                answer += row[b++] * matrixB[c += ma];
                answer += row[b++] * matrixB[c += ma];
                answer += row[b++] * matrixB[c += ma];
                answer += row[b++] * matrixB[c += ma];
                answer += row[b++] * matrixB[c += ma];
                answer += row[b++] * matrixB[c += ma];
                answer += row[b++] * matrixB[c += ma];
                ans[a] += answer;
            }
        }

    }
}
__global__ void warmup()
{
    /*预热GPU，调用一个空的核函数*/
}
void printDeviceProp(const cudaDeviceProp& prop)
{
    printf("Device Name : %s\n", prop.name);
    //printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
    //printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
    //printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d\n", prop.warpSize);
    //printf("memPitch : %d.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    //printf("totalConstMem : %d.\n", prop.totalConstMem);
    //printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d\n", prop.clockRate);
    //printf("textureAlignment : %d.\n", prop.textureAlignment);
    //printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d\n", prop.multiProcessorCount);
}
bool InitCUDA(){
    cudaError_t cudaStatus;
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0)
    {
        fprintf(stderr, "There is no device.\n");
    }
    int i;
    for (i = 0; i < count; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("CUDA device info as below:\n");
        printDeviceProp(prop);
        GPU_CLOCK_RATE = prop.clockRate;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
        {
            if (prop.major >= 1)
            {
                break;
            }
        }
    }
    if (i == count)
    {
        fprintf(stderr, "There is no device supporting CUDA 11.x.\n");
    }
    cudaSetDevice(i);
    return true;
}

int main()
{
    cout << RESET;
    printf("-------------Matrix Product---------------\nEnter number to choose:\n --------------------------------------\n|1-Calculate by inputing matrix A and B|\n --------------------------------------\n|2-Calculate by existed 200000000 data |\n --------------------------------------\nYour choice:");
    int op; cin >> op;
    if (op == 1) {
        printf("Enter the length of Matrix A(\"length\" \"width\"):");
        int ma; int na;
        cin >> ma; cin >> na;
        float* matrixA = new float[ma * na];
        printf("Enter the Matrix A line by line:\n");
        for (int i = 0; i < ma; i++) {
            for (int j = 0; j < na; j++) {
                cin >> matrixA[i * na + j];
            }
        }
        cout<<BLUE << ("Matrix A input complete:\n");
        
        for (int i = 0; i < ma; i++) {
            for (int j = 0; j < na; j++) {
                cout << setw(13) << left<<setfill(' ') << matrixA[i * na + j]<<" ";
            }
            cout << endl;
        }
        cout << RESET;
        printf("Enter the length of Matrix B:(\"length\" \"width\"):");
        int mb, nb;
        cin >> mb; cin >> nb;
        float* matrixB = new float[mb * nb];
        printf("Enter the Matrix B line by line:\n");
        for (int i = 0; i < mb; i++) {
            for (int j = 0; j < nb; j++) {
                cin >> matrixB[i * nb + j];
            }
        }
        cout <<BLUE<< ("Matrix B input complete:\n");
        
        for (int i = 0; i < mb; i++) {
            for (int j = 0; j < nb; j++) {
                cout << setw(13) << left << setfill(' ') << matrixB[i * nb + j] << " ";
            }
            cout << endl;
        }
        cout << RESET;
        if (na != mb) {
            cout <<RED<< "Wrong length!Please cheack carefully" << endl;
            cout << RESET;
        }
        else {
            
            float* ans = matrix_product_v1(matrixA, matrixB, ma, na, mb, nb);
            cout <<YELLOW<< "-------------------------------\nComplete!The answer matrix is:\n";
            cout << RESET;
            for (int i = 0; i < ma; i++) {
                for (int j = 0; j < nb; j++) {
                    cout << setw(13) << left << setfill(' ') << ans[i * nb + j] << " ";
                }
                cout << endl;
            }

        }   

    }
    else {
        cout<<GREEN<<"Initializing CUDA device......\n";
        cout << RESET;
        InitCUDA();
        float* input = new float[DATA_HALF_LENGTH];
        float* matrixA = new float[DATA_HALF_LENGTH];
        float* matrixB = new float[DATA_HALF_LENGTH];
        cout << GREEN << "Reading data from disk.....Please wait\n";
        cout << RESET;
        readIn(matrixA,matrixB);
        //读入数据
        float** ans;
        start = clock();
        cout << GREEN << "Calculating in brute Force....." << RED << "WARNING:This will take a long time!" << endl;
        cout << RESET;
        /*float* ans1 = matrix_product_v1(matrixA, matrixB, DATA_WIDTH, DATA_WIDTH, DATA_WIDTH, DATA_WIDTH);
        v1 = clock();
        cout << "Calculate time for [brute force in prue CPU] : " << (double)(v1 - start) / (CLOCKS_PER_SEC * 60) <<" Minutes"<< endl;
        cout <<BLUE<< "Calculate answer :\n "<< endl;
        for (int j = 0; j < 10; j++) {
            cout << setw(13) << left << setfill(' ') << ans1[j] << " ";
        }
        cout << RESET;
        cout << endl;
        cout << "Cores in this computer : " << THREAD_NUM << endl;*/
        float** Acut = matrixCut(matrixA);
        float** Bcut = matrixCut(matrixB);
        /*cout <<GREEN<< "Calculating in [Strassen + multi CPU]....." << endl;
        cout << RESET;
        ans = matrix_product_v2(Acut,Bcut);
        cout << "Calculate time for [Strassen + multi CPU] : " << (double)(finish - start) / (CLOCKS_PER_SEC*60) <<" Minutes"<< endl;
        cout <<BLUE<< "Calculate answer (first 10 numbers in answer matrix) : "<< endl;
        for (int j = 0; j < 10; j++) {
            cout << setw(13) << left << setfill(' ') << ans[0][j] << " ";
        }
        cout << RESET;
        cout << endl;*/
        cout << GREEN << "Calculating by [CUDA+Strassen]......." << endl;
        cout << RESET;
        ans = matrix_product_CUDA(Acut,Bcut);
    }
    return 0;
}

int readIn(float* matrixA,float* matrixB) {
    start = clock();
    int count = 0;
    ifstream ifile;
    ifile.open("D:\\Cpppppppp\\ldFeature.bin", ios::binary);
    float b;
    int i = 0;
    while (i++ < DATA_HALF_LENGTH) {
        ifile.read((char*)&b, sizeof(b));
        //cout << b << " ";
        matrixA[count++] = b;

    }
    i = 0; count = 0;
    while (i++ < DATA_HALF_LENGTH) {
        ifile.read((char*)&b, sizeof(b));
        //cout << b << " ";
        matrixB[count++] = b;

    }
    ifile.close();
    cout << "Total numbers in vector:" << count*2 << endl;
    finish = clock();
    cout << "IO read-in time : " << (double)(finish - start) / CLOCKS_PER_SEC << " Seconds"<<endl;
    return 1;
}
float** matrixCut(float* matrixA) {
        float** ans = new float*[4];
        float* temp1 = new float[DATA_WIDTH_S* DATA_WIDTH_S];
        float* temp2 = new float[DATA_WIDTH_S* DATA_WIDTH_S];
        for (int i = 0; i < DATA_WIDTH_S; i++) {
            for (int j = 0; j < DATA_WIDTH_S; j++) {
                temp1[i * DATA_WIDTH_S + j] = matrixA[i * DATA_WIDTH + j];
            }
        }
        for (int i = 0; i < DATA_WIDTH_S; i++) {
            for (int j = 0; j < DATA_WIDTH_S; j++) {
                temp2[i * DATA_WIDTH_S + j] = matrixA[i * DATA_WIDTH + j+DATA_WIDTH_S];
            }
        }
        ans[0] = temp1; ans[1] = temp2;
        float* temp3 = new float[DATA_WIDTH_S * DATA_WIDTH_S];
        float* temp4 = new float[DATA_WIDTH_S * DATA_WIDTH_S];
        int pass = DATA_WIDTH_S * DATA_WIDTH;
        for (int i = 0; i < DATA_WIDTH_S; i++) {
            for (int j = 0; j < DATA_WIDTH_S; j++) {
                temp3[i * DATA_WIDTH_S + j] = matrixA[i * DATA_WIDTH + j+pass];
            }
        }
        for (int i = 0; i < DATA_WIDTH_S; i++) {
            for (int j = 0; j < DATA_WIDTH_S; j++) {
                temp4[i * DATA_WIDTH_S + j] = matrixA[i * DATA_WIDTH + j+pass+DATA_WIDTH_S];
            }
        }
        ans[2] = temp3; ans[3] = temp4;
        delete[DATA_HALF_LENGTH] matrixA;
        return ans;
}
void threadMinus(float* ans, float* matrixA, float* matrixB, int ma, int na) {
    int count = 0;
    for (int i = 0; i < ma; i++) {
        for (int j = 0; j < na; j++) {
            ans[count] = matrixA[count] - matrixB[count];
            count++;
        }
    }
}
void threadAdd(float* ans, float* matrixA, float* matrixB, int ma, int na) {
    int count = 0;
    for (int i = 0; i < ma; i++) {
        for (int j = 0; j < na; j++) {
            ans[count] = matrixA[count] + matrixB[count];
            count++;
        }
    }
}
float* matrix_product_v1(float* matrixA, float* matrixB, int ma, int na, int mb, int nb) {
    float* ans = new float[ma * nb]{ 0 };
    for (int i = 0; i < ma; i++) {
        for (int j = 0; j < nb; j++) {
            for (int k = 0; k< na; k++) {
                ans[i * nb +j] += matrixA[i* na+k] * matrixB[k* nb+j];
            }
        }
    }
    return ans;
}
void thread_matrix_product(float* ans, float* matrixA, float* matrixB, int ma) {
    for (int i = 0; i < ma; i++) {
        for (int j = 0; j < ma; j++) {
            for (int k = 0; k + 9 < ma; k += 10) {
                int a = i * ma + j; int b = i * ma + k; int c = k * ma + j;
                ans[a] += matrixA[b++] * matrixB[c];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
                ans[a] += matrixA[b++] * matrixB[c += ma];
            }
        }

    }
}

float** matrix_product_v2(float** matrixA, float** matrixB) {
    float* A1 = matrixA[0];
    float* A2 = matrixA[1];
    float* A3 = matrixA[2];
    float* A4 = matrixA[3];
    float* B1 = matrixB[0];
    float* B2 = matrixB[1];
    float* B3 = matrixB[2];
    float* B4 = matrixB[3];
    start = clock();
    thread thread_pool_S[10];
    int temp = DATA_WIDTH_S * DATA_WIDTH_S;
    float* S1 = new float[temp];
    float* S2 = new float[temp];
    float* S3 = new float[temp];
    float* S4 = new float[temp];
    float* S5 = new float[temp];
    float* S6 = new float[temp];
    float* S7 = new float[temp];
    float* S8 = new float[temp];
    float* S9 = new float[temp];
    float* S10 = new float[temp];
    thread_pool_S[0] = thread(threadMinus, S1, B1, B4, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_S[1] = thread(threadAdd, S2, A1, A2, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_S[2] = thread(threadAdd, S3, A3, A4, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_S[3] = thread(threadMinus, S4, B3, B1, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_S[4] = thread(threadAdd, S5, A1, A4, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_S[5] = thread(threadAdd, S6, B1, B4, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_S[6] = thread(threadMinus, S7, A2, A4, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_S[7] = thread(threadAdd, S8, B3, B4, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_S[8] = thread(threadMinus, S9, A1, A3, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_S[9] = thread(threadAdd, S10, B1, B2, DATA_WIDTH_S, DATA_WIDTH_S);
    for (int i = 0; i < 10; i++) {
        thread_pool_S[i].join();
    }
    
    thread thread_pool_P[7];
    float* P1 = new float[temp] {0};
    float* P2 = new float[temp] {0};
    float* P3 = new float[temp] {0};
    float* P4 = new float[temp] {0};
    float* P5 = new float[temp] {0};
    float* P6 = new float[temp] {0};
    float* P7 = new float[temp] {0};
    thread_pool_P[0] = thread(thread_matrix_product, P1, A1, S1, DATA_WIDTH_S);
    thread_pool_P[1] = thread(thread_matrix_product, P2, S2, B4, DATA_WIDTH_S);
    thread_pool_P[2] = thread(thread_matrix_product, P3, S3, B1, DATA_WIDTH_S);
    thread_pool_P[3] = thread(thread_matrix_product, P4, A4, S4, DATA_WIDTH_S);
    thread_pool_P[4] = thread(thread_matrix_product, P5, S5, S6, DATA_WIDTH_S);
    thread_pool_P[5] = thread(thread_matrix_product, P6, S7, S8, DATA_WIDTH_S);
    thread_pool_P[6] = thread(thread_matrix_product, P7, S9, S10, DATA_WIDTH_S);
    for (int i = 0; i < 7; i++) {
        thread_pool_P[i].join();
    }
    delete[]    S9, S10;
    float** ans = new float*[4];
    thread thread_pool_F[4];
    thread_pool_F[0] = thread(threadAdd,S5, P5, P4, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_F[1] = thread(threadMinus, S6, P2, P6, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_F[2] = thread(threadAdd, S7, P5, P1, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_F[3] = thread(threadAdd, S8, P3, P7, DATA_WIDTH_S, DATA_WIDTH_S);
    for (int i = 0; i < 4; i++) {
        thread_pool_F[i].join();
    }
    thread_pool_F[0] = thread(threadMinus, S1, S5, S6, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_F[1] = thread(threadAdd, S2, P1, P2, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_F[2] = thread(threadAdd, S3, P3, P4, DATA_WIDTH_S, DATA_WIDTH_S);
    thread_pool_F[3] = thread(threadMinus, S4, S7, S8, DATA_WIDTH_S, DATA_WIDTH_S);
    for (int i = 0; i < 4; i++) {
        thread_pool_F[i].join();
    }
    ans[0] = S1;
    ans[1] = S2;
    ans[2] = S3;
    ans[3] = S4;
    finish = clock();
    return ans;
}
float** matrix_product_CUDA(float** matrixA, float** matrixB) {
    cout << "Copying data from RAM to GPU memory....." << endl;
    cudaError_t cudaStatus;
    int length = DATA_WIDTH_S * DATA_WIDTH_S;
    float* dev_A1; float* dev_A2; float* dev_A3; float* dev_A4; float* dev_B1; float* dev_B2; float* dev_B3; float* dev_B4;
    float* S1; float* S2; float* S3; float* S4; float* S5; float* S6; float* S7; float* S8; float* S9; float* S10;
    float* P1; float* P2; float* P3; float* P4; float* P5; float* P6; float* P7;
    cudaStatus = cudaMalloc((void**)&dev_A1, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_A2, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_A3, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_A4, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_B1, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_B2, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_B3, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_B4, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&S1, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&S2, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&S3, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&S4, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&S5, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&S6, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&S7, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&S8, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&S9, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&S10, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&P1, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&P2, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&P3, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&P4, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&P5, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&P6, length * sizeof(float));
    cudaStatus = cudaMalloc((void**)&P7, length * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMemcpy(dev_A1, matrixA[0], length, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaStatus = cudaMemcpy(dev_A2, matrixA[1], length * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_A3, matrixA[2], length * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_A4, matrixA[3], length * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_B1, matrixB[0], length * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_B2, matrixB[1], length * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_B3, matrixB[2], length * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_B4, matrixB[3], length * sizeof(float), cudaMemcpyHostToDevice);
    finish = clock();
    cout << "Data copy time for GPU : " << (double)(finish - start) / CLOCKS_PER_SEC << " Seconds" << endl;
    start = clock();
    warmup << <1, 1 >> > ();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    {
        matrix_minus_Kernel << <8, 256 >> > (S1, dev_B1, dev_B4);
        matrix_add_Kernel << <8, 256 >> > (S2, dev_A1, dev_A2);
        matrix_add_Kernel << <8, 256 >> > (S3, dev_A3, dev_A4);
        matrix_minus_Kernel << <8, 256 >> > (S4, dev_B3, dev_B1);
        matrix_add_Kernel << <8, 256 >> > (S5, dev_A1, dev_A4);
        matrix_add_Kernel << <8, 256 >> > (S6, dev_B1, dev_B4);
        matrix_minus_Kernel << <8, 256 >> > (S7, dev_A2, dev_A4);
        matrix_add_Kernel << <8, 256 >> > (S8, dev_B3, dev_B4);
        matrix_minus_Kernel << <8, 256 >> > (S9, dev_A1, dev_A3);
        matrix_add_Kernel << <8, 256 >> > (S10, dev_B1, dev_B2);
        cudaDeviceSynchronize();
        cout << "S part finished" << endl;
        matrix_product_Kernel << <8, 256 >> > (P1, dev_A1, S1);
        matrix_product_Kernel << <8, 256 >> > (P2, S2, dev_B4);
        matrix_product_Kernel << <8, 256 >> > (P3, S3, dev_B1);
        matrix_product_Kernel << <8, 256 >> > (P4, dev_A4, S4);
        matrix_product_Kernel << <8, 256 >> > (P5, S5, S6);
        matrix_product_Kernel << <8, 256 >> > (P6, S7, S8);
        matrix_product_Kernel << <8, 256 >> > (P7, S9, S10);
        cudaDeviceSynchronize();
        cout << "P part finished" << endl;
        matrix_add_Kernel << <8, 256 >> > (S5, P5, P4);
        matrix_minus_Kernel << <8, 256 >> > (S6, P2, P6);
        matrix_add_Kernel << <8, 256 >> > (S7, P5, P1);
        matrix_add_Kernel << <8, 256 >> > (S8, P3, P7);
        cudaDeviceSynchronize();
        matrix_minus_Kernel << <8, 256 >> > (S1, S5, S6);
        matrix_add_Kernel << <8, 256 >> > (S2, P1, P2);
        matrix_add_Kernel << <8, 256 >> > (S3, P3, P4);
        matrix_minus_Kernel << <8, 256 >> > (S4, S7, S8);
        cout << "C part finished" << endl;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    float costtime;
    cudaEventElapsedTime(&costtime, start, stop);
    float** ans = new float* [4];
    float* ans0 = new float[length];
    float* ans1 = new float[length];
    float* ans2 = new float[length];
    float* ans3 = new float[length];
    cudaStatus = cudaMemcpy(ans0, S1, length, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(ans1, S2, length, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(ans2, S3, length, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(ans3, S4, length, cudaMemcpyDeviceToHost);
    ans[0] = ans0;
    ans[1] = ans1;
    ans[2] = ans1;
    ans[3] = ans3;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cout << "Calculate time for [CUDA+Strassen] : " << costtime / (1000 * 60) << " Minutes" << endl;
    //cudaStatus = cudaMemcpy(answer, dev_answer, active_thread * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cout << BLUE << "Calculate answer (first 10 numbers in answer matrix) : " << endl;
    for (int j = 0; j < 10; j++) {
        cout << setw(13) << left << setfill(' ') << ans[0][j] << " ";
    }
    cout << RESET;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    {
        matrix_minus_Kernel << <8, 256 >> > (S1, dev_B1, dev_B4);
        matrix_add_Kernel << <8, 256 >> > (S2, dev_A1, dev_A2);
        matrix_add_Kernel << <8, 256 >> > (S3, dev_A3, dev_A4);
        matrix_minus_Kernel << <8, 256 >> > (S4, dev_B3, dev_B1);
        matrix_add_Kernel << <8, 256 >> > (S5, dev_A1, dev_A4);
        matrix_add_Kernel << <8, 256 >> > (S6, dev_B1, dev_B4);
        matrix_minus_Kernel << <8, 256 >> > (S7, dev_A2, dev_A4);
        matrix_add_Kernel << <8, 256 >> > (S8, dev_B3, dev_B4);
        matrix_minus_Kernel << <8, 256 >> > (S9, dev_A1, dev_A3);
        matrix_add_Kernel << <8, 256 >> > (S10, dev_B1, dev_B2);
        cudaDeviceSynchronize();
        cout << "S part finished" << endl;
        smatrix_product_Kernel << <8, 256 >> > (P1, dev_A1, S1);
        smatrix_product_Kernel << <8, 256 >> > (P2, S2, dev_B4);
        smatrix_product_Kernel << <8, 256 >> > (P3, S3, dev_B1);
        smatrix_product_Kernel << <8, 256 >> > (P4, dev_A4, S4);
        smatrix_product_Kernel << <8, 256 >> > (P5, S5, S6);
        smatrix_product_Kernel << <8, 256 >> > (P6, S7, S8);
        smatrix_product_Kernel << <8, 256 >> > (P7, S9, S10);
        cudaDeviceSynchronize();
        cout << "P part finished" << endl;
        matrix_add_Kernel << <8, 256 >> > (S5, P5, P4);
        matrix_minus_Kernel << <8, 256 >> > (S6, P2, P6);
        matrix_add_Kernel << <8, 256 >> > (S7, P5, P1);
        matrix_add_Kernel << <8, 256 >> > (S8, P3, P7);
        cudaDeviceSynchronize();
        matrix_minus_Kernel << <8, 256 >> > (S1, S5, S6);
        matrix_add_Kernel << <8, 256 >> > (S2, P1, P2);
        matrix_add_Kernel << <8, 256 >> > (S3, P3, P4);
        matrix_minus_Kernel << <8, 256 >> > (S4, S7, S8);
        cout << "C part finished" << endl;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&costtime, start, stop);
    cudaStatus = cudaMemcpy(ans0, S1, length, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(ans1, S2, length, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(ans2, S3, length, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(ans3, S4, length, cudaMemcpyDeviceToHost);
    ans[0] = ans0;
    ans[1] = ans1;
    ans[2] = ans1;
    ans[3] = ans3;
    cout << "Calculate time for [CUDA+Strassen+Shared Memory] : " << costtime / (1000 * 60) << " Minutes" << endl;
    cout << BLUE << "Calculate answer (first 10 numbers in answer matrix) : " << endl;
    for (int j = 0; j < 10; j++) {
        cout << setw(13) << left << setfill(' ') << ans[0][j] << " ";
    }
    cout << RESET;
    return ans;
}