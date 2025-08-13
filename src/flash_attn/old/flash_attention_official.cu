#include <math_constants.h>

constexpr int B_r = 16;
constexpr int B_c = 16;
constexpr int d = 128;
constexpr int n_out_max = 4096;
constexpr int block_dim_x = 32;
constexpr int block_dim_y = 32;
constexpr int o_per_thread_x = 1;
constexpr int o_per_thread_y = 128 / 32;

#define NEG_INFINITY __int_as_float(0xff800000)




extern "C" __global__ void flash_attention_k(
    float *out, 
    float *out_l, 
    float *K, 
    float *Q, 
    float *V, 
    float scaling, 
    int n, 
    int T_r, 
    int T_c
) {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    __shared__ float Q_i[B_r][d];
    __shared__ float K_j[B_c][d];
    __shared__ float V_j[B_c][d];

    __shared__ float S_i[B_r][B_c];

    float l_i[o_per_thread_x];
    float m_i[o_per_thread_x];
    float O_i[o_per_thread_x][o_per_thread_y];

    for (int i = 0; i < T_r; i++) {
        for (int ii = 0; ii < o_per_thread_x; ii++) {
            for (int dd = 0; dd < o_per_thread_y; dd++) {
                O_i[ii][dd] = 0;
            }
            l_i[ii] = 0.f;
            m_i[ii] = NEG_INFINITY;
        }
        
        for (int ii = tid_y; ii < B_r; ii += blockDim.y) {
            for (int dd = tid_x; dd < d; dd += blockDim.x) {
                Q_i[ii][dd] = Q[(ii + i * B_r) * d + dd];
            }
        }
        
        for (int j = 0; j < T_c; j++) {
            __syncthreads();
            for (int jj = tid_y; jj < B_c; jj += blockDim.y) {
                for (int dd = tid_x; dd < d; dd += blockDim.x) {
                    K_j[jj][dd] = K[(jj + j * B_c) * d + dd];
                    V_j[jj][dd] = V[(jj + j * B_c) * d + dd];
                }
            }
            __syncthreads();
            // S_i = scale_factor * (Q_i @ K_j.T);
            for (int ii = tid_x; ii < B_r; ii += blockDim.x) {
                for (int jj = tid_y; jj < B_c; jj += blockDim.y) {
                    float S_ij = 0.f;
                    for (int dd = 0; dd < d; dd++) {
                        S_ij += Q_i[ii][dd] * K_j[jj][dd];
                    }
                    S_ij = scaling * S_ij;
                    S_i[ii][jj] = S_ij;
                }
            }
            __syncthreads();
            for (int ii = 0; ii < o_per_thread_x; ii++) {
                float m = m_i[ii];
                float last_m = m;
                for (int jj = 0; jj < B_c; jj+=1) {
                    if (m < S_i[ii * blockDim.x + tid_x][jj]) {
                        m = S_i[ii * blockDim.x + tid_x][jj];
                    }
                }
                m_i[ii] = m;
                float l = exp(last_m - m) * l_i[ii];
                for (int dd = 0; dd < o_per_thread_y; dd++) {
                    O_i[ii][dd] *= exp(last_m - m);
                }
            
                for (int jj = 0; jj < B_c; jj++) {
                    float S_ij = exp(S_i[ii * blockDim.x + tid_x][jj] - m);
                    l += S_ij;
                    for (int dd = 0; dd < o_per_thread_y; dd++) {
                        O_i[ii][dd] += S_ij * V_j[jj][dd * blockDim.y + tid_y];
                    }
                }
                l_i[ii] = l;
                
            }
        }
        for (int ii = 0; ii < o_per_thread_x; ii++) {
            for (int dd = 0; dd < o_per_thread_y; dd++) {
                out[(ii * blockDim.x + tid_x + i * B_r) * d + dd * blockDim.y + tid_y] = O_i[ii][dd] / l_i[ii];
                out_l[ii * blockDim.x + tid_x + i * B_r] = l_i[ii];
            }
        }
    }
}



__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a+b-1)/b; }


std::tuple<torch::Tensor, torch::Tensor> flash_attention_official(torch::Tensor K, torch::Tensor Q, torch::Tensor V) {
    int n = Q.size(0);
    int n_inp = K.size(0);
    int d = Q.size(1);
    
    assert (n_out_max >= n && "Max size of rows exceeded!");
    assert (d == V.size(1) && "Size mismatch!");
    assert (d == K.size(1) && "Size mismatch!");
    assert (K.size(0) == V.size(0) && "Size mismatch!");
    auto out = torch::zeros({n, d}, Q.options());
    auto out_l = torch::zeros({n,}, Q.options());

    float scaling = 1.0f / sqrt((float)d);

    int T_r = cdiv(n, B_r);
    int T_c = cdiv(n_inp, B_c);

    dim3 tpb(1, 1);      
    dim3 blocks(block_dim_y, block_dim_x); 
    flash_attention_k<<<blocks, tpb>>>(
        out.data_ptr<float>(),
        out_l.data_ptr<float>(),
        K.data_ptr<float>(), 
        Q.data_ptr<float>(), 
        V.data_ptr<float>(), 
        scaling,
        n,
        T_r,
        T_c
    );
    return std::make_tuple(out, out_l);
}

