#include <math_constants.h>

constexpr int B_r = 16;
constexpr int B_c = 16;
constexpr int d = 128;
constexpr int n_out_max = 4096;


extern "C" __global__ void silly_attn(
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
    // Thread indices
    int tid_x = threadIdx.x; // 0..3 (block_x_dim)
    int tid_y = threadIdx.y; // 0..31 (block_y_dim)

    // Shared memory buffers for Q, K, V blocks
    __shared__ float Q_i[B_r][d];  // 16 x 128
    __shared__ float K_j[B_c][d];  // 16 x 128
    __shared__ float V_j[B_c][d];  // 16 x 128

    // Local accumulators per thread for output block
    float O_i[B_r][d];
    float l_i[B_r];
    float m_i[B_r];
    // S_i: B_c temporary storage for scores
    float S_i[B_c];

    // Loop over output tile blocks (T_r)
    for (int i=0; i < T_r; i++) {
        // Init O_i, l_i, m_i into registers
    
        // Load Q_i tile into shared mem
        for (int ii = tid_y; ii < B_r; ii += blockDim.y) {
            for (int dd = tid_x; dd < d; dd += blockDim.x) {
                Q_i[ii][dd] = Q[(ii + i * B_r) * d + dd];
                O_i[ii][dd] = 0.0f;
            }
            l_i[ii] = 0.0f;
            m_i[ii] = -INFINITY;
        }
        __syncthreads();

        for (int j = 0; j < T_c; j++){
            // Load K_j, V_j into shared memory
            for (int jj = tid_y; jj < B_c; jj += blockDim.y) {
                for (int dd = tid_x; dd < d; dd += blockDim.x) {
                    K_j[jj][dd] = K[(jj + j * B_c) * d + dd];
                    V_j[jj][dd] = V[(jj + j * B_c) * d + dd];
                }
            }
            __syncthreads();
            // S_i = scaling * Q_i @ K_j.T
            for (int ii = tid_x; ii < B_r; ii += blockDim.x) {
                for (int jj = tid_y; jj < B_c; jj += blockDim.y) {
                    float S_ij = 0.0f;
                    for (int dd = 0; dd < d; dd ++) {
                        S_ij += Q_i[ii][dd] * K_j[jj][dd];
                    }
                    S_ij = scaling * S_ij;
                    S_i[jj] = S_ij;
                }
                __syncthreads();
                
                float m = m_i[ii];
                float last_m = m;
                for (int jj = 0; jj < B_c; jj++) {
                    if (m < S_i[jj]) {
                        m = S_i[jj];
                    }
                }
                m_i[ii] = m;
                float l = exp(last_m - m) * l_i[ii];

                for (int dd = tid_x; dd < d; dd += blockDim.x) {
                    O_i[ii][dd] *= exp(last_m - m); // Scale row elements
                }

                for (int jj = 0; jj < B_c; jj++) {
                    float S_ij = exp(S_i[jj] - m);
                    l += S_ij;
                    for (int dd = tid_x; dd < d; dd += blockDim.x) {
                        O_i[ii][dd] +=  S_ij * V_j[jj][dd];
                    }
                }
                l_i[ii] = l;

                for (int dd = tid_x; dd < d; dd += blockDim.x) {
                    out[(ii + i * B_r) * d + dd] = O_i[ii][dd] / l_i[ii];
                }
                out_l[ii + i * B_r] = l_i[ii]; 
            }
        }
    }
}



__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a+b-1)/b; }


std::tuple<torch::Tensor, torch::Tensor> flash_attention_register_spilling(torch::Tensor K, torch::Tensor Q, torch::Tensor V) {
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
    dim3 blocks(32, 4); 
    silly_attn<<<blocks, tpb>>>(
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

