#include<ap_int.h>
#include<hls_stream.h>

#define M 128
#define N 128
#define K 128

typedef float data_t;

static void load_A(const data_t *A, data_t Atile[M][K], int m0, int k0) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            #pragma HLS PIPELINE II=1
            Atile[i][j] = A[(m0 + i) * N + (k0 + j)];
        }
    }
}

static void load_B(const data_t *B, data_t Btile[K][N], int k0, int n0) {

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            Btile[i][j] = B[(k0 + i) * N + (n0 + j)];
        }
    }
}

static void compute_tile(const data_t Atile[M][K], const data_t Btile[K][N], data_t Ctile[M][N]) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            data_t sum = 0;
            for (int k = 0; k < K; k++) {
                #pragma HLS UNROLL
                sum += Atile[i][k] * Btile[k][j];
            }
            Ctile[i][j] += sum;
        }
    }
}

static void store_C(data_t *C, const data_t Ctile[M][N], int m0, int n0) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            C[(m0 + i) * N + (n0 + j)] = Ctile[i][j];
        }
    }
}

extern "C" {
void matmul_hls(const data_t *A, const data_t *B, data_t *C) {
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=A bundle=control
    #pragma HLS INTERFACE s_axilite port=B bundle=control
    #pragma HLS INTERFACE s_axilite port=C bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t Atile[M][K];
    data_t Btile[K][N];
    data_t Ctile[M][N];

    #pragma HLS ARRAY_PARTITION variable=Atile complete dim=2
    #pragma HLS ARRAY_PARTITION variable=Btile complete dim=1

    for(int m0 = 0; m0 < N; m0 += M) {
        for(int n0 = 0; n0 < N; n0 += N) {
            // Initialize Ctile to zero
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    #pragma HLS PIPELINE II=1
                    Ctile[i][j] = 0;
                }
            }
            for(int k0 = 0; k0 < K; k0 += K) {
                #pragma HLS DATAFLOW
                load_A(A, Atile, m0, k0);
                load_B(B, Btile, k0, n0);
                compute_tile(Atile, Btile, Ctile);
            }
            store_C(C, Ctile, m0, n0);
        }
    }
}
}