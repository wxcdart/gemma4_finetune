/*
 * 05_zero_ml_dl_signal_suite.c
 * ----------------------------
 * Complete QNZ example using zero_ml, zero_dl, zero_signal, zero_vision, and zero_nlp.
 *
 * Compilation:
 *   gcc -O2 -std=c99 -D_GNU_SOURCE 05_zero_ml_dl_signal_suite.c -I/home/coder/workspace/qnz -I/home/coder/workspace/qnz/src/bearssl -lpthread -lm -o zero_suite_demo
 *   ./zero_suite_demo
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "zero.h"
#include "zero_solver.h"
#include "zero_features.h"
#include "zero_ml.h"
#include "zero_dl.h"
#include "zero_signal.h"

void demo_zero_signal(void) {
    printf("[1] --- DEMO: zero_signal (Digital Signal Processing) ---\n");

    const int N = 16;
    double window[16];
    
    // Create Hann window
    zero_signal_create_window(ZERO_WINDOW_HANN, window, N);
    printf("   Hann Window created. First 4 coefficients: [%.3f, %.3f, %.3f, %.3f]\n",
           window[0], window[1], window[2], window[3]);

    // Fast Fourier Transform (FFT) test signal
    ZeroComplex cplx_signal[16];
    for (int i = 0; i < N; i++) {
        double val = sin(2.0 * 3.141592653589793 * 2.0 * i / N);
        cplx_signal[i].real = val;
        cplx_signal[i].imag = 0.0;
    }

    // Forward FFT
    zero_signal_fft(cplx_signal, N, false);
    double mag = sqrt(cplx_signal[2].real * cplx_signal[2].real + cplx_signal[2].imag * cplx_signal[2].imag);
    printf("   FFT computed. Bin 2 magnitude: %.3f\n", mag);

    // Inverse FFT
    zero_signal_fft(cplx_signal, N, true);
    printf("   Inverse FFT reconstructed sample 4: %.3f\n\n", cplx_signal[4].real);
}

void demo_zero_ml(void) {
    printf("[2] --- DEMO: zero_ml (Classical Machine Learning) ---\n");

    // 1. Linear Regression GD Fit
    ZeroMLLinearRegression lr;
    zero_ml_linreg_init(&lr, 2, 0.01);
    double X_lr[4][2] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
    double y_lr[4] = {2.0, 4.0, 6.0, 8.0};
    zero_ml_linreg_fit_gd(&lr, (double*)X_lr, y_lr, 4, 0.05, 500);
    double q_lr[2] = {5.0, 5.0};
    double p_lr = zero_ml_linreg_predict(&lr, q_lr);
    printf("   Linear Regression Pred for {5,5}: %.2f\n", p_lr);

    // 2. Logistic Regression Probabilities
    ZeroMLLogisticRegression logreg;
    zero_ml_logistic_init(&logreg, 2);
    double X_log[4][2] = {{1, 1}, {1, 2}, {5, 5}, {6, 6}};
    double y_log[4] = {0.0, 0.0, 1.0, 1.0};
    zero_ml_logistic_fit(&logreg, (double*)X_log, y_log, 4, 0.1, 500);
    double q_log[2] = {7.0, 7.0};
    double p_log = zero_ml_logistic_predict_prob(&logreg, q_log);
    printf("   Logistic Regression Prob for {7,7}: %.4f\n", p_log);

    // 3. K-Means Clustering
    ZeroMLKMeans km;
    zero_ml_kmeans_init(&km, 2, 2);
    ZeroArena km_arena;
    zero_arena_init(&km_arena, 4096);
    double X_km[6][2] = {{1, 1}, {1, 2}, {2, 1}, {10, 10}, {10, 11}, {11, 10}};
    zero_ml_kmeans_fit(&km, (double*)X_km, 6, 20, &km_arena);
    double q_km1[2] = {1.5, 1.5};
    double q_km2[2] = {10.5, 10.5};
    size_t c1 = zero_ml_kmeans_predict(&km, q_km1);
    size_t c2 = zero_ml_kmeans_predict(&km, q_km2);
    zero_arena_destroy(&km_arena);
    printf("   K-Means Clusters: Cluster A=%zu, Cluster B=%zu\n\n", c1, c2);
}

void demo_zero_dl(void) {
    printf("[3] --- DEMO: zero_dl (Deep Learning & Autograd Graph) ---\n");

    ZeroDLGraph g;
    zero_dl_init(&g);
    
    ZeroDLNode* in  = zero_dl_var(&g, 1, 2, 0);       /* Input x: [1, 2] */
    ZeroDLNode* w1  = zero_dl_var(&g, 2, 2, 1);       /* Weight W1: [2, 2] */
    ZeroDLNode* b1  = zero_dl_var(&g, 1, 2, 1);       /* Bias B1: [1, 2] */
    ZeroDLNode* tgt = zero_dl_var(&g, 1, 2, 0);      /* Target y: [1, 2] */

    in->val[0] = 1.0f;  in->val[1] = 2.0f;
    w1->val[0] = 0.5f;  w1->val[1] = -0.5f;
    w1->val[2] = 0.5f;  w1->val[3] = 0.5f;
    b1->val[0] = 0.1f;  b1->val[1] = 0.1f;
    tgt->val[0] = 2.0f; tgt->val[1] = 1.0f;

    ZeroDLNode* z1   = zero_dl_dense(&g, in, w1, b1);
    ZeroDLNode* a1   = zero_dl_relu(&g, z1);
    ZeroDLNode* loss = zero_dl_mse_loss(&g, a1, tgt);

    zero_dl_forward(&g);
    float initial_loss = loss->val[0];
    zero_dl_backward(&g);
    zero_dl_step_sgd(&g, 0.1f);
    zero_dl_forward(&g);
    float updated_loss = loss->val[0];

    printf("   Autograd MLP Graph Step: Initial Loss = %.4f -> Updated Loss = %.4f\n",
           initial_loss, updated_loss);
    zero_dl_destroy(&g);

    // Vision Conv2D / MaxPool2D
    float img[16] = {1, 2, 0, 1, 0, 3, 1, 0, 2, 1, 4, 2, 0, 1, 2, 1};
    float pool_out[4];
    zero_vision_maxpool2d(img, 1, 4, 4, pool_out, 2, 2);
    printf("   MaxPool2D Output [0]=%.1f, [3]=%.1f\n\n", pool_out[0], pool_out[3]);
}

int main(void) {
    printf("==================================================\n");
    printf("   QNZ zero_ml / zero_dl / zero_signal Suite    \n");
    printf("==================================================\n\n");

    demo_zero_signal();
    demo_zero_ml();
    demo_zero_dl();

    printf("[OK] All QNZ tool suites (zero_ml, zero_dl, zero_signal) executed successfully!\n");
    return 0;
}
