#define _POSIX_C_SOURCE 199309L
#include "zero_dl_transfer.h"
#include "portable/containers/zero_ztensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define IN_DIM 64
#define HIDDEN_DIM 32
#define NUM_CLASSES 4
#define EPOCHS 50
#define LEARNING_RATE 0.05f

int main(void) {
    printf("===================================================================\n");
    printf("  Fine-Tuning Domain Model in Pure ISO C99 via zero_dl\n");
    printf("  Engine: zero_dl_transfer.h + .ztensors Memory-Mapped Checkpoints\n");
    printf("===================================================================\n\n");

    const char* model_path = "/home/coder/workspace/dl-project-101-zero-dl/cat_dog_classifier.ztensors";
    printf("[1/4] Loading pre-trained base model weights from %s...\n", model_path);

    ZeroDLTransferModel* model = (ZeroDLTransferModel*)calloc(1, sizeof(ZeroDLTransferModel));
    assert(model != NULL);

    int ok = zero_dl_transfer_init_from_ztensors(
        model,
        model_path,
        "features.0.weight",
        "features.0.bias",
        IN_DIM,     /* 64 Input Features */
        HIDDEN_DIM, /* 32 Pre-trained Latent Embedding */
        NUM_CLASSES /* 4 Specialized Domain Tasks */
    );

    if (!ok) {
        printf("  [!] Failed to load model weights directly.\n");
        free(model);
        return 1;
    }

    printf("  [✓] Pretrained Backbone Loaded & Frozen (is_trainable = 0)!\n");
    printf("  [✓] Domain Adaptation Head Attached (is_trainable = 1)!\n\n");

    /* Domain Adaptation Training Samples (C99 / zero_udt tool prediction) */
    float X_domain[4][IN_DIM];
    float y_domain[4][NUM_CLASSES];
    memset(X_domain, 0, sizeof(X_domain));
    memset(y_domain, 0, sizeof(y_domain));

    /* Sample 0: C99 / zugrep query */
    X_domain[0][0] = 1.0f; X_domain[0][12] = 1.0f;
    y_domain[0][0] = 1.0f; /* Class 0: zugrep */

    /* Sample 1: Linux / zufind query */
    X_domain[1][1] = 1.0f; X_domain[1][44] = 1.0f;
    y_domain[1][1] = 1.0f; /* Class 1: zufind */

    /* Sample 2: zero_udt TPHT lookup query */
    X_domain[2][2] = 1.0f; X_domain[2][32] = 1.0f;
    y_domain[2][2] = 1.0f; /* Class 2: zudict */

    /* Sample 3: zero_wal crash safety query */
    X_domain[3][3] = 1.0f; X_domain[3][48] = 1.0f;
    y_domain[3][3] = 1.0f; /* Class 3: zuwal */

    printf("[2/4] Executing Pure C99 zero_dl Backprop Fine-Tuning Loop (%d Epochs)...\n", EPOCHS);
    struct timespec s, e;
    clock_gettime(CLOCK_MONOTONIC, &s);

    float initial_loss = 0.0f, final_loss = 0.0f;

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        for (int i = 0; i < 4; ++i) {
            float loss = zero_dl_transfer_train_step(model, X_domain[i], y_domain[i], LEARNING_RATE);
            epoch_loss += loss;
        }
        if (epoch == 1) initial_loss = epoch_loss / 4.0f;
        if (epoch == EPOCHS) final_loss = epoch_loss / 4.0f;

        if (epoch % 10 == 0 || epoch == 1) {
            printf("  Epoch %2d/%2d -> Domain Adaptation Loss: %.6f\n", epoch, EPOCHS, epoch_loss / 4.0f);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &e);

    double elapsed_ms = (double)((e.tv_sec - s.tv_sec) * 1000000000ULL + (e.tv_nsec - s.tv_nsec)) / 1000000.0;
    printf("\n[3/4] Fine-Tuning Summary:\n");
    printf("  - Initial Loss: %.6f -> Final Loss: %.6f (%.1f%% error reduction)\n",
           initial_loss, final_loss, (1.0f - final_loss / initial_loss) * 100.0f);
    printf("  - Total Training Time: %.2f ms (%.3f ms/epoch)\n\n", elapsed_ms, elapsed_ms / (double)EPOCHS);

    /* [4/4] Test Fine-Tuned Domain Prediction */
    printf("[4/4] Validating Specialized Prediction:\n");
    float pred[NUM_CLASSES];
    zero_dl_transfer_predict(model, X_domain[2], pred);
    printf("  Input Query: 'zero_udt TPHT lookup'\n");
    printf("  Specialized Tool Prediction Probabilities:\n");
    printf("    [0] zugrep: %.4f | [1] zufind: %.4f | [2] zudict (TPHT): %.4f (Target = 1.0) | [3] zuwal: %.4f\n",
           pred[0], pred[1], pred[2], pred[3]);

    assert(pred[2] > pred[0] && pred[2] > pred[1]);
    printf("\n✓ Fine-Tuned Model Verified via Pure C99 zero_dl Engine!\n");

    zero_dl_transfer_destroy(model);
    free(model);
    return 0;
}
