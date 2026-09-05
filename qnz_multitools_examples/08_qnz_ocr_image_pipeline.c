/*
 * 08_qnz_ocr_image_pipeline.c
 * ---------------------------
 * QNZ Optical Character Recognition (OCR) & Image Processing Pipeline Example.
 * Demonstrates:
 *   1. Allocating RGB image buffers on ZeroArena (zero_image.h)
 *   2. Rendering synthetic text glyphs into raw pixel buffers
 *   3. Grayscale conversion & Otsu threshold binarization
 *   4. Extracting Text OCR Glyphs & Page recognition (zero_ocr.h)
 *   5. Table Structure Extraction (zero_ocr_extract_table)
 *
 * Compilation:
 *   gcc -O2 -std=c99 -D_GNU_SOURCE 08_qnz_ocr_image_pipeline.c \
 *       -I/home/coder/workspace/qnz \
 *       -I/home/coder/workspace/qnz/include \
 *       -I/home/coder/workspace/qnz/src/bearssl \
 *       -lpthread -lm -ldl -o qnz_ocr_demo
 *   ./qnz_ocr_demo
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zero.h"
#include "zero_sugar.h"
#include "zero_image.h"
#include "zero_ocr.h"

int main(void) {
    printf("==================================================================\n");
    printf("   QNZ Optical Character Recognition (OCR) & Image Pipeline       \n");
    printf("==================================================================\n\n");

    ZeroArena arena;
    zero_arena_init(&arena, 2 * 1024 * 1024);

    // 1. Allocate 64x32 RGB Image
    printf("[1] Allocating 64x32 RGB Image Buffer...\n");
    ZeroImage rgb;
    if (!zero_image_alloc_arena(&rgb, 64, 32, 3, &arena)) {
        printf("   FAILED: Image allocation error.\n");
        return 1;
    }

    // Fill background with white (255, 255, 255)
    memset(rgb.data, 255, rgb.width * rgb.height * 3);

    // Draw two simulated black text glyph bars
    printf("[2] Rendering synthetic text glyphs into image buffer...\n");
    for (int y = 8; y < 24; y++) {
        for (int x = 10; x < 18; x++) {
            size_t idx = (y * rgb.stride) + (x * 3);
            rgb.data[idx] = 0; rgb.data[idx + 1] = 0; rgb.data[idx + 2] = 0;
        }
        for (int x = 30; x < 38; x++) {
            size_t idx = (y * rgb.stride) + (x * 3);
            rgb.data[idx] = 0; rgb.data[idx + 1] = 0; rgb.data[idx + 2] = 0;
        }
    }

    // 2. Grayscale & Otsu Threshold Binarization
    printf("[3] Converting RGB to Grayscale and applying Otsu Binarization...\n");
    ZeroImage gray, binary;
    zero_image_convert_to_grayscale(&rgb, &gray, &arena);
    uint8_t otsu_val = zero_image_otsu_threshold(&gray);
    zero_image_binarize(&gray, &binary, otsu_val, &arena);
    printf("   Otsu Threshold calculated: %u\n", otsu_val);

    // 3. OCR Page Recognition Pipeline
    printf("[4] Executing zero_ocr Page Recognition Pipeline...\n");
    ZeroOCRResult ocr_res;
    if (zero_ocr_recognize_page(&rgb, &ocr_res, &arena)) {
        printf("   OCR Detection Successful: Extracted %zu glyph(s). Recognised Text=\"%s\"\n",
               ocr_res.num_glyphs, ocr_res.text);
    }

    // 4. OCR Table Grid Structure Extraction
    printf("[5] Extracting OCR Table Structure...\n");
    ZeroOCRTable table;
    if (zero_ocr_extract_table(&rgb, &table, &arena)) {
        printf("   OCR Table Extracted: %zu rows, %zu cols, %zu cells\n",
               table.rows, table.cols, table.cell_count);
    }

    zero_arena_destroy(&arena);
    printf("\n==================================================================\n");
    printf("  ✓ QNZ OCR & IMAGE PIPELINE EXECUTED SUCCESSFULLY!               \n");
    printf("==================================================================\n");
    return 0;
}
