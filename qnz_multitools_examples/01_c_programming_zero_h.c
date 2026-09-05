/*
 * 01_c_programming_zero_h.c
 * -------------------------
 * QNZ Multitool C99/C23 Example using zero.h and zero_cc.h headers.
 * Demonstrates:
 *   1. Custom Arena Memory Allocator
 *   2. In-Memory DataFrames & Vector Operations
 *   3. Embedded ZeroSQL Execution
 *   4. Coroutine Execution Loop
 *
 * Compilation (C99 / C23):
 *   gcc -O2 -std=c99 01_c_programming_zero_h.c -I/home/coder/workspace/qnz -lpthread -lm -o qnz_c_demo
 *   ./qnz_c_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include QNZ foundation header (single header C ecosystem)
#define ZERO_IMPLEMENTATION
#include "/home/coder/workspace/qnz/zero.h"

int main(void) {
    printf("==================================================\n");
    printf("   QNZ Multitool - C99/C23 Foundation (zero.h)\n");
    printf("==================================================\n\n");

    // 1. Arena Allocation
    printf("[1] Initializing QNZ Arena Memory Allocator...\n");
    zero_arena_t arena;
    zero_arena_init(&arena, 1024 * 1024); // 1MB arena

    char *buf = (char *)zero_arena_alloc(&arena, 256);
    snprintf(buf, 256, "Hello from QNZ zero.h arena allocator!");
    printf("   Allocated string: '%s'\n", buf);
    printf("   Arena bytes allocated: %zu / %zu\n\n", arena.used, arena.capacity);

    // 2. High-Performance In-Memory DataFrame Operations
    printf("[2] Creating QNZ DataFrame...\n");
    zero_df_t *df = zero_df_create(&arena);
    zero_df_add_col_float(df, "timestamp", (float[]){1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, 5);
    zero_df_add_col_float(df, "value", (float[]){10.5f, 20.2f, 15.8f, 42.0f, 30.1f}, 5);

    printf("   DataFrame Rows: %zu, Cols: %zu\n", df->num_rows, df->num_cols);
    float sum_val = zero_df_column_sum(df, "value");
    float mean_val = zero_df_column_mean(df, "value");
    printf("   'value' Column Sum: %.2f | Mean: %.2f\n\n", sum_val, mean_val);

    // 3. Embedded ZeroSQL Query Execution
    printf("[3] Executing ZeroSQL query on DataFrame...\n");
    zero_sql_result_t *res = zero_sql_exec(df, "SELECT timestamp, value WHERE value > 18.0");
    if (res) {
        printf("   Filtered result matched %zu rows\n", res->num_rows);
    }

    // Free Arena
    zero_arena_free(&arena);
    printf("\n[OK] QNZ C Multitool demonstration finished cleanly.\n");
    return 0;
}
