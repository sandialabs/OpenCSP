// Memory allocator by gingerbill https://www.gingerbill.org/article/2019/02/16/memory-allocation-strategies-004/

#pragma once

#include <stddef.h>
#include <stdint.h>

#if !defined(__cplusplus)
#if (defined(_MSC_VER) && _MSC_VER < 1800) || (!defined(_MSC_VER) && !defined(__STDC_VERSION__))
#ifndef true
#define true  (0 == 0)
#endif
#ifndef false
#define false (0 != 0)
#endif
typedef unsigned char bool;
#else
#include <stdbool.h>
#endif
#endif

#include <stdio.h>
#include <assert.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

bool is_power_of_two(uintptr_t x);

uintptr_t align_forward_uintptr(uintptr_t ptr, uintptr_t align);

size_t align_forward_size(size_t ptr, size_t align);

#ifndef DEFAULT_ALIGNMENT
#define DEFAULT_ALIGNMENT 8
#endif

typedef struct Pool_Free_Node Pool_Free_Node;
struct Pool_Free_Node {
	Pool_Free_Node* next;
};

typedef struct Pool Pool;
struct Pool {
	unsigned char* buf;
	size_t buf_len;
	size_t chunk_size;

	Pool_Free_Node* head; // Free List Head
};

void pool_free_all(Pool* p);

void pool_init(Pool* p, void* backing_buffer, size_t backing_buffer_length,
			   size_t chunk_size, size_t chunk_alignment);


void* pool_alloc(Pool* p);


void pool_free(Pool* p, void* ptr);


#ifdef __cplusplus
}
#endif
