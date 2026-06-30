#include "pool.h"


// Returns true if x is a power of two (required for alignment checks).
bool is_power_of_two(uintptr_t x)
{
	return (x & (x - 1)) == 0;
}


// Rounds ptr up to the next multiple of align (pointer-sized).
uintptr_t align_forward_uintptr(uintptr_t ptr, uintptr_t align)
{
	uintptr_t a, p, modulo;

	assert(is_power_of_two(align));

	a = align;
	p = ptr;
	modulo = p & (a - 1);
	if (modulo != 0)
	{
		p += a - modulo;
	}
	return p;
}


// Rounds ptr up to the next multiple of align (size_t variant, used for chunk sizes).
size_t align_forward_size(size_t ptr, size_t align)
{
	size_t a, p, modulo;

	assert(is_power_of_two((uintptr_t)align));

	a = align;
	p = ptr;
	modulo = p & (a - 1);
	if (modulo != 0)
	{
		p += a - modulo;
	}
	return p;
}


// Initializes a pool allocator over the provided backing buffer, aligning chunks and building the free list.
void pool_init(Pool* p, void* backing_buffer, size_t backing_buffer_length,
	size_t chunk_size, size_t chunk_alignment)
{
	// Align backing buffer to the specified chunk alignment
	uintptr_t initial_start = (uintptr_t)backing_buffer;
	uintptr_t start = align_forward_uintptr(initial_start, (uintptr_t)chunk_alignment);
	backing_buffer_length -= (size_t)(start - initial_start);

	// Align chunk size up to the required chunk_alignment
	chunk_size = align_forward_size(chunk_size, chunk_alignment);

	// Assert that the parameters passed are valid
	assert(chunk_size >= sizeof(Pool_Free_Node) && "Chunk size is too small");
	assert(backing_buffer_length >= chunk_size && "Backing buffer length is smaller than the chunk size");

	// Store the adjusted parameters
	p->buf = (unsigned char*)backing_buffer;
	p->buf_len = backing_buffer_length;
	p->chunk_size = chunk_size;
	p->head = NULL; // Free List Head

	// Set up the free list for free chunks
	pool_free_all(p);
}


// Pops one zeroed chunk from the free list; asserts if the pool is exhausted.
void* pool_alloc(Pool* p)
{
	// Get latest free node
	Pool_Free_Node* node = p->head;

	if (node == NULL)
	{
		assert(0 && "Pool allocator has no free memory");
		return NULL;
	}

	// Pop free node
	p->head = p->head->next;

	// Zero memory by default
	return memset(node, 0, p->chunk_size);
}


// Returns one chunk back to the pool's free list; ignores NULL and asserts on out-of-range pointers.
void pool_free(Pool* p, void* ptr)
{
	Pool_Free_Node* node;

	void* start = p->buf;
	void* end = &p->buf[p->buf_len];

	if (ptr == NULL)
	{
		// Ignore NULL pointers
		return;
	}

	if (!(start <= ptr && ptr < end))
	{
		assert(0 && "Memory is out of bounds of the buffer in this pool");
		return;
	}

	// Push free node
	node = (Pool_Free_Node*)ptr;
	node->next = p->head;
	p->head = node;
}


// Rebuilds the free list so every chunk in the backing buffer is available again.
void pool_free_all(Pool* p)
{
	size_t chunk_count = p->buf_len / p->chunk_size;
	size_t i;

	// Set all chunks to be free
	for (i = 0; i < chunk_count; i++)
	{
		void* ptr = &p->buf[i * p->chunk_size];
		Pool_Free_Node* node = (Pool_Free_Node*)ptr;
		// Push free node onto thte free list
		node->next = p->head;
		p->head = node;
	}
}
