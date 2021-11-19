__kernel void gemm(const unsigned int m, const unsigned int n, const unsigned int k, __global float* a, __global float* b, __global float* c) {
	const unsigned int i = get_local_id(1);  // n
	const unsigned int j = get_local_id(0);  // k
	const unsigned int global_i = BLOCK * get_group_id(1) + i;  // n
	const unsigned int global_j = BLOCK * get_group_id(0) + j;  // k
	__local float local_a[BLOCK][BLOCK];
	__local float local_b[BLOCK][BLOCK];
	const unsigned int num_tiles = m / BLOCK;
	float result = .0;
	for (unsigned int t = 0; t < num_tiles; t++) {
		const unsigned int tiled_i = BLOCK * t + i;
		const unsigned int tiled_j = BLOCK * t + j;
		local_a[i][j] = a[global_i * m + tiled_j];
		local_b[i][j] = b[tiled_i * k + global_j];
		barrier(CLK_LOCAL_MEM_FENCE);
		for (unsigned int l = 0; l < BLOCK; l++) {
			result += local_a[i][l] * local_b[l][j];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	c[global_i * k + global_j] = result;
}

__kernel void gemm_image(const unsigned int m, const unsigned int n, const unsigned int k, __read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c) {
	const unsigned int i = get_local_id(1);  // n
	const unsigned int j = get_local_id(0);  // k
	const unsigned int global_i = BLOCK * get_group_id(1) + i;  // n
	const unsigned int global_j = BLOCK * get_group_id(0) + j;  // k
	__local float local_a[BLOCK][BLOCK];
	__local float local_b[BLOCK][BLOCK];
	const unsigned int num_tiles = m / BLOCK;
	float value = .0;
	for (unsigned int t = 0; t < num_tiles; t++) {
		const unsigned int tiled_i = BLOCK * t + i;
		const unsigned int tiled_j = BLOCK * t + j;
		const int2 idx_a = { global_i, tiled_j };
		const int2 idx_b = { tiled_i, global_j };
		local_a[i][j] = read_imagef(a, idx_a).x;
        local_b[i][j] = read_imagef(b, idx_b).x;
		barrier(CLK_LOCAL_MEM_FENCE);
		for (unsigned int l = 0; l < BLOCK; l++) {
			value += local_a[i][l] * local_b[l][j];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	const int2 idx_c = { global_i, global_j };
    write_imagef(c, idx_c, value);  
}
