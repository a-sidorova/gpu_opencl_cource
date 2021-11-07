__kernel void matmul (const unsigned int m, const unsigned int n, const unsigned int k, __global float* a, __global float* b, __global float* c) {
    const unsigned int i = get_global_id(0);  // m
    const unsigned int j = get_global_id(1);  // k
    c[i * k + j] = .0;
    for (unsigned int l = 0; l < n; l++) {
        c[i * k + j] += a[i * n + l] * b[l * k + j];
    }
}
