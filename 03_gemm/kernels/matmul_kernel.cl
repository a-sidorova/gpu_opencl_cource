__kernel void matmul (const unsigned int m, const unsigned int n, const unsigned int k, __global float* a, __global float* b, __global float* c) {
    const unsigned int i = get_global_id(1);  // k
    const unsigned int j = get_global_id(0);  // m
    float result = .0;
    for (unsigned int l = 0; l < n; l++) {
        result += a[i * n + l] * b[l * k + j];
    }
    c[i * k + j] = result;
}
