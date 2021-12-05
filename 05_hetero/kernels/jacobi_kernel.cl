__kernel void jacobi(__global float *A, __global float *b, __global float *x0, 
                     __global float *x1, __global float *norm, unsigned int size, unsigned int stride) {
    const unsigned int ithr = get_global_id(0);
    const unsigned int global_stride = stride + ithr;
    if (global_stride >= size)
        return;

    float sum = .0f;
    for (unsigned int j = 0; j < size; j++) {
        sum += A[j * size + global_stride] * x0[j] * (float)(global_stride != j);
    }

    x1[ithr] = (b[ithr] - sum) / A[global_stride * size + global_stride];
    norm[ithr] = x1[ithr] - x0[global_stride];
}
