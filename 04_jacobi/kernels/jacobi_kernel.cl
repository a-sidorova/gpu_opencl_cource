__kernel void jacobi(__global float *A, __global float *b, __global float *x0, 
                     __global float *x1, __global float *norm, unsigned int size) {
    const unsigned int ithr = get_global_id(0);
    if (ithr >= size)
        return;

    float sum = .0f;
    for (unsigned int j = 0; j < size; j++) {
        sum += A[j * size + ithr] * x0[j] * (float)(ithr != j);
    }

    x1[ithr] = (b[ithr] - sum) / A[ithr * size + ithr];
    norm[ithr] = x1[ithr] - x0[ithr];
}