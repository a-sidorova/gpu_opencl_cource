__kernel void saxpy(int n, float a, __global float *x, int incx, __global float *y, int incy) {
    size_t i = get_global_id(0);
    if (i < n) {
        y[i * incy] += a * x[i * incx];
    }
}
