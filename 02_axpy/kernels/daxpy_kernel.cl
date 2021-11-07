__kernel void daxpy(int n, double a, __global double *x, int incx, __global double *y, int incy) {
    size_t i = get_global_id(0);
    if (i < n) {
        y[i * incy] += a * x[i * incx];
    }
}