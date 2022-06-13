extern "C" __global__ void add(int *a, int *b, int *c)
{
    int idx = blockIdx.x;
    c[idx] = a[idx] + b[idx];
}