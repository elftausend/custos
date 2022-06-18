use std::fmt::Write;
use crate::{libs::opencl::{KernelOptions, cl_device::InternCLDevice}, Error, CDatatype, Buffer};

pub fn cl_gemm<T: CDatatype>(device: InternCLDevice, m: usize, k: usize, n: usize, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Result<Buffer<T>, Error> {
    let mut mw = 1;
    for x in &[16, 8, 4, 2, 1] {
        if m % x == 0 {
            mw = *x;
            break;
        }
    }
    let mut kw = 1;
    for x in &[8, 4, 2, 1] {
        if n % x == 0 && k % x == 0 {
            kw = *x;
            break;
        }
    }
    let nw = kw;
    let mt = (((m/mw) as f32).floor()) as usize;
    let kt = (((k/kw) as f32).floor()) as usize;

    let f = (((m/mw) as f32).floor()) as usize;
    let s = (((n/nw) as f32).floor()) as usize;
    //'testing'/excellent code for gemm - 'currently' stolen from litenn

    let mut float_mw = String::new();
    if mw == 1 {
        write!(&mut float_mw, "{}", T::as_c_type_str()).unwrap();
    } else {
        write!(&mut float_mw, "{}{}", T::as_c_type_str(), mw).unwrap();
    }

    let mut float_kw = String::new();
    if kw == 1 {
        write!(&mut float_kw, "{}", T::as_c_type_str()).unwrap();
    } else {
        write!(&mut float_kw, "{}{}", T::as_c_type_str(), kw).unwrap();
    }

    let dt = T::as_c_type_str();

    let src = format!("
        #define K {k}
        #define N {n}
        #define MW {mw}     // M tile Width
        #define NW {nw}     // N tile Width  -- NW & KW should be the same !
        #define KW {kw}     // K tile Width
        #define MT {mt}  // MT is max for 'mt' (M tile count)
        #define KT {kt}  // KT is max for 'kt' (K tile count)
        #define floatMW {float_mw}
        #define floatKW {float_kw}
        __kernel void GeMM(const __global floatMW* restrict A, const __global floatKW* restrict B, __global floatMW* C)
            {{
                size_t mt = get_global_id(0);    //global M-tile id
                size_t nc = get_global_id(1);    //global N-tile id

                {dt} AT[KW][MW]; // sub tiles
                {dt} BT[NW][KW];
                {dt} CT[NW][MW];

                #pragma unroll
                for (uint i=0; i<NW*MW; ++i) // zero CT tile
                    (({dt }*) CT)[i] = 0.0;

                for (uint kt=0; kt<KT; ++kt)  // iterate over K-dim tiles
                {{
                    #pragma unroll
                    for (uint k=0; k<KW; ++k)  // every k-element inside K-dim tile
                        *( (floatMW*) AT[k] ) = A[(kt*KW + k)*MT + mt]; // store M-Width floats

                    #pragma unroll
                    for (uint n=0; n<NW; ++n)  // every n-element inside N-dim tile
                        *( (floatKW*) BT[n] ) = B[(nc*NW + n)*KT + kt]; // store K-Width floats

                    #pragma unroll
                    for (uint k=0; k<KW; ++k)
                    #pragma unroll
                    for (uint n=0; n<NW; ++n)  // sub tiles multiplication
                    #pragma unroll
                    for (uint m=0; m<MW; ++m)
                        CT[n][m] += AT[k][m] * BT[n][k];
                }}

                #pragma unroll
                for (uint n=0; n<NW; ++n)
                    C[(nc*NW + n)*MT + mt] = *( (floatMW*) CT[n]);
            }}");

    let gws = [f, s, 0];
    
    KernelOptions::new(&device, lhs, gws, &src)?
        .with_rhs(rhs)
        .with_output(n*m)
        .run()    
}


