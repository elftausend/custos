use std::fmt::Write;

use crate::{libs::opencl::{GenericOCL, CLDevice, KernelOptions, api::OCLError}, Matrix};


pub fn ocl_gemm<T: GenericOCL>(device: CLDevice, lhs: Matrix<T>, rhs: Matrix<T>) -> Result<Matrix<T>, OCLError> {
    let mut src = String::new();

    let m = lhs.dims().1;
    let k = lhs.dims().0;
    let n = rhs.dims().0;

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
    //forever !
    
    let mut float_mw = String::new();
    if mw == 1 {
        write!(&mut float_mw, "{}", T::as_ocl_type_str()).unwrap();
    } else {
        write!(&mut float_mw, "{}{}", T::as_ocl_type_str(), mw).unwrap();
    }

    let mut float_kw = String::new();
    if kw == 1 {
        write!(&mut float_kw, "{}", T::as_ocl_type_str()).unwrap();
    } else {
        write!(&mut float_kw, "{}{}", T::as_ocl_type_str(), kw).unwrap();
    }
    write!(&mut src, r#"
        #define K {0}
        #define N {1}
        #define MW {2}     // M tile Width
        #define NW {3}     // N tile Width  -- NW & KW should be the same !
        #define KW {4}     // K tile Width
        #define MT {5}  // MT is max for 'mt' (M tile count)
        #define KT {6}  // KT is max for 'kt' (K tile count)
        #define floatMW {7}
        #define floatKW {8}
        __kernel void GeMM(const __global floatMW* restrict A, const __global floatKW* restrict B, __global floatMW* C)
            {{
                size_t mt = get_global_id(0);    //global M-tile id
                size_t nc = get_global_id(1);    //global N-tile id

                float AT[KW][MW]; // sub tiles
                float BT[NW][KW];
                float CT[NW][MW];

                #pragma unroll
                for (uint i=0; i<NW*MW; ++i) // zero CT tile
                    ((float*) CT)[i] = 0.0;

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
            }}
    "#, k, n, mw, nw, kw, mt, kt, float_mw, float_kw).unwrap();

    let gws = [f, s, 0];
    
    KernelOptions::new(device, lhs, gws, &src)
        .with_rhs(rhs)
        .with_output((rhs.dims().0, lhs.dims().1))
        .run()

    
}