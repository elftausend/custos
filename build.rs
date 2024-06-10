fn main() {
    println!("cargo:rustc-check-cfg=cfg(unified_cl)");
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    #[cfg(not(docsrs))]
    #[cfg(feature = "opencl")]
    if has_device_unified_mem() {
        println!("cargo:rustc-cfg=unified_cl");
    }

    #[cfg(not(docsrs))]
    #[cfg(feature = "opencl")]
    cl_check_kernel_exec();

    {
        #[cfg(target_os = "macos")]
        {
            // check if debug or release
            if Ok("debug".into()) == std::env::var("PROFILE") {
                std::env::set_var("CUSTOS_CUDA_LINK_ON_BUILD", "false");
            }
        }

        #[cfg(not(docsrs))]
        #[cfg(feature = "cuda")]
        if check_cuda_link() {
            link_cuda();
        }
    }
}

#[cfg(not(docsrs))]
#[cfg(feature = "opencl")]
fn cl_check_kernel_exec() {
    use min_cl::CLDevice;

    println!("cargo:rerun-if-env-changed=CUSTOS_CL_KERNEL_EXEC_ON_BUILD");

    let run_cl_check = std::env::var("CUSTOS_CL_KERNEL_EXEC_ON_BUILD")
        .unwrap_or_else(|_| "false".into())
        .parse::<bool>()
        .expect("CUSTOS_CL_KERNEL_EXEC_ON_BUILD must be either true or false");

    if run_cl_check {
        // Runs a simple kernel to measure performance and functionality in general
        CLDevice::fastest().unwrap();
    }
}

#[cfg(not(docsrs))]
#[cfg(feature = "opencl")]
fn has_device_unified_mem() -> bool {
    println!("cargo:rerun-if-env-changed=CUSTOS_CL_DEVICE_IDX");
    println!("cargo:rerun-if-env-changed=CUSTOS_CU_DEVICE_IDX");
    println!("cargo:rerun-if-env-changed=CUSTOS_USE_UNIFIED");

    let device_idx = std::env::var("CUSTOS_CL_DEVICE_IDX")
        .unwrap_or_else(|_| "0".into())
        .parse::<usize>()
        .expect("Value in variable 'CUSTOS_CL_DEVICE_IDX' must be a usize value.");

    // this environment variable (CUSTOS_USE_UNIFIED) is used to either:
    // ... disable unified memory on unified memory devices, or
    // ... activate unified memory on devices with dedicated memory to check if
    // the code would compile on a device with unified memory.
    if let Ok(value) = std::env::var("CUSTOS_USE_UNIFIED") {
        if &value.to_ascii_lowercase() != "default" {
            let force_unified_mem = value.parse()
                .expect("'CUSTOS_USE_UNIFIED' must be either true, false or default. 
                    [
                        default=it is checked whether the device can use unified memory automatically.
                        true='simulates' unified memory to know if your code would compile on a device with unified memory.
                        false=deactivates unified memory
                    ]");
            if force_unified_mem {
                println!("Device forcefully uses unified memory!")
            } else {
                println!("Device won't use unified memory!")
            }
            return force_unified_mem;
        }
    }

    min_cl::CLDevice::new(device_idx)
        .unwrap_or_else(|_| panic!("Could not get an OpenCL device (at index {device_idx}). Set `CUSTOS_CL_DEVICE_IDX` to a valid device index."))
        .unified_mem
}

#[cfg(feature = "cuda")]
use std::path::{Path, PathBuf};

#[cfg(feature = "cuda")]
fn check_cuda_link() -> bool {
    println!("cargo:rerun-if-env-changed=CUSTOS_CUDA_LINK_ON_BUILD");
    std::env::var("CUSTOS_CUDA_LINK_ON_BUILD")
        .unwrap_or_else(|_| "true".into())
        .parse::<bool>()
        .expect("CUSTOS_CUDA_LINK_ON_BUILD must be either true or false")
}

// https://github.com/coreylowman/cudarc/blob/main/build.rs
#[cfg(feature = "cuda")]
fn link_cuda() {
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");

    let candidates: Vec<PathBuf> = root_candidates().collect();

    let toolkit_root = root_candidates()
        .find(|path| path.join("include").join("cuda.h").is_file())
        .unwrap_or_else(|| {
            panic!(
                "Unable to find `include/cuda.h` under any of: {candidates:?}. Set the `CUDA_ROOT` environment variable to `$CUDA_ROOT/include/cuda.h` to override path.",
            )
        });

    for path in lib_candidates(&toolkit_root) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=nvrtc");
    println!("cargo:rustc-link-lib=dylib=curand");

    println!("cargo:rustc-link-lib=dylib=cublas");
}

#[cfg(feature = "cuda")]
fn root_candidates() -> impl Iterator<Item = PathBuf> {
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok);

    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];
    let roots = roots.into_iter().map(Into::into);
    env_vars.chain(roots).map(Into::<PathBuf>::into)
}

#[cfg(feature = "cuda")]
fn lib_candidates(root: &Path) -> Vec<PathBuf> {
    [
        "lib",
        "lib/x64",
        "lib/Win32",
        "lib/x86_64",
        "lib/x86_64-linux-gnu",
        "lib64",
        "lib64/stubs",
        "targets/x86_64-linux",
        "targets/x86_64-linux/lib",
        "targets/x86_64-linux/lib/stubs",
    ]
    .iter()
    .map(|&p| root.join(p))
    .filter(|p| p.is_dir())
    .collect()
}
