// mod using_sliced;
mod using_custos_nnapi;
mod using_nnapi_rs;

use std::ffi::{c_void, CStr, CString};
use std::mem::size_of_val;
use std::os::raw::c_char;
use std::ptr::null_mut;

// linking one time -> all other fns are found too
/*#[link(name = "neuralnetworks")]
extern "C" {
    pub fn ANeuralNetworksModel_create(
        model: *mut *mut ANeuralNetworksModel,
    ) -> ::std::os::raw::c_int;
}*/

use custos::{Base, Device};
use ndk_sys::{__android_log_print, android_LogPriority, ANeuralNetworks_getRuntimeFeatureLevel};
use nnapi::nnapi_sys::{
    ANeuralNetworksCompilation, ANeuralNetworksCompilation_create,
    ANeuralNetworksCompilation_finish, ANeuralNetworksCompilation_free, ANeuralNetworksEvent,
    ANeuralNetworksEvent_free, ANeuralNetworksEvent_wait, ANeuralNetworksExecution,
    ANeuralNetworksExecution_create, ANeuralNetworksExecution_free,
    ANeuralNetworksExecution_setInput, ANeuralNetworksExecution_setOutput,
    ANeuralNetworksExecution_startCompute, ANeuralNetworksModel, ANeuralNetworksModel_addOperand,
    ANeuralNetworksModel_addOperation, ANeuralNetworksModel_create, ANeuralNetworksModel_finish,
    ANeuralNetworksModel_free, ANeuralNetworksModel_identifyInputsAndOutputs,
    ANeuralNetworksModel_setOperandValue, ANeuralNetworksOperandType, OperandCode, OperationCode,
    ANEURALNETWORKS_FUSED_NONE,
};
use using_custos_nnapi::run_custos_model;
use using_nnapi_rs::run_nnapi_rs_model;

pub fn log(priority: android_LogPriority, msg: &str) {
    let tag = CString::new("MyApp").unwrap();
    let msg = CString::new(msg).unwrap();
    unsafe { __android_log_print(priority.0 as i32, tag.as_ptr(), msg.as_ptr()) };
}

pub fn inner_rust_greeting(to: &str) -> String {
    let device = custos::CPU::<Base>::new();
    // let lhs = device.buffer([1, 2, 3, 4, 5]);
    // let rhs = device.buffer([1, 2, 3, 4, 5]);

    let tensor9x_type = ANeuralNetworksOperandType {
        type_: OperandCode::ANEURALNETWORKS_TENSOR_FLOAT32 as i32,
        dimensionCount: 1,
        dimensions: [9].as_ptr(),
        scale: 0.,
        zeroPoint: 0,
    };

    let activation_type = ANeuralNetworksOperandType {
        type_: OperandCode::ANEURALNETWORKS_INT32 as i32,
        dimensionCount: 0,
        dimensions: null_mut(),
        scale: 0.,
        zeroPoint: 0,
    };

    let mut model: *mut ANeuralNetworksModel = null_mut();

    // unsafe { AHardwareBuffer_allocate(null_mut(), null_mut()) };

    let version = unsafe { ANeuralNetworks_getRuntimeFeatureLevel() };

    if unsafe { ANeuralNetworksModel_create(&mut model) } != 0 {
        panic!("Failed to create model");
    }

    if unsafe { ANeuralNetworksModel_addOperand(model, &tensor9x_type) } != 0 {
        panic!("Failed to add operand 0");
    }
    unsafe { ANeuralNetworksModel_addOperand(model, &tensor9x_type) };
    unsafe { ANeuralNetworksModel_addOperand(model, &activation_type) };
    unsafe { ANeuralNetworksModel_addOperand(model, &tensor9x_type) };

    let lhs_idx = 0;
    let rhs_idx = 1;
    let activation_idx = 2;
    let out_idx = 3;

    let outputs = [out_idx];

    let none_value = ANEURALNETWORKS_FUSED_NONE;

    // set the activation operand to a none value (fixes "Graph contains at least one cycle or one never-written operand")
    if 0 != unsafe {
        ANeuralNetworksModel_setOperandValue(
            model,
            activation_idx as i32,
            &none_value as *const _ as *const _,
            4,
        )
    } {
        log(
            android_LogPriority::ANDROID_LOG_ERROR,
            "Failed to set activation operand value!",
        );
        panic!("Failed to set activation operand value!");
    }

    if 0 != unsafe {
        // For whatever reason an activation is needed
        let inputs = [lhs_idx, rhs_idx, activation_idx];
        ANeuralNetworksModel_addOperation(
            model,
            OperationCode::ANEURALNETWORKS_ADD as i32,
            3, // inputs.len().try_into().unwrap(),
            inputs.as_ptr(),
            1, // outputs.len().try_into().unwrap(),
            outputs.as_ptr(),
        )
    } {
        panic!("Failed to add add operation!");
    }
    unsafe {
        let inputs = [lhs_idx, rhs_idx];
        ANeuralNetworksModel_identifyInputsAndOutputs(
            model,
            inputs.len().try_into().unwrap(),
            inputs.as_ptr(),
            outputs.len().try_into().unwrap(),
            outputs.as_ptr(),
        )
    };

    unsafe { ANeuralNetworksModel_finish(model) };

    let mut compilation: *mut ANeuralNetworksCompilation = null_mut();
    unsafe { ANeuralNetworksCompilation_create(model, &mut compilation) };

    unsafe { ANeuralNetworksCompilation_finish(compilation) };

    let mut run1: *mut ANeuralNetworksExecution = null_mut();
    unsafe { ANeuralNetworksExecution_create(compilation, &mut run1) };

    let lhs = [1f32, 2., 3., 4., 5., 6., 7., 8., 9.];
    let rhs = [1f32, 2., 3., 4., 5., 6., 7., 8., 9.];

    unsafe {
        ANeuralNetworksExecution_setInput(
            run1,
            0,
            null_mut(),
            lhs.as_ptr() as *const c_void,
            (lhs.len() * std::mem::size_of::<f32>()).try_into().unwrap(),
        )
    };
    unsafe {
        ANeuralNetworksExecution_setInput(
            run1,
            1,
            null_mut(),
            rhs.as_ptr() as *const c_void,
            (rhs.len() * std::mem::size_of::<f32>()).try_into().unwrap(),
        )
    };

    let mut out = [0f32; 9];

    unsafe {
        ANeuralNetworksExecution_setOutput(
            run1,
            0,
            null_mut(),
            out.as_mut_ptr() as *mut c_void,
            (out.len() * std::mem::size_of::<f32>()).try_into().unwrap(),
        )
    };

    let mut run1_end: *mut ANeuralNetworksEvent = null_mut();
    unsafe { ANeuralNetworksExecution_startCompute(run1, &mut run1_end) };

    unsafe { ANeuralNetworksEvent_wait(run1_end) };
    unsafe { ANeuralNetworksEvent_free(run1_end) };
    unsafe { ANeuralNetworksExecution_free(run1) };

    unsafe { ANeuralNetworksCompilation_free(compilation) };
    unsafe { ANeuralNetworksModel_free(model) };

    // format!("SDK Version: {version}, Rust community: Hello {to}")
    format!("{out:?}")
}

/// # Safety
#[no_mangle]
pub unsafe extern "C" fn rust_run_model(to: *const c_char) -> *mut c_char {
    let c_str = CStr::from_ptr(to);
    let recipient = match c_str.to_str() {
        Err(_) => "there",
        Ok(string) => string,
    };

    /*CString::new(inner_rust_greeting(recipient))
    .unwrap()
    .into_raw()*/
    CString::new(run_custos_model().unwrap())
        .unwrap()
        .into_raw()
}

/// # Safety
#[no_mangle]
pub unsafe extern "C" fn rust_string_free(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    let _ = CString::from_raw(s);
}

#[cfg(target_os = "android")]
#[allow(non_snake_case)]
pub mod android {
    extern crate jni;

    use self::jni::objects::{JClass, JString};
    use self::jni::sys::jstring;
    use self::jni::JNIEnv;
    use super::*;

    #[no_mangle]
    pub unsafe extern "C" fn Java_com_example_newrusttest_RustBindings_run(
        env: JNIEnv,
        _: JClass,
        java_pattern: JString,
    ) -> jstring {
        // Our Java companion code might pass-in "world" as a string, hence the name.
        let world = rust_run_model(
            env.get_string(java_pattern)
                .expect("invalid pattern string")
                .as_ptr(),
        );
        // Retake pointer so that we can use it below and allow memory to be freed when it goes out of scope.
        let output = env
            .new_string(CStr::from_ptr(world).to_str().unwrap())
            .expect("Couldn't create java string!");
        rust_string_free(world);

        output.into_inner()
    }
}
