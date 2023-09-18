use nnapi::nnapi_sys;

pub trait AsOperandCode {
    const OPERAND_CODE: nnapi_sys::OperandCode;
}

impl AsOperandCode for f32 {
    const OPERAND_CODE: nnapi_sys::OperandCode =
        nnapi_sys::OperandCode::ANEURALNETWORKS_TENSOR_FLOAT32;
}

impl AsOperandCode for i32 {
    const OPERAND_CODE: nnapi_sys::OperandCode =
        nnapi_sys::OperandCode::ANEURALNETWORKS_TENSOR_INT32;
}

/// not useable for tensors!
impl AsOperandCode for u32 {
    const OPERAND_CODE: nnapi_sys::OperandCode = nnapi_sys::OperandCode::ANEURALNETWORKS_UINT32;
}

impl AsOperandCode for bool {
    const OPERAND_CODE: nnapi_sys::OperandCode =
        nnapi_sys::OperandCode::ANEURALNETWORKS_TENSOR_BOOL8;
}

#[cfg(feature = "half")]
impl AsOperandCode for half::f16 {
    const OPERAND_CODE: nnapi_sys::OperandCode =
        nnapi_sys::OperandCode::ANEURALNETWORKS_TENSOR_FLOAT16;
}
