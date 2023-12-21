use custos::{nnapi::NnapiDevice, Base, Buffer, Device, Lazy, WithShape, Retriever, Dim1};
use ndk_sys::android_LogPriority;
use nnapi::{nnapi_sys::OperationCode, Operand};

use crate::log;

pub fn run_custos_model() -> custos::Result<String> {
    let device = NnapiDevice::<i32, Lazy<Base>>::new()?;

    let lhs = Buffer::with(&device, [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]);
    let rhs = Buffer::with(&device, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    /*
    // a single operation
    let out = {
        let out = device.retrieve::<i32, Dim1<10>>(lhs.len(), ());
        let activation_idx = device.add_operand(&Operand::activation())?;
        let mut model = device.model.borrow_mut();
        model.set_activation_operand_value(activation_idx as i32)?;
        model.add_operation(
            OperationCode::ANEURALNETWORKS_ADD,
            &[lhs.ptr.idx, rhs.ptr.idx, activation_idx],
            &[out.ptr.idx],
        )?;
        out
    };


    let out = {
        let out1 = device.retrieve::<i32, Dim1<10>>(lhs.len(), ());
        let activation_idx = device.add_operand(&Operand::activation())?;
        let mut model = device.model.borrow_mut();
        model.set_activation_operand_value(activation_idx as i32)?;
        model.add_operation(
            OperationCode::ANEURALNETWORKS_MUL,
            &[out.ptr.idx, rhs.ptr.idx, activation_idx],
            &[out1.ptr.idx],
        )?;
        out1
    };*/

    let out: Buffer<i32, _, Dim1<10>> = device.retrieve(lhs.len(), ());

    {
        let activation_idx = device.add_operand(&custos::nnapi::Operand::activation()).unwrap();
        let mut model = device.model.borrow_mut();

        model
            .set_activation_operand_value(activation_idx as i32)
            .unwrap();
        model
            .add_operation(
                OperationCode::ANEURALNETWORKS_ADD,
                &[lhs.base().idx, rhs.base().idx, activation_idx],
                &[out.base().idx],
            )
            .unwrap();
    }
    let out1: Buffer<i32, _, Dim1<10>> = device.retrieve(lhs.len(), ());

    {
        let activation_idx = device.add_operand(&custos::nnapi::Operand::activation()).unwrap();
        let mut model = device.model.borrow_mut();
        model
            .set_activation_operand_value(activation_idx as i32)
            .unwrap();

        model
            .add_operation(
                OperationCode::ANEURALNETWORKS_MUL,
                &[out.base().idx, rhs.base().idx, activation_idx],
                &[out1.base().idx],
            )
            .unwrap();
    }
    for _ in 0..100 {
        let out = device.run_with_vec()?;
        log(android_LogPriority::ANDROID_LOG_DEBUG, &format!("{out:?}"));
        assert_eq!(out, &[2, 8, 18, 32, 50, 72, 98, 128, 162, 210]);
    }

    // let out = device.run(out)?;
    // log(android_LogPriority::ANDROID_LOG_ERROR, &format!("{out:?}"));

    // Ok(format!("out: {out:?}"))
    let out = device.run_with_vec()?;
    Ok(format!("{out:?}"))
}
