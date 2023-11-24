use crate::{Combiner, Eval};

#[macro_export]
macro_rules! derivative {
    ($forward_fn:expr, $x:ident) => {{
        // $(
        //     let mut marker = ['\0'; 32];
        //     marker[0] = '(';
        //     marker[31] = ')';
        //     let add = "+0.00001";

        //     for (marker_char, src_marker_char) in marker[1..30 - add.len()].iter_mut().zip($x.marker) {
        //         *marker_char = src_marker_char;
        //     }

        //     for (marker_char, add_char) in marker[30 - add.len()..].iter_mut().zip(add.chars()) {
        //         *marker_char = add_char;
        //     }
        // );
        let mut marker = ['\0'; 32];
        marker[0] = '(';
        marker[31] = ')';
        let add = "+0.00001";

        for (marker_char, src_marker_char) in marker[1..30 - add.len()].iter_mut().zip($x.marker) {
            *marker_char = src_marker_char;
        }

        for (marker_char, add_char) in marker[30 - add.len()..].iter_mut().zip(add.chars()) {
            *marker_char = add_char;
        }

        let lhs = $forward_fn(Resolve {
            val: $x.val + 0.00001,
            marker,
        });
        let rhs = $forward_fn($x);
        lhs.sub(rhs).div(0.00001)
    }};
}
