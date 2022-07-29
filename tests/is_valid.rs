use custos::{BufFlag, is_buf_valid};

#[test]
fn test_is_buf_valid() {
    let flag = BufFlag::None;
    assert!(is_buf_valid(&flag));

    let flag = BufFlag::Wrapper;
    assert!(is_buf_valid(&flag));

    let bool = true;
    let flag = BufFlag::Cache(&bool as *const bool);
    assert!(is_buf_valid(&flag));
}