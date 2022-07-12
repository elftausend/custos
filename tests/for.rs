use custos::range;

#[test]
fn test_range() {
    let mut count = 0;
    for epoch in range(10) {
        assert_eq!(epoch, count);
        count += 1;
    }
}
