use custos::range;

#[test]
fn test_range() {
    let mut count = 0;
    for epoch in range(10) {
        assert_eq!(epoch, count);
        count += 1;
    }
}

#[test]
fn test_range1() {
    let mut count = 0;
    for epoch in range(0..10) {
        assert_eq!(epoch, count);
        count += 1;
        assert!(epoch < 10)
    }
}

#[test]
fn test_range_inclusive() {
    let mut count = 0;
    for epoch in range(0..=10) {
        assert_eq!(epoch, count);
        count += 1;
        assert!(epoch < 11)
    }
}
