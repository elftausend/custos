
struct Count(usize, usize);

struct CountIntoIter(usize);

impl IntoIterator for Count {
    type Item = usize;

    type IntoIter = CountIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

#[test]
fn counting() {
    for i in Count(0, 100) {}

}