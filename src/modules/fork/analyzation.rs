use core::time::Duration;

#[derive(Debug, PartialEq, Eq, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Analyzation {
    pub input_lengths: Vec<usize>,
    pub output_lengths: Vec<usize>,
    pub gpu_dur: Duration,
    pub cpu_dur: Duration,
}

impl PartialOrd for Analyzation {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Analyzation {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // self.cpu_dur.cmp(&other.cpu_dur)
        // self.gpu_dur.cmp(&other.gpu_dur)
        self.input_lengths
            .iter()
            .sum::<usize>()
            .cmp(&other.input_lengths.iter().sum::<usize>())
    }
}
