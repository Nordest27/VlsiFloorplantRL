use rand;
use rand::random;
use crate::sequence_pair::SequencePair;

mod sequence_pair;

fn main() {
    let n: i32 = 10;
    let sp: SequencePair = SequencePair::new_shuffled(n);
    let mut widths = vec![0; n as usize];
    let mut heights = vec![0; n as usize];
    for i in 0..n as usize {
        widths[i] = random::<i32>().abs()%3+3;
        heights[i] = random::<i32>().abs()%3+3;
    }
    sp.visualize(&widths, &heights);
}
