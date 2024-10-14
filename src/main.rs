use std::time::Instant;
use rand;
use rand::random;
use crate::domain::{SequencePair, FloorPlantProblem};
use crate::local_search::simulated_annealing;
mod domain;
mod local_search;

fn main() {
    let now = Instant::now();
    for _ in 0..1 {
        let n: i32 = 16;
        let mut min_widths = vec![0; n as usize];
        let mut min_heights = vec![0; n as usize];
        let mut blocks_area = vec![0; n as usize];
        for i in 0..n as usize {
            min_widths[i] = random::<i32>().abs() % 1 + 1;
            min_heights[i] = random::<i32>().abs() % 1 + 1;
            blocks_area[i] = min_widths[i] * min_heights[i];
        }
        let mut fpp: FloorPlantProblem = FloorPlantProblem {
            n,
            sp: SequencePair::new_shuffled(n),
            min_widths,
            min_heights,
            blocks_area,
            connected_to: vec![vec![false]]
        };
        fpp.visualize();
        simulated_annealing(
            &mut fpp,
            100.0,
            10e-9,
            0.9999

        );
        //fpp.sp = fpp.get_random_sp_neighbour();
        //fpp.visualize();
        //fpp.get_base_widths();
        //fpp.get_base_heights();
    }
    let elapsed_time = now.elapsed();
    println!("Elapsed time in us: {}", elapsed_time.as_micros());
    println!("Elapsed time: {}s {}ms {}us",
             elapsed_time.as_secs(), elapsed_time.as_millis()%1000, elapsed_time.as_micros()%1000);
}
