use std::time::Instant;
use rand;
use rand::random;
use crate::domain::{SequencePair, FloorPlantProblem};
use crate::local_search::{hill_climbing, simulated_annealing};
mod domain;
mod local_search;

fn main() {
    let now = Instant::now();
    let n: i32 = 25;
    let mut min_widths = vec![0; n as usize];
    let mut min_heights = vec![0; n as usize];
    let mut blocks_area = vec![0; n as usize];
    let mut connected_to =  vec![vec![false; n as usize]; n as usize];
    for i in 0..n as usize {
        min_widths[i] = random::<i32>().abs() % (1+(i+1).ilog2() as i32) + 1;
        min_heights[i] = random::<i32>().abs() % (1+(i+1).ilog2() as i32) + 1;
        blocks_area[i] = min_widths[i] * min_heights[i];
        for _ in 0..1+random::<usize>()%(1+(1+i).ilog2() as usize) {
            let mut j = random::<usize>() % n as usize;
            /*while connected_to[i][j] && i != j {
                j = random::<usize>() % n as usize;
            }*/
            connected_to[i][j] = true;
            connected_to[j][i] = true;
        }
    }
    let mut fpp: FloorPlantProblem = FloorPlantProblem {
        n,
        best_sp: SequencePair::new_shuffled(n),
        min_widths,
        min_heights,
        blocks_area,
        connected_to: connected_to.clone()
    };

    simulated_annealing(
        &mut fpp,
        100.0,
        0.1,
        1.0-1e-7,
        0.9
    );

    hill_climbing(
        &mut fpp,
        0.5
    );
    //fpp.sp = fpp.get_random_sp_neighbour();
    //fpp.visualize();
    //fpp.get_base_widths();
    //fpp.get_base_heights();
    let elapsed_time = now.elapsed();
    println!("Elapsed time in us: {}", elapsed_time.as_micros());
    println!("Elapsed time: {}s {}ms {}us",
             elapsed_time.as_secs(), elapsed_time.as_millis()%1000, elapsed_time.as_micros()%1000);

    for i in 0..n as usize {print!("{: >2}|", i)}
    println!();
    for i in 0..n as usize {
        for j in 0..n as usize {
            if j <= i {
                print!("\\\\\\");
                continue;
            }
            match connected_to[i][j]  {
                true => print!("|| "),
                false => print!("O0 ")
            }
        }
        println!();
    }
    println!();
}
