use std::time::Instant;
use rand;
use rand::random;
use crate::domain::{SequencePair, FloorPlantProblem};
use crate::local_search::{hill_climbing, monte_carlo_estimation_search, simulated_annealing};
mod domain;
mod local_search;

fn main() {
    let now = Instant::now();
    let n: i32 = 64;
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
    let mut fpp_clone = fpp.clone();

    let fpp_to_visualize = FloorPlantProblem {
        n: 8,
        best_sp: SequencePair {
            y: vec![2, 7, 5, 3, 6, 4, 1, 0],
            x: vec![0, 4, 1, 7, 5, 3, 6, 2]
        },
        min_widths: vec![4, 3, 3, 3, 3, 3, 1, 2],
        min_heights: vec![2, 1, 3, 5, 2, 5, 2, 4],
        blocks_area: vec![8, 3, 9, 15, 6, 15, 2, 8],
        connected_to: vec![vec![false; 8]; 8]

    };
    fpp_to_visualize.visualize(&fpp_to_visualize.best_sp);
    simulated_annealing(
        &mut fpp,
        100.0,
        0.1,
        1.0-1e-6,
        0.5,
        false
    );
    return;
    /*
    hill_climbing(
        &mut fpp,
        0.5
    );
    */
    //fpp.sp = fpp.get_random_sp_neighbour();
    //fpp.visualize();
    //fpp.get_base_widths();
    //fpp.get_base_heights();
    let result = monte_carlo_estimation_search(&fpp_clone, 100, 5, -1.0);
    let result = monte_carlo_estimation_search(&fpp_clone, 1000, 5, -1.0);
    let result = monte_carlo_estimation_search(&fpp_clone, 10000, 5, -1.0);
    let result = monte_carlo_estimation_search(&fpp_clone, 100000, 5, -1.0);

    /*
    let mut should_see = -1.0;
    for _ in n..10000 {
        let obj = fpp_clone.get_wire_length_estimate_and_area(&fpp_clone.best_sp);
        println!("Current obj :{}", obj.0+obj.1);
        let result = monte_carlo_estimation_search(
            &fpp_clone,
            10000,
            5,
            should_see
        );
        if result.0.len() == 0 {break}
        fpp_clone.best_sp = result.1;
        fpp_clone.visualize(&fpp_clone.best_sp);
        should_see = result.2

    }

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
    */
}
