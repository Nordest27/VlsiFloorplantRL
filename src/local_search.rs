use rand::random;
use crate::domain::{FloorPlantProblem, SequencePair};

pub fn simulated_annealing(
    fpp: &mut FloorPlantProblem,
    temp: f32,
    temp_min: f32,
    alpha: f32
) -> i32 {
    assert!(alpha < 1.0 && alpha > 0.0);
    let mut objective = fpp.get_base_half_perimeter();
    let mut best_sp = fpp.sp.clone();
    let mut temp = temp;
    while temp > temp_min {
        let previous_sp = fpp.sp.clone();
        fpp.sp = fpp.get_random_sp_neighbour();
        let new_objective= fpp.get_base_half_perimeter();
        let cost = new_objective - objective;
        if cost < 0 {
            best_sp = fpp.sp.clone();
            objective = new_objective;
            println!("New best floor plan found! {objective}");
            println!("Temp {temp}");
            fpp.visualize();
        }
        else if random::<f32>() < (-cost as f32/temp).exp() {
            fpp.sp = previous_sp;
        }
        temp = temp*alpha;
    }
    fpp.sp = best_sp;
    println!("Simulated annealing solution: {} area", objective);
    fpp.visualize();
    objective
}