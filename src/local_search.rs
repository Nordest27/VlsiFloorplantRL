use rand::random;
use crate::domain::{FloorPlantProblem};

pub fn simulated_annealing(
    fpp: &mut FloorPlantProblem,
    temp: f32,
    temp_min: f32,
    alpha: f32
) -> f32 {
    assert!(alpha < 1.0 && alpha > 0.0);
    let mut objective = fpp.get_wire_length_estimate_and_area();
    println!("Initial solution: wire length: {}, area {}",
             objective.0, objective.1);
    let mut best_sp = fpp.sp.clone();
    let mut temp = temp;
    while temp > temp_min {
        let previous_sp = fpp.sp.clone();
        fpp.sp = fpp.get_random_sp_neighbour();
        let new_objective= fpp.get_wire_length_estimate_and_area();
        let cost = new_objective.0 + new_objective.1 - objective.0 - objective.1;
        if cost < 0.0 {
            best_sp = fpp.sp.clone();
            objective = new_objective;
            println!("New best floor plan found! wire length: {}, area {}",
                     objective.0, objective.1);
            println!("Temp {temp}");
            fpp.visualize();
        }
        else if random::<f32>() < (-cost/temp).exp() {
            fpp.sp = previous_sp;
        }
        temp = temp*alpha;
    }
    fpp.sp = best_sp;
    println!("Simulated annealing solution: wire length: {}, area {}",
             objective.0, objective.1);
    fpp.visualize();
    objective.0 + objective.1
}