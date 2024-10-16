use rand::random;
use crate::domain::{FloorPlantProblem, SequencePair};

pub fn simulated_annealing(
    fpp: &mut FloorPlantProblem,
    temp: f32,
    temp_min: f32,
    alpha: f32
) -> f32 {
    println!("Alpha {}", alpha);
    assert!(alpha < 1.0 && alpha > 0.0);
    let mut objective = fpp.get_wire_length_estimate_and_area(&fpp.best_sp);
    println!("Initial solution: wire length: {}, area {}",
             objective.0, objective.1);
    fpp.visualize(&fpp.best_sp);
    let mut temp = temp;
    let mut current_sp: SequencePair = fpp.best_sp.clone();
    while temp > temp_min {
        let (sp, new_objective) = fpp.get_random_sp_neighbour_with_obj(&current_sp);
        let cost = new_objective.0 + new_objective.1 - objective.0 - objective.1;
        if cost < 0.0 {
            current_sp = sp;
            fpp.best_sp = current_sp.clone();
            objective = new_objective;
            println!("New best floor plan found! wire length: {}, area {}",
                     objective.0, objective.1);
            println!("Temp {temp}");
            fpp.visualize(&fpp.best_sp);
        }
        else if random::<f32>() < (-cost/temp).exp() {
            current_sp = sp;
        }
        else if cost > 20.0 {
            current_sp = fpp.best_sp.clone();
        }
        /*
        if cost > 0.0 {
            println!("temp {temp}, cost {cost}, (-cost/temp).exp() {},", (-cost/temp).exp());
        }*/
        temp = temp*alpha;
    }
    println!("Simulated annealing solution: wire length: {}, area {}",
             objective.0, objective.1);
    fpp.visualize(&fpp.best_sp);
    objective.0 + objective.1
}