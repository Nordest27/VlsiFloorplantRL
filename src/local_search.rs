use std::f32::EPSILON;
use rand::random;
use crate::domain::{FloorPlantProblem, SequencePair};

pub fn simulated_annealing(
    fpp: &mut FloorPlantProblem,
    temp: f32,
    temp_min: f32,
    alpha: f32,
    area_importance: f32
) -> f32 {
    println!("Alpha {}", alpha);
    assert!(alpha < 1.0 && alpha > 0.0);
    let mut best_objective = fpp.get_wire_length_estimate_and_area(&fpp.best_sp);
    let initial_objective = best_objective.clone();
    println!("Initial solution: wire length: {}, area {}",
             best_objective.0, best_objective.1);
    fpp.visualize(&fpp.best_sp);
    let mut temp = temp;
    let print_temp_interval = temp*0.1;
    let mut print_temp = temp - print_temp_interval;
    let mut current_sp: SequencePair = fpp.best_sp.clone();
    let mut evaluations = 0;
    while temp > temp_min {
        evaluations += 1;
        let (sp, new_objective) = fpp.get_random_sp_neighbour_with_obj(&current_sp);
        let cost = (1.0-area_importance)*new_objective.0 + area_importance * new_objective.1
                   - (1.0-area_importance)* best_objective.0 - area_importance* best_objective.1;
        if cost < -0.05 {
            current_sp = sp;
            fpp.best_sp = current_sp.clone();
            best_objective = new_objective;
            println!("New best floor plan found! wire length: {}, area {}",
                     best_objective.0, best_objective.1);
            println!("Temp {temp}, cost: {cost}");
            fpp.visualize(&fpp.best_sp);
        }
        else if random::<f32>() < (-cost/temp).exp() {
            current_sp = sp;
        }
        else if cost > 0.25*(best_objective.0 + best_objective.1) {
            current_sp = fpp.best_sp.clone();
        }

        if temp < print_temp {
            println!("Current temp: {temp}, evaluations: {evaluations}");
            print_temp = temp - print_temp_interval;
        }
        /*
        if cost > 0.0 {
            println!("temp {temp}, cost {cost}, (-cost/temp).exp() {},", (-cost/temp).exp());
        }*/
        temp = temp*alpha;
    }
    println!("Simulated annealing solution: wire length: {}, area {}",
             best_objective.0, best_objective.1);
    println!("Improved from initial_objective: wire length: {}, area {}",
            initial_objective.0, initial_objective.1);
    println!("Total evaluations: {evaluations}");
    fpp.visualize(&fpp.best_sp);
    best_objective.0 + best_objective.1
}

pub fn hill_climbing(
    fpp: &mut FloorPlantProblem,
    area_importance: f32
) -> f32 {
    let mut best_objective = fpp.get_wire_length_estimate_and_area(&fpp.best_sp);
    let initial_objective = best_objective.clone();
    println!("Initial solution: wire length: {}, area {}",
             best_objective.0, best_objective.1);
    fpp.visualize(&fpp.best_sp);
    let mut evaluations = 0;
    loop {
        let mut best_sp = fpp.best_sp.clone();
        let mut better_found = false;
        for (sp, new_objective) in fpp.get_all_sp_neighbours_with_obj(&fpp.best_sp) {
            evaluations += 1;
            let cost = (1.0 - area_importance) * new_objective.0 + area_importance * new_objective.1
                - (1.0 - area_importance) * best_objective.0 - area_importance * best_objective.1;
            if cost < 0.0 {
                best_sp = sp;
                fpp.best_sp = best_sp.clone();
                println!("new obj: {} + {} = {}, best obj: {} + {} = {}",
                         new_objective.0, new_objective.1,
                         new_objective.0 + new_objective.1,
                         best_objective.0, best_objective.1,
                         best_objective.0 + best_objective.1
                );
                best_objective = new_objective;
                println!("New best floor plan found! wire length: {}, area {}",
                         best_objective.0, best_objective.1);
                println!("Cost: {cost}, evaluations: {evaluations}");
                //fpp.visualize(&fpp.best_sp);
                better_found = true;
            }
        }
        if !better_found { break }
    }
    println!("Hill climbing solution: wire length: {}, area {}",
             best_objective.0, best_objective.1);
    println!("Improved from initial_objective: wire length: {}, area {}",
             initial_objective.0, initial_objective.1);
    println!("Total evaluations: {evaluations}");
    fpp.visualize(&fpp.best_sp);
    best_objective.0 + best_objective.1
}