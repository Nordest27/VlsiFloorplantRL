use rand::random;
use crate::domain::{FloorPlantProblem, SequencePair};

pub fn simulated_annealing(
    fpp: &mut FloorPlantProblem,
    temp: f32,
    temp_min: f32,
    alpha: f32,
    area_importance: f32,
    enshitify: bool
) -> f32 {
    //println!("Alpha {}", alpha);
    assert!(alpha < 1.0 && alpha > 0.0);
    let mut best_objective = fpp.get_wire_length_estimate_and_area(&fpp.best_sp);
    
    let initial_objective = best_objective.clone();
    //println!("Initial solution: wire length: {}, area {}",
    //         best_objective.0, best_objective.1);
    //fpp.visualize(&fpp.best_sp);
    let mut temp = temp;
    let print_temp_interval = temp*0.1;
    let mut print_temp = temp - print_temp_interval;
    let mut current_sp: SequencePair = fpp.best_sp.clone();
    let mut evaluations = 0;
    while temp > temp_min {
        evaluations += 1;
        let (sp, new_objective) = fpp.get_random_sp_neighbour_with_obj(&current_sp, false);
        let cost = (1.0-area_importance)*new_objective.0 + area_importance * new_objective.1
                   - (1.0-area_importance)* best_objective.0 - area_importance* best_objective.1;
        if cost < -0.05 {
            current_sp = sp;
            fpp.best_sp = current_sp.clone();
            best_objective = new_objective;
            /*println!("New best floor plan found! wire length: {}, area {}",
                     best_objective.0, best_objective.1);
            println!("Temp {temp}, cost: {cost}");
            fpp.visualize(&fpp.best_sp);*/
        }
        else if random::<f32>() < (-cost/temp).exp() {
            current_sp = sp;
        }
        else if cost > 0.25*(best_objective.0 + best_objective.1) {
            current_sp = fpp.best_sp.clone();
        }

        if temp < print_temp {
            //println!("Current temp: {temp}, evaluations: {evaluations}");
            print_temp = temp - print_temp_interval;
        }
        /*
        if cost > 0.0 {
            println!("temp {temp}, cost {cost}, (-cost/temp).exp() {},", (-cost/temp).exp());
        }*/
        temp = temp*alpha;
    }
    /*
    println!("Simulated annealing solution: wire length: {}, area {}",
             best_objective.0, best_objective.1);
    println!("Improved from initial_objective: wire length: {}, area {}",
            initial_objective.0, initial_objective.1);
    println!("Total evaluations: {evaluations}");*/
    //fpp.visualize(&fpp.best_sp);
    best_objective.0 + best_objective.1
}

pub fn hill_climbing(
    fpp: &mut FloorPlantProblem,
    area_importance: f32,
    max_steps: i32
) -> f32 {
    let mut best_objective = fpp.get_wire_length_estimate_and_area(&fpp.best_sp);
    let initial_objective = best_objective.clone();
    /*println!("Initial solution: wire length: {}, area {}",
             best_objective.0, best_objective.1);
    fpp.visualize(&fpp.best_sp);*/
    let mut evaluations = 0;
    let mut max_steps = max_steps;
    loop {
        let mut best_sp = fpp.best_sp.clone();
        let mut better_found = false;
        for (sp, new_objective) in fpp.get_all_sp_neighbours_with_obj(&fpp.best_sp, false) {
            evaluations += 1;
            let cost = (1.0 - area_importance) * new_objective.0 + area_importance * new_objective.1
                - (1.0 - area_importance) * best_objective.0 - area_importance * best_objective.1;
            if cost < 0.0 {
                best_sp = sp;
                fpp.best_sp = best_sp.clone();
                /*println!("new obj: {} + {} = {}, best obj: {} + {} = {}",
                         new_objective.0, new_objective.1,
                         new_objective.0 + new_objective.1,
                         best_objective.0, best_objective.1,
                         best_objective.0 + best_objective.1
                );*/
                best_objective = new_objective;
                /*println!("New best floor plan found! wire length: {}, area {}",
                         best_objective.0, best_objective.1);
                println!("Cost: {cost}, evaluations: {evaluations}");
                //fpp.visualize(&fpp.best_sp);*/
                better_found = true;
            }
        }
        max_steps -= 1;
        if !better_found || max_steps == 0 { break }
    }
    /*
    println!("Hill climbing solution: wire length: {}, area {}",
             best_objective.0, best_objective.1);
    println!("Improved from initial_objective: wire length: {}, area {}",
             initial_objective.0, initial_objective.1);
    println!("Total evaluations: {evaluations}");
    fpp.visualize(&fpp.best_sp);*/
    best_objective.0 + best_objective.1
}


pub fn monte_carlo_estimation_search(
    fpp: &FloorPlantProblem,
    samples: usize,
    n_moves: usize,
    should_see: f32,
) -> (Vec<f32>, SequencePair, f32) {
    let initial_objective = fpp.get_wire_length_estimate_and_area(&fpp.best_sp);
    let init_obj = initial_objective.0 + initial_objective.1;
    let mut should_see = should_see;
    if should_see < 0.0 { should_see = init_obj - 1.0 }
    let init_moves: Vec<SequencePair> = fpp.get_all_sp_neighbours(&fpp.best_sp, true);
    let init_moves_len = init_moves.len();
    let mut best_sp_seen = fpp.best_sp.clone();
    let mut best_obj = init_obj;
    let mut init_move_value_estimations: Vec<f32> = vec![0.0; init_moves_len];
    let mut best_init_move_index: usize = 0;

    while best_obj > should_see  {
        for _ in 0..samples {
            let initial_move = random::<usize>() % init_moves_len;
            let mut sp = init_moves[initial_move].clone();
            for _ in 0..n_moves {
                let mut aux_sp = fpp.get_random_sp_neighbour(&sp, true);
                while aux_sp.x != sp.x && aux_sp.y != sp.y &&
                    aux_sp.x != fpp.best_sp.x && aux_sp.y != fpp.best_sp.y {
                    aux_sp = fpp.get_random_sp_neighbour(&sp, true);
                }
                sp = aux_sp
            }
            let obj = fpp.get_wire_length_estimate_and_area(&sp);
            let obj_sum = obj.0 + obj.1;
            if best_obj > obj_sum {
                best_obj = obj_sum;
                best_sp_seen = sp;
                best_init_move_index = initial_move;
            }
            let diff_with_initial = init_obj - obj_sum;
            if diff_with_initial > init_move_value_estimations[initial_move] {
                init_move_value_estimations[initial_move] = diff_with_initial;
            }
        }
        should_see += 1.0;
    }
    /*
    println!("Init move value estimations with {samples} \
              samples and n moves {n_moves}");
    println!("Best sp obj {best_obj} and obj diff seen {}", init_obj - best_obj);
    //fpp.visualize(&best_sp_seen);


    let mut i = 0;
    print!("[");
    for mv in &init_move_value_estimations {
        print!("{i}: {mv}, ");
        i += 1;
    }
    println!("]");
    */
    if best_obj >= init_obj {
        // println!("Nothing better found");
        //return (vec![], fpp.best_sp.clone(), best_obj)
    }
    let mut aux_obj = fpp.get_wire_length_estimate_and_area(
        &init_moves[best_init_move_index]
    );
    let mut best_move_obj = aux_obj.0 + aux_obj.1;
    for i in 0..init_moves_len {
        if init_move_value_estimations[i] == init_move_value_estimations[best_init_move_index] {
            aux_obj = fpp.get_wire_length_estimate_and_area(&init_moves[i]);
            let other_obj = aux_obj.0+aux_obj.1;
            if best_move_obj > other_obj {
                best_init_move_index = i;
                best_move_obj = other_obj;
            }
        }
    }
    //println!("Chosen move: {} with estimate {}",
    //         best_init_move_index, init_move_value_estimations[best_init_move_index]);
    (init_move_value_estimations, init_moves[best_init_move_index].clone(), best_obj)
}

struct MonteCarloNode {
    sp: SequencePair,
    childs: Vec<MonteCarloNode>
}
