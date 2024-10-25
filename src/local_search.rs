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

#[derive(Clone)]
struct MonteCarloNode {
    sp: SequencePair,
    obj: f32,
    diff_with_previous: f32,
    first_move_index: usize,
}

pub fn monte_carlo_estimation_search(
    fpp: &FloorPlantProblem,
    nodes_to_expand: i32,
    exploration_factor: f32,
) -> (Vec<f32>, SequencePair) {
    let initial_objective = fpp.get_wire_length_estimate_and_area(&fpp.best_sp);
    let init_moves: Vec<(SequencePair, (f32, f32))> = fpp.get_all_sp_neighbours_with_obj(&fpp.best_sp);
    let init_moves_len = init_moves.len();
    let mut best_sp_seen = fpp.best_sp.clone();
    let mut best_obj = initial_objective.0 + initial_objective.1;
    let mut init_move_value_estimations: Vec<f32> = vec![0.0; init_moves_len];
    let mut best_init_move_index: usize = 0;
    let mut nodes: Vec<MonteCarloNode> = vec![];
    for i in 0..init_moves_len {
        let obj_sum = init_moves[i].1.0 + init_moves[i].1.1;
        let diff_with_previous = initial_objective.0 + initial_objective.1 - obj_sum;
        nodes.push(MonteCarloNode {
            sp: init_moves[i].0.clone(),
            obj: obj_sum,
            diff_with_previous,
            first_move_index: i
        });
        init_move_value_estimations[i] = diff_with_previous;
    }
    for i in init_moves_len..(init_moves_len + nodes_to_expand as usize) {
        let mut should_explore = random::<f32>();
        let mut rand_i = random::<usize>()%i;

        while should_explore > exploration_factor && nodes[rand_i].diff_with_previous < 0.0 {
            rand_i = random::<usize>()%i;
            should_explore -= 0.0001;
        }
        let sp_to_expand = &nodes[rand_i].sp;
        let first_move_index = nodes[rand_i].first_move_index;
        let (sp, obj) = fpp.get_random_sp_neighbour_with_obj(sp_to_expand);
        let obj_sum = obj.0 + obj.1;
        let diff_with_previous = &nodes[rand_i].obj - obj_sum;
        if best_obj > obj_sum {
            best_obj = obj_sum;
            best_sp_seen  = sp.clone();
            best_init_move_index = first_move_index;
        }
        nodes.push(MonteCarloNode { sp, obj: obj_sum, diff_with_previous, first_move_index });

        let diff_with_initial = initial_objective.0 + initial_objective.1 - obj_sum;
        if diff_with_initial > init_move_value_estimations[first_move_index] {
            init_move_value_estimations[first_move_index] = diff_with_initial;
        }
    }
    println!("Init move value estimations with {nodes_to_expand} \
              samples and exploration factor {exploration_factor}");
    println!("Best sp obj {best_obj} and obj diff seen {}", initial_objective.0 + initial_objective.1 - best_obj);
    //fpp.visualize(&best_sp_seen);
    /*
    print!("[");
    for mv in &init_move_value_estimations {
        print!("{mv}, ");
    }
    println!("]");
    */
    for i in 0..init_moves_len {
        if init_move_value_estimations[i] == init_move_value_estimations[best_init_move_index] &&
            (nodes[i].obj < nodes[best_init_move_index].obj ||
                nodes[best_init_move_index].obj == initial_objective.0 + initial_objective.1)
        {
            best_init_move_index = i;
        }
    }
    (init_move_value_estimations, nodes[best_init_move_index].sp.clone())
}