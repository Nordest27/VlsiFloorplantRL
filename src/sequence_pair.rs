use std::cmp::PartialEq;

#[derive(PartialEq, Debug, Copy, Clone)]
enum LcsState {
    Nothing,
    Equal,
    XLonger,
    YLonger
}
fn iterative_compute_lcs_table(
    lcs_table: &mut Vec<Vec<(LcsState, i32)>>,
    weights: &Vec<i32>,
    x: &Vec<i32>,
    y: &Vec<i32>,
    inp_i: i32,
    inp_j: i32
) -> i32 {
    if inp_i < 0 || inp_j < 0 { return 0; }
    let mut positions: Vec<(i32, i32)> = vec![(inp_i, inp_j)];
    let mut eval_positions: Vec<(usize, usize)> = Vec::new();

    while let Some((i, j)) = positions.pop() {
        if i < 0 || j < 0 { continue }
        let ui = i as usize;
        let uj = j as usize;
        if lcs_table[ui][uj].1 != -1 { continue }
        lcs_table[ui][uj].1 = 0;

        //println!("first ui: {}, uj: {}", ui, uj);
        eval_positions.push((ui, uj));

        if x[ui] == y[uj] {
            positions.push((i - 1, j - 1));
            continue;
        }
        positions.push((i - 1, j));
        positions.push((i, j - 1));
    }

    eval_positions.sort();


    for (ui, uj) in eval_positions {
        //println!("second ui: {}, uj: {}", ui, uj);
        if x[ui] == y[uj] {
            let max_len = match ui > 0 && uj > 0 {
                true => lcs_table[ui - 1][uj - 1].1,
                false => 0
            };
            lcs_table[ui][uj] = (LcsState::Equal, max_len + weights[x[ui] as usize]);
            continue;
        }
        let max_len_x = match ui > 0 {
            true => lcs_table[ui - 1][uj].1,
            false => 0
        };
        let max_len_y = match uj > 0 {
            true => lcs_table[ui][uj - 1].1,
            false => 0
        };
        if max_len_x >= max_len_y {
            lcs_table[ui][uj] = (LcsState::XLonger, max_len_x);
        } else {
            lcs_table[ui][uj] = (LcsState::YLonger, max_len_y);
        }
    }

    lcs_table[inp_i as usize][inp_j as usize].1
}

fn compute_lcs_table(
    lcs_table: &mut Vec<Vec<(LcsState, i32)>>,
    weights: &Vec<i32>,
    x: &Vec<i32>,
    y: &Vec<i32>,
    i: i32,
    j: i32
) -> i32 {
    if i < 0 || j < 0 { return 0 }
    let ui = i as usize;
    let uj = j as usize;
    let (state, max_len) = lcs_table[ui][uj];
    if state != LcsState::Nothing { return max_len }

    if x[ui] == y[uj] {
        // print!("EQ! x[{}] = {}, y[{}] = {}\n", ui, x[ui], uj, y[uj]);
        let max_len = compute_lcs_table(
            lcs_table, weights, x, y,i-1, j-1
        );
        // println!("Weights length: {}", weights.len());
        // println!("X length: {}", x.len());
        lcs_table[ui][uj] = (LcsState::Equal, max_len + weights[x[ui] as usize]);
        return lcs_table[ui][uj].1;
    }
    // print!("NEQ! x[{}] = {}, y[{}] = {}\n", ui, x[ui], uj, y[uj]);

    let max_len_x = compute_lcs_table(
        lcs_table, weights, x, y, i-1, j
    );
    let max_len_y = compute_lcs_table(
        lcs_table, weights, x, y, i, j-1
    );

    if max_len_x >= max_len_y {
        lcs_table[ui][uj] = (LcsState::XLonger, max_len_x);
        max_len_x
    } else {
        lcs_table[ui][uj] = (LcsState::YLonger, max_len_y);
        max_len_y
    }
}

fn print_lcs_table(
    lcs_table: &Vec<Vec<(LcsState, i32)>>
) {
    for v in lcs_table {
        println!();
        for (state, max_weight) in v {
            print!("(");
            match state {
                LcsState::Equal   => print!("EQUAL  , "),
                LcsState::Nothing => print!("NOTHING, "),
                LcsState::XLonger => print!("XLONGER, "),
                LcsState::YLonger => print!("YLONGER, ")
            }
            print!("{}), ", max_weight);

        }
    }
    println!();
}

fn lcs(
    lcs_table: &mut Vec<Vec<(LcsState, i32)>>,
    weights: &Vec<i32>,
    x: &Vec<i32>,
    y: &Vec<i32>,
    ini: usize,
    i: usize,
    j: usize,
) -> Vec<i32> {
    assert_eq!(x.len(), y.len());
    assert!(x.len() > i);
    assert!(y.len() > j);
    assert!(ini <= j && ini <= i);
    compute_lcs_table(lcs_table, weights, x, y, i as i32, j as i32);
    let mut result: Vec<i32> = Vec::new();
    let ini: i32 = ini as i32;
    let mut i: i32 = i as i32;
    let mut j: i32 = j as i32;
    while i >= ini && j >= ini && lcs_table[i as usize][j as usize].0 != LcsState::Nothing {
        if lcs_table[i as usize][j as usize].0 == LcsState::Equal {
            result.insert(0, x[i as usize]);
            i -= 1;
            j -= 1;
        }
        else if lcs_table[i as usize][j as usize].0 == LcsState::XLonger {
            i -= 1;
        }
        else if lcs_table[i as usize][j as usize].0 == LcsState::YLonger {
            j -= 1;
        }
    }
    result
}

pub fn get_lcs_init_table(n: usize) -> Vec<Vec<(LcsState, i32)>> {
    vec![vec![(LcsState::Nothing, -1); n]; n]
}

fn get_base_weights(
    weights: &Vec<i32>,
    x: &Vec<i32>,
    y: &Vec<i32>,
) -> (Vec<i32>, i32) {
    let n = x.len();
    let mut base_weights: Vec<i32> = vec![0; n];
    let mut lcs_table = get_lcs_init_table(n);
    let mut x_positions = vec![0; n];
    let mut y_positions = vec![0; n];
    let total_weight = iterative_compute_lcs_table(
        &mut lcs_table,
        weights, &x, &y,
        (n as i32)-1,
        (n as i32)-1
    );
    for i in 0..n {
        x_positions[x[i] as usize] = i as i32;
        y_positions[y[i] as usize] = i as i32;
    }
    for i in 0..n {
        base_weights[i] = iterative_compute_lcs_table(
            &mut lcs_table,
            weights, &x, &y,
            x_positions[i]-1,
            y_positions[i]-1,
        )
    }
    //print_lcs_table(&lcs_table);
    //println!();
    (base_weights, total_weight)
}

pub struct SequencePair {
    x: Vec<i32>,
    y: Vec<i32>,
}

impl SequencePair {
    pub fn new(n: i32) -> Self {
        let mut x: Vec<i32> = vec![0; n as usize];
        let mut y: Vec<i32> = vec![0; n as usize];
        for i in 0..n {
            x[i as usize] = i;
            y[i as usize] = i;
        }
        SequencePair {x, y}
    }

    fn shuffle(&mut self) {
        for i in (1..self.x.len()).rev() {
            self.x.swap(i, (rand::random::<i32>()%(i as i32)).abs() as usize);
            self.y.swap(i, (rand::random::<i32>()%(i as i32)).abs() as usize);
        }
    }

    pub fn new_shuffled(n: i32) -> Self {
        let mut sp = SequencePair::new(n);
        sp.shuffle();
        sp
    }

    pub fn get_base_widths(&self, widths: &Vec<i32>) -> (Vec<i32>, i32) {
        get_base_weights(widths, &self.x, &self.y)
    }

    pub fn get_base_heights(&self, heights: &Vec<i32>) -> (Vec<i32>, i32) {
        let rev_x = self.x.iter().copied().rev().collect();
        get_base_weights(heights, &rev_x, &self.y)
    }

    pub fn visualize(&self, widths: &Vec<i32>, heights: &Vec<i32>) {
        println!("Getting widths");
        let (width_offsets, width) = self.get_base_widths(widths);
        println!("Getting heights");
        let (height_offsets, height) = self.get_base_heights(heights);

        println!("Filling visualization matrix");
        let mut vis_mat: Vec<Vec<i32>> = vec![vec![-1; width as usize]; height as usize];
        let n = self.x.len();
        for i in 0..n {
            let width_offset: usize = width_offsets[i] as usize;
            let height_offset: usize = height_offsets[i] as usize;
            for h in 0..(heights[i] as usize) {
                for w in 0..(widths[i] as usize) {
                    vis_mat[h+height_offset][w+width_offset] = i as i32
                }
            }
        }
        println!("Showing matrix");
        for i in 0..(height as usize) {
            for j in 0..(width as usize) {
                match vis_mat[i][j] {
                    -1 => print!("    | "),
                    v => print!("{: >4}| ", v)
                }
            }
            println!();
            for _ in 0..(width as usize) {print!("----|-")}
            println!();
        }
        println!();
    }
}

pub struct FloorPlantProblem {
    n: i32,
    blocks_max_widths: Vec<i32>,
    blocks_min_widths: Vec<i32>,
    blocks_max_heights: Vec<i32>,
    blocks_min_heights: Vec<i32>,
    blocks_area: Vec<i32>,
}

#[cfg(test)]
mod tests {
    use rand::random;
    use super::*;

    #[test]
    fn test_sequence_pair_creation() {
        let expected: [i32; 4] = [0, 1, 2, 3];
        let sequence_pair = SequencePair::new(4);
        assert_eq!(sequence_pair.x, expected);
        assert_eq!(sequence_pair.y, expected);
    }

    #[test]
    fn test_sequence_pair_shuffled() {
        for i in 1..100 {
            let sequence_pair = SequencePair::new_shuffled(10);
            let all_true: Vec<bool> = vec![true; 10];
            let mut x_exists: Vec<bool> = vec![false; sequence_pair.y.len()];
            let mut y_exists: Vec<bool> = vec![false; sequence_pair.y.len()];
            for j in 0..sequence_pair.x.len() {
                x_exists[sequence_pair.x[j] as usize] = true;
                y_exists[sequence_pair.y[j] as usize] = true;
            }
            assert_eq!(x_exists, all_true);
            assert_eq!(y_exists, all_true);
        }
    }

    #[test]
    fn simple_test_compute_lcs_table() {
        let n: usize = 4;
        let sp = SequencePair::new(n as i32);
        println!("x, y");
        for i in 0..n {
           println!("{}, {}", sp.x[i], sp.y[i]);
        }
        let expected_table: Vec<Vec<LcsState>> = vec![
            vec![LcsState::Equal, LcsState::Nothing, LcsState::Nothing, LcsState::Nothing],
            vec![LcsState::Nothing, LcsState::Equal, LcsState::Nothing, LcsState::Nothing],
            vec![LcsState::Nothing, LcsState::Nothing, LcsState::Equal, LcsState::Nothing],
            vec![LcsState::Nothing, LcsState::Nothing, LcsState::Nothing, LcsState::Equal],
        ];
        let mut lcs_table = get_lcs_init_table(n);
        print_lcs_table(&lcs_table);
        println!();
        let weights = vec![1; n];
        lcs(&mut lcs_table, &weights, &sp.x, &sp.y,0, n-1, n-1);

        let mut result_table = vec![vec![LcsState::Nothing; n]; n];
        for i in 0..lcs_table.len() {
            for j in 0..lcs_table[i].len() {
                result_table[i][j] = lcs_table[i][j].0;
            }
        }
        print_lcs_table(&lcs_table);
        assert_eq!(result_table, expected_table);
    }

    #[test]
    fn simple_test_2_compute_lcs_table() {
        let n: usize = 4;
        let x = vec![0, 1, 2, 3];
        let y = vec![2, 0, 1, 3];
        let sp = SequencePair {x, y};
        let expected_table: Vec<Vec<LcsState>> = vec![
            vec![LcsState::XLonger, LcsState::Equal, LcsState::Nothing, LcsState::Nothing],
            vec![LcsState::XLonger, LcsState::XLonger, LcsState::Equal, LcsState::Nothing],
            vec![LcsState::Equal, LcsState::XLonger, LcsState::XLonger, LcsState::Nothing],
            vec![LcsState::Nothing, LcsState::Nothing, LcsState::Nothing, LcsState::Equal],
        ];

        let mut lcs_table = get_lcs_init_table(n);
        let weights = vec![1; n];
        lcs(&mut lcs_table, &weights, &sp.x, &sp.y, 0, n-1, n-1);

        let mut result_table = vec![vec![LcsState::Nothing; n]; n];
        for i in 0..lcs_table.len() {
            for j in 0..lcs_table[i].len() {
                result_table[i][j] = lcs_table[i][j].0;
            }
        }
        assert_eq!(result_table, expected_table);
    }

    #[test]
    fn simple_test_3_compute_lcs_table() {
        let n: usize = 4;
        let x = vec![3, 1, 2, 0];
        let y = vec![3, 0, 1, 2];
        let sp = SequencePair {x, y};

        let expected_table: Vec<Vec<LcsState>> = vec![
            vec![LcsState::Equal, LcsState::YLonger, LcsState::Nothing, LcsState::Nothing],
            vec![LcsState::XLonger, LcsState::XLonger, LcsState::Equal, LcsState::Nothing],
            vec![LcsState::XLonger, LcsState::XLonger, LcsState::XLonger, LcsState::Equal],
            vec![LcsState::Nothing, LcsState::Equal, LcsState::XLonger, LcsState::XLonger],
        ];

        let weights = vec![1; n];
        let mut lcs_table = get_lcs_init_table(n);
        lcs(&mut lcs_table, &weights, &sp.x, &sp.y, 0, n-1, n-1);

        let mut result_table = vec![vec![LcsState::Nothing; n]; n];
        for i in 0..lcs_table.len() {
            for j in 0..lcs_table[i].len() {
                result_table[i][j] = lcs_table[i][j].0;
            }
        }
        assert_eq!(result_table, expected_table);
    }

    #[test]
    fn test_random_lcs_table() {
        let n: usize = 10;
        for _ in 0..100 {
            let sp = SequencePair::new_shuffled(n as i32);
            let mut lcs_table = get_lcs_init_table(n);
            let weights = vec![1; n];
            lcs(&mut lcs_table, &weights, &sp.x, &sp.y, 0, sp.x.len()-1, sp.y.len()-1);
            let mut how_many_equals: i32 = 0;
            for i in 0..sp.x.len() {
                for j in 0..sp.y.len() {
                    if lcs_table[i][j].0 == LcsState::Equal {
                        assert_eq!(sp.x[i], sp.y[j]);
                        how_many_equals += 1;
                    }
                    else {
                        assert_ne!(sp.x[i], sp.y[j]);
                    }
                }
            }
            assert_eq!(how_many_equals, 10);
        }
    }

    #[test]
    fn test_lcs() {
        let n: usize = 10;
        let x: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let y: Vec<i32> = vec![0, 4, 5, 7, 8, 1, 3, 2, 9, 6];

        let weights = vec![1; n];
        let sp = SequencePair {x, y};
        let mut lcs_table = get_lcs_init_table(n);

        let expected_lcs: Vec<i32> = vec![0, 4, 5, 7, 8, 9];
        let result_lcs: Vec<i32> = lcs(&mut lcs_table, &weights, &sp.x, &sp.y, 0, 9, 9);
        assert_eq!(expected_lcs, result_lcs);

        let expected_lcs: Vec<i32> = vec![1, 2];
        let result_lcs: Vec<i32> = lcs(&mut lcs_table, &weights, &sp.x, &sp.y, 1, 5, 8);
        assert_eq!(expected_lcs, result_lcs);

        let expected_lcs: Vec<i32> = vec![4, 5, 7];
        let result_lcs: Vec<i32> = lcs(&mut lcs_table, &weights, &sp.x, &sp.y, 1, 7, 3);
        assert_eq!(expected_lcs, result_lcs);

        let expected_lcs: Vec<i32> = vec![8, 9];
        let result_lcs: Vec<i32> = lcs(&mut lcs_table, &weights, &sp.x, &sp.y, 4, 9, 9);
        assert_eq!(expected_lcs, result_lcs);
    }

    #[test]
    fn test_widths_and_heights() {
        let n: i32 = 6;
        let sp: SequencePair = SequencePair {
            x: vec![1, 3, 2, 4, 5, 0],
            y: vec![3, 1, 0, 4, 5, 2]
        };
        let heights = vec![6, 6, 3, 3, 6, 6];
        let widths = vec![3, 3, 4, 4, 2, 2];
        let (height_offsets, total_height) = sp.get_base_heights(&heights);
        let (width_offsets, total_width) = sp.get_base_widths(&widths);

        let expected_height_offsets = vec![0, 3, 12, 0, 6, 6];
        let expected_total_height = 15;
        let expected_width_offsets = vec![4, 0, 4, 0, 4, 6];
        let expected_total_width = 8;

        assert_eq!(expected_height_offsets, height_offsets);
        assert_eq!(expected_total_height, total_height);
        assert_eq!(expected_width_offsets, width_offsets);
        assert_eq!(expected_total_width, total_width);

    }

    #[test]
    fn test_widths_and_heights_2() {
        let n: i32 = 6;
        let sp: SequencePair = SequencePair {
            x: vec![1, 0, 2, 4, 5, 3],
            y: vec![0, 1, 3, 4, 5, 2]
        };
        let heights = vec![6, 6, 3, 3, 6, 6];
        let widths = vec![3, 3, 4, 4, 2, 2];
        let (height_offsets, total_height) = sp.get_base_heights(&heights);
        let (width_offsets, total_width) = sp.get_base_widths(&widths);

        let expected_height_offsets = vec![0, 6, 9, 0, 3, 3];
        let expected_total_height = 12;
        let expected_width_offsets = vec![0, 0, 3, 3, 3, 5];
        let expected_total_width = 7;

        assert_eq!(expected_height_offsets, height_offsets);
        assert_eq!(expected_total_height, total_height);
        assert_eq!(expected_width_offsets, width_offsets);
        assert_eq!(expected_total_width, total_width);
    }

    #[test]
    fn test_recursive_and_iterative_compute_lcs_table() {
        let n: usize = 4;
        for iter in 0..100 {
            println!("Iter {}", iter);
            let sp = SequencePair::new_shuffled(n as i32);
            let mut weights = vec![0; n];
            for i in 0..n { weights[i] = random::<i32>().abs()%10 }
            let mut rec_lcs_table = get_lcs_init_table(n);
            let mut ite_lcs_table = get_lcs_init_table(n);
            let rec_full_weight = compute_lcs_table(
                &mut rec_lcs_table, &weights,
                &sp.x, &sp.y,
                (n-1) as i32, (n-1) as i32
            );
            let ite_full_weight = iterative_compute_lcs_table(
                &mut ite_lcs_table, &weights,
                &sp.x, &sp.y,
                (n-1) as i32, (n-1) as i32
            );
            print_lcs_table(&rec_lcs_table);
            print_lcs_table(&ite_lcs_table);
            assert_eq!(rec_full_weight, ite_full_weight);
            assert_eq!(rec_lcs_table, ite_lcs_table);
        }
    }
}