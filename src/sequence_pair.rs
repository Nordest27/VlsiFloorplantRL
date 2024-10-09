use std::cmp::PartialEq;

#[derive(PartialEq, Debug, Copy, Clone)]
enum LcsState {
    Nothing,
    Equal,
    XLonger,
    YLonger
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
        print!("EQ! x[{}] = {}, y[{}] = {}\n", ui, x[ui], uj, y[uj]);
        let max_len = compute_lcs_table(
            lcs_table, weights, x, y,i-1, j-1
        );
        println!("Weights length: {}", weights.len());
        println!("X length: {}", x.len());
        lcs_table[ui][uj] = (LcsState::Equal, max_len + weights[x[ui] as usize]);
        return max_len + 1;
    }
    print!("NEQ! x[{}] = {}, y[{}] = {}\n", ui, x[ui], uj, y[uj]);

    let max_len_x = compute_lcs_table(
        lcs_table, weights, x, y, i-1, j
    );
    let max_len_y = compute_lcs_table(
        lcs_table, weights, x, y, i, j-1
    );

    if max_len_x == 0 && max_len_y == 0 { return 0; }
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
        println!("({}, {})", i, j);
        if lcs_table[i as usize][j as usize].0 == LcsState::Equal {
            println!("{}", x[i as usize]);
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
    vec![vec![(LcsState::Nothing, 0); n]; n]
}

fn get_base_weights(
    weights: &Vec<i32>,
    x: &Vec<i32>,
    y: &Vec<i32>,
) -> Vec<i32> {
    let n = x.len();
    let mut base_weights: Vec<i32> = vec![0; n];
    let mut lcs_table = get_lcs_init_table(n);
    for i in 0..n {
        let positions = (x.binary_search(&(i as i32)), x.binary_search(&(i as i32)));
        match positions {
            (Ok(x_pos), Ok(y_pos)) =>
                base_weights[i] = compute_lcs_table(
                    &mut lcs_table,
                    weights,
                    &x,
                    &y,
                    (x_pos as i32)-1,
                    (y_pos as i32)-1,
                ),
            _ => println!("ERROR this should not happen!")
        }
    }
    base_weights 
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

    pub fn get_base_widths(&self, widths: &Vec<i32>) -> Vec<i32> {
        get_base_weights(widths, &self.x, &self.y)
    }

    pub fn get_base_heights(&self, heights: &Vec<i32>) -> Vec<i32> {
        let rev_x = self.x.iter().rev().map(|x| *x).collect::<Vec<i32>>();
        get_base_weights(heights, &rev_x, &self.y)
    }
}

pub struct FloorPlantProblem {
    n: i32,
    blocks_max_widths: Vec<i32>,
    blocks_min_widths: Vec<i32>,
    blocks_max_heights: Vec<i32>,
    blocks_min_heights: Vec<i32>,
}

#[cfg(test)]
mod tests {
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
            vec![LcsState::Nothing, LcsState::Equal, LcsState::Nothing, LcsState::Nothing],
            vec![LcsState::Nothing, LcsState::XLonger, LcsState::Equal, LcsState::Nothing],
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
        for _ in 0..100 {
            let n: usize = 10;
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
}