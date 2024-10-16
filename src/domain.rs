use std::cmp::PartialEq;
use rand::{random, thread_rng};

fn iterative_compute_lcs_table(
    weights: &Vec<i32>,
    x: &Vec<i32>,
    y: &Vec<i32>,
) -> Vec<Vec<i32>> {
    let n = weights.len();
    let mut lcs_table = vec![vec![0; n+1]; n+1];
    for ui in 1..(n+1) {
        for uj in 1..(n+1) {
            if x[ui-1] == y[uj-1] {
                lcs_table[ui][uj] = lcs_table[ui-1][uj-1] + weights[x[ui-1] as usize];
            }
            else {
                lcs_table[ui][uj] = lcs_table[ui - 1][uj].max(lcs_table[ui][uj - 1]);
            }
        }
    }
    lcs_table
}

pub fn get_lcs_init_table(n: usize) -> Vec<Vec<i32>> {
    vec![vec![-1; n+1]; n+1]
}

fn compute_lcs_table(
    lcs_table: &mut Vec<Vec<i32>>,
    weights: &Vec<i32>,
    x: &Vec<i32>,
    y: &Vec<i32>,
    i: i32,
    j: i32
) -> i32{
    if i < 1 || j < 1 { return 0 }

    let ui = i as usize;
    let uj = j as usize;

    if lcs_table[ui][uj] != -1 { return lcs_table[ui][uj] }

    if x[ui-1] == y[uj-1] {
        compute_lcs_table(lcs_table, weights, x, y,i-1, j-1);
        lcs_table[ui][uj] = lcs_table[ui-1][uj-1].max(0) + weights[x[ui-1] as usize];
        return lcs_table[ui][uj];
    }
    compute_lcs_table(lcs_table, weights, x, y, i-1, j);
    compute_lcs_table(lcs_table, weights, x, y, i, j-1);

    lcs_table[ui][uj] = lcs_table[ui - 1][uj].max(lcs_table[ui][uj - 1]).max(0);
    lcs_table[ui][uj]
}

fn print_lcs_table(
    lcs_table: &Vec<Vec<i32>>
) {
    for v in lcs_table {
        println!();
        for max_weight in v {
            print!("{}, ", max_weight);
        }
    }
    println!();
}
/*
fn lcs(
    lcs_table: &mut Vec<Vec<i32>>,
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
    while i >= ini && j >= ini && lcs_table[i as usize][j as usize] != -1 {
        if lcs_table[i as usize][j as usize] == LcsState::Equal {
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
*/

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
    let total_weight = compute_lcs_table(
        &mut lcs_table,
        weights, &x, &y,
        n as i32,
        n as i32
    );
    for i in 0..n {
        x_positions[x[i] as usize] = i as i32;
        y_positions[y[i] as usize] = i as i32;
    }
    for i in 0..n {
        base_weights[i] = compute_lcs_table(
            &mut lcs_table,
            weights, &x, &y,
            x_positions[i],
            y_positions[i],
        )
    }
    //print_lcs_table(&lcs_table);
    //println!();
    (base_weights, total_weight)
}

fn iterative_get_base_weights(
    weights: &Vec<i32>,
    x: &Vec<i32>,
    y: &Vec<i32>,
) -> (Vec<i32>, i32) {
    let n = x.len();
    let mut base_weights: Vec<i32> = vec![0; n];
    let mut x_positions = vec![0; n];
    let mut y_positions = vec![0; n];
    let lcs_table = iterative_compute_lcs_table(weights, &x, &y);
    for i in 0..n {
        x_positions[x[i] as usize] = i;
        y_positions[y[i] as usize] = i;
    }
    for i in 0..n {
        base_weights[i] = lcs_table[x_positions[i]][y_positions[i]];
    }
    (base_weights, lcs_table[n][n])
}

#[derive(Clone)]
pub struct SequencePair {
    pub x: Vec<i32>,
    pub y: Vec<i32>,
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
            self.x.swap(i, (random::<i32>()%(i as i32)).abs() as usize);
            self.y.swap(i, (random::<i32>()%(i as i32)).abs() as usize);
        }
    }

    pub fn new_shuffled(n: i32) -> Self {
        let mut sp = SequencePair::new(n);
        sp.shuffle();
        sp
    }
}

#[derive(PartialEq)]
enum SpMove {
    SwapX,
    SwapY,
    SwapXY,
    MoveRightX,
    MoveRightY,
    MoveRightXY,
    MoveLeftX,
    MoveLeftY,
    MoveLeftXY,
}


impl SpMove {
    fn random() -> Self {
        match random::<usize>()%9 {
            0 => SpMove::SwapX,
            1 => SpMove::SwapY,
            2 => SpMove::SwapXY,
            3 => SpMove::MoveRightX,
            4 => SpMove::MoveRightY,
            5 => SpMove::MoveRightXY,
            6 => SpMove::MoveLeftX,
            7 => SpMove::MoveLeftY,
            _ => SpMove::MoveLeftXY,
        }
    }

    pub fn execute_move(
        self,
        sp: &SequencePair,
        xi: usize, xj: usize,
        yi: usize, yj: usize,
    ) -> SequencePair {
        let (mut xi, mut xj, mut yi, mut yj) = (xi, xj, yi, yj);
        let mut new_sp = sp.clone();
        if xi > xj {
            xi = xi + xj;
            xj = xi - xj;
            xi = xi - xj;
        }
        if yi > yj {
            yi = yi + yj;
            yj = yi - yj;
            yi = yi - yj;
        }
        if self == SpMove::SwapX || self == SpMove::SwapXY {
            new_sp.x.swap(xi, xj);
            assert_ne!(sp.x, new_sp.x);
        }
        if self == SpMove::SwapY || self == SpMove::SwapXY {
            new_sp.y.swap(yj, yi);
            assert_ne!(sp.y, new_sp.y);
        }
        if self == SpMove::MoveRightX || self == SpMove::MoveRightXY {
            while xi < xj {
                new_sp.x.swap(xi, xi+1);
                xi += 1;
            }
            assert_ne!(sp.x, new_sp.x);
        }
        if self == SpMove::MoveRightY || self == SpMove::MoveRightXY {
            while yi < yj {
                new_sp.y.swap(yi, yi+1);
                yi += 1;
            }
            assert_ne!(sp.y, new_sp.y);
        }
        if self == SpMove::MoveLeftX || self == SpMove::MoveLeftXY {
            while xi < xj {
                new_sp.x.swap(xj-1, xj);
                xj -= 1;
            }
            assert_ne!(sp.x, new_sp.x);
        }
        if self == SpMove::MoveLeftY || self == SpMove::MoveLeftXY {
            while yi < yj {
                new_sp.y.swap(yj-1, yj);
                yj -= 1;
            }
            assert_ne!(sp.y, new_sp.y);
        }
        new_sp
    }


}

pub struct FloorPlantProblem {
    pub n: i32,
    pub best_sp: SequencePair,
    pub min_widths: Vec<i32>,
    pub min_heights: Vec<i32>,
    pub blocks_area: Vec<i32>,
    pub connected_to: Vec<Vec<bool>>,
}

impl FloorPlantProblem {
    pub fn get_base_widths(&self, sp: &SequencePair) -> (Vec<i32>, i32) {
        iterative_get_base_weights(&self.min_widths, &sp.x, &sp.y)
    }

    pub fn get_base_heights(&self, sp: &SequencePair) -> (Vec<i32>, i32) {
        let rev_x = sp.x.iter().copied().rev().collect();
        let mut max_heights = self.min_widths.clone();
        for i in 0..max_heights.len() {
            max_heights[i] = self.blocks_area[i] / max_heights[i];
        }
        iterative_get_base_weights(&max_heights, &rev_x, &sp.y)
    }

    pub fn get_base_area(&self, sp: &SequencePair) -> i32 {
        let (_, width) = self.get_base_widths(sp);
        let (_, height) = self.get_base_heights(sp);
        width * height
    }

    pub fn get_base_half_perimeter(&self, sp: &SequencePair) -> i32 {
        let (_, width) = self.get_base_widths(sp);
        let (_, height) = self.get_base_heights(sp);
        width + height
    }

    pub fn get_wire_length_estimate_and_area(&self, sp: &SequencePair) -> (f32, f32) {
        let (width_offsets, full_width) = self.get_base_widths(sp);
        let (height_offsets, full_height) = self.get_base_heights(sp);
        let mut max_heights = self.min_widths.clone();
        for i in 0..max_heights.len() {
            max_heights[i] = self.blocks_area[i] / max_heights[i];
        }
        let n = self.n as usize;
        let mut centers = vec![(0.0, 0.0); n];
        for i in 0..n {
            centers[i] = (
                (width_offsets[i] + self.min_widths[i] / 2) as f32,
                (height_offsets[i] + max_heights[i]) as f32
            );
        }
        let mut wire_length_estimate: f32 = 0.0;
        let mut penalize_touching_sides: f32 = 0.0;
        for i in 0..n {
            if width_offsets[i] + self.min_widths[i] == full_width {
                penalize_touching_sides += 0.5;
            }
            if height_offsets[i] + max_heights[i] == full_height {
                penalize_touching_sides += 0.5;
            }
            for j in i + 1..n {
                if self.connected_to[i][j] {
                    wire_length_estimate +=
                        (centers[i].0 - centers[j].0).abs() + (centers[i].1 - centers[j].1).abs();
                    if wire_length_estimate == 0.0 {
                        println!("centers for {i}: x-{} y-{}", centers[i].0, centers[i].1);
                        println!("centers for {j}: x-{} y-{}", centers[j].0, centers[j].1);
                        assert_ne!(wire_length_estimate, 0.0)
                    }
                }
            }
        }
        (wire_length_estimate, (full_width * full_height) as f32 + penalize_touching_sides)
    }

    pub fn get_random_sp_neighbour_with_obj(
        &self, sp: &SequencePair
    ) -> (SequencePair, (f32, f32)) {
        let n = self.n as usize;
        let which_first = random::<usize>() % n;
        let mut which_second = random::<usize>() % n;
        while which_second == which_first {
            which_second = random::<usize>() % n;
        }
        let mut x_positions = vec![0; n];
        let mut y_positions = vec![0; n];

        for i in 0..n {
            x_positions[sp.x[i] as usize] = i;
            y_positions[sp.y[i] as usize] = i;
        }

        /*
        println!("Which first {which_first}, which second {which_second}");
        println!("Which first x pos {}", x_positions[which_first]);
        println!("Which second x pos {}", x_positions[which_second]);
        println!("Which first y pos {}", y_positions[which_first]);
        println!("Which second y pos {}", y_positions[which_second]);
        */
        let new_sp = SpMove::random().execute_move(
            sp,
            x_positions[which_first],
            x_positions[which_second],
            y_positions[which_first],
            y_positions[which_second]
        );

        (new_sp, self.get_wire_length_estimate_and_area(sp))

    }

    pub fn visualize(&self, sp: &SequencePair) {

        let mut max_heights = self.min_widths.clone();
        for i in 0..max_heights.len() {
            max_heights[i] = self.blocks_area[i] / max_heights[i];
        }

        println!("Getting widths");
        let (width_offsets, width) = self.get_base_widths(sp);
        println!("Getting heights");
        let (height_offsets, height) = self.get_base_heights(sp);

        println!("Filling visualization matrix");
        let mut vis_mat: Vec<Vec<i32>> = vec![vec![-1; width as usize]; height as usize];
        for i in 0..self.n as usize{
            let width_offset: usize = width_offsets[i] as usize;
            let height_offset: usize = height_offsets[i] as usize;
            for h in 0..(max_heights[i] as usize) {
                for w in 0..(self.min_widths[i] as usize) {
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
    /*
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
    */
    #[test]
    fn test_widths_and_heights() {
        let fpp: FloorPlantProblem = FloorPlantProblem {
            n: 6,
            best_sp: SequencePair {
                x: vec![1, 3, 2, 4, 5, 0],
                y: vec![3, 1, 0, 4, 5, 2]
            },
            min_widths: vec![3, 3, 4, 4, 2, 2],
            min_heights: vec![6, 6, 3, 3, 6, 6],
            blocks_area: vec![3*6, 3*6, 4*3, 4*3, 2*6, 2*6],
            connected_to: vec![vec![false]]
        };
        let (height_offsets, total_height) = fpp.get_base_heights(&fpp.best_sp);
        let (width_offsets, total_width) = fpp.get_base_widths(&fpp.best_sp);

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
        let fpp: FloorPlantProblem = FloorPlantProblem {
            n: 6,
            best_sp: SequencePair {
                x: vec![1, 0, 2, 4, 5, 3],
                y: vec![0, 1, 3, 4, 5, 2]
            },
            min_heights: vec![6, 6, 3, 3, 6, 6],
            min_widths: vec![3, 3, 4, 4, 2, 2],
            blocks_area: vec![3*6, 3*6, 4*3, 4*3, 2*6, 2*6],
            connected_to: vec![vec![false]]
        };

        let (height_offsets, total_height) = fpp.get_base_heights(&fpp.best_sp);
        let (width_offsets, total_width) = fpp.get_base_widths(&fpp.best_sp);

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
        let n: usize = 10;
        for iter in 0..1000 {
            println!("Iter {}", iter);
            let sp = SequencePair::new_shuffled(n as i32);
            let mut weights = vec![0; n];
            for i in 0..n { weights[i] = random::<i32>().abs()%10 }
            let rec_weight_offsets = get_base_weights(&weights, &sp.x, &sp.y);
            let ite_weight_offsets = iterative_get_base_weights(&weights, &sp.x, &sp.y);
            assert_eq!(rec_weight_offsets, ite_weight_offsets);
        }
    }
}