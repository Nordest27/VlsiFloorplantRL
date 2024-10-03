use std::cmp::PartialEq;

#[derive(PartialEq, Debug, Copy, Clone)]
enum LcsState {
    Nothing,
    Equal,
    XLonger,
    YLonger
}

fn inner_compute_lcs_table(
    lcs_table: &mut Vec<Vec<LcsState>>,
    x: &Vec<i32>,
    y: &Vec<i32>,
    i: i32,
    j: i32
) -> i32 {
    if i < 0 || j < 0 || lcs_table[i as usize][j as usize] != LcsState::Nothing { return 0; }
    if x[i as usize] == y[j as usize] {
        let max_len = inner_compute_lcs_table(lcs_table, x, y, i-1, j-1);
        lcs_table[i as usize][j as usize] = LcsState::Equal;
        return max_len + 1;
    }
    let max_len_x = inner_compute_lcs_table(lcs_table, x, y, i-1, j);
    let max_len_y = inner_compute_lcs_table(lcs_table, x, y, i, j-1);

    if max_len_x == 0 && max_len_y == 0 { return 0; }
    if max_len_x >= max_len_y {
        lcs_table[i as usize][j as usize] = LcsState::XLonger;
        max_len_x
    } else {
        lcs_table[i as usize][j as usize] = LcsState::YLonger;
        max_len_y
    }
}

fn compute_lcs_table(
    x: &Vec<i32>,
    y: &Vec<i32>,
) -> Vec<Vec<LcsState>> {
    let len = x.len();
    assert_eq!(x.len(), y.len());
    let mut lcs_table: Vec<Vec<LcsState>> = vec![vec![LcsState::Nothing; len]; len];
    inner_compute_lcs_table(&mut lcs_table, x, y, (len-1) as i32, (len-1) as i32);
    lcs_table
}

fn lcs(
    lcs_table: &Vec<Vec<LcsState>>,
    x: &Vec<i32>,
    ini: usize,
    i: usize,
    j: usize,
) -> Vec<i32> {
    let mut result: Vec<i32> = Vec::new();
    let mut i: usize = i;
    let mut j: usize = j;
    while i >= ini && j >= ini && lcs_table[i][j] != LcsState::Nothing {
        if lcs_table[i][j] == LcsState::Equal {
            result.push(x[i]);
            i -= 1;
            j -= 1;
        }
        else if lcs_table[i][j] == LcsState::XLonger {
            i -= 1;
        }
        else if lcs_table[i][j] == LcsState::YLonger {
            j -= 1;
        }
    }
    result
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

    pub fn get_base_widths(widths: &Vec<i32>) -> Vec<i32> {
        vec![widths[0], widths[1], widths[2], widths[3], widths[4]]
    }

    pub fn get_base_heights(heights: &Vec<i32>) -> Vec<i32> {
        vec![heights[0], heights[1], heights[2], heights[3], heights[4]]
    }
}

pub struct FloorPlantProblem {
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
        let expected: [i32;4] = [0, 1, 2, 3];
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
        let sp = SequencePair::new(4);
        let expected_table: Vec<Vec<LcsState>> = vec![
            vec![LcsState::Equal, LcsState::Nothing, LcsState::Nothing, LcsState::Nothing],
            vec![LcsState::Nothing, LcsState::Equal, LcsState::Nothing, LcsState::Nothing],
            vec![LcsState::Nothing, LcsState::Nothing, LcsState::Equal, LcsState::Nothing],
            vec![LcsState::Nothing, LcsState::Nothing, LcsState::Nothing, LcsState::Equal],
        ];
        let lcs_table = compute_lcs_table(&sp.x, &sp.y);
        assert_eq!(lcs_table, expected_table);
    }

    #[test]
    fn simple_test_2_compute_lcs_table() {
        let x = vec![1, 2, 3, 4];
        let y = vec![3, 1, 2, 4];
        let expected_table: Vec<Vec<LcsState>> = vec![
            vec![LcsState::Nothing, LcsState::Equal, LcsState::Nothing, LcsState::Nothing],
            vec![LcsState::Nothing, LcsState::Nothing, LcsState::Equal, LcsState::Nothing],
            vec![LcsState::Equal, LcsState::YLonger, LcsState::XLonger, LcsState::Nothing],
            vec![LcsState::Nothing, LcsState::Nothing, LcsState::Nothing, LcsState::Equal],
        ];
        let lcs_table = compute_lcs_table(&x, &y);
        assert_eq!(lcs_table, expected_table);
    }

    #[test]
    fn test_random_lcs_table() {
        for _ in 0..100 {
            let sp = SequencePair::new_shuffled(10);
            let lcs_table = compute_lcs_table(&sp.x, &sp.y);
            let mut how_many_equals: i32 = 0;
            for i in 0..sp.x.len() {
                for j in 0..sp.y.len() {
                    if lcs_table[i][j] == LcsState::Equal {
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



}