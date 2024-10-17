// Using as a guide https://saidvandeklundert.net/learn/2021-11-18-calling-rust-from-python-using-pyo3/
use pyo3::prelude::*;
use crate::domain::{FloorPlantProblem, SpMove};

mod domain;
mod local_search;

#[pyclass]
pub struct PyFloorPlantProblem {
    fpp: FloorPlantProblem,
    #[pyo3(get)]
    pub x: Vec<i32>,
    #[pyo3(get)]
    pub y: Vec<i32>,
    #[pyo3(get)]
    pub widths: Vec<i32>,
    #[pyo3(get)]
    pub heights: Vec<i32>,
    #[pyo3(get)]
    pub connected_to: Vec<Vec<bool>>
}

#[pymethods]
impl PyFloorPlantProblem {
    #[new]
    pub fn new(n: usize) -> PyFloorPlantProblem {
        let fpp = FloorPlantProblem::generate_new(n);
        let x = fpp.best_sp.x.clone();
        let y = fpp.best_sp.y.clone();
        let widths: Vec<i32> = fpp.min_widths.clone();
        let heights: Vec<i32> = fpp.get_max_heights();
        let connected_to: Vec<Vec<bool>> = fpp.connected_to.clone();

        PyFloorPlantProblem { fpp, x, y, widths, heights, connected_to }
    }

    pub fn get_current_sp_objective(&self) -> PyResult<f32> {
        let aux_obj = self.fpp.get_wire_length_estimate_and_area(&self.fpp.best_sp);
        Ok(aux_obj.0 + aux_obj.1)
    }

    pub fn apply_sp_move(
        &mut self,
        i: usize,
        j: usize,
        move_type: usize
    ) -> PyResult<f32> {
        let n = self.fpp.best_sp.x.len();

        let mut x_positions = vec![0; n];
        let mut y_positions = vec![0; n];

        for u in 0..n {
            x_positions[self.fpp.best_sp.x[u] as usize] = u;
            y_positions[self.fpp.best_sp.y[u] as usize] = u;
        }
        let mut obj = -100.0;
        if i < n && j < n && move_type < 9 {
            self.fpp.best_sp = SpMove::new(move_type).execute_move(
                &self.fpp.best_sp,
                x_positions[i],
                x_positions[j],
                y_positions[i],
                y_positions[j]
            );
            let aux_obj = self.fpp.get_wire_length_estimate_and_area(&self.fpp.best_sp);
            obj = aux_obj.0 + aux_obj.1;
        }
        Ok(obj)
    }

    pub fn visualize(&self) {
        self.fpp.visualize(&self.fpp.best_sp);
    }
}

#[pymodule]
fn vlsi_floorplant(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFloorPlantProblem>()?;
    Ok(())
}

fn main() {}