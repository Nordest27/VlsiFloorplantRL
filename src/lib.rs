// Using as a guide https://saidvandeklundert.net/learn/2021-11-18-calling-rust-from-python-using-pyo3/
use pyo3::prelude::*;
use crate::domain::{FloorPlantProblem, SpMove};

mod domain;
mod local_search;

#[pyclass]
pub struct PyFloorPlantProblem {
    fpp: FloorPlantProblem
}

#[pymethods]
impl PyFloorPlantProblem {
    #[new]
    pub fn new(n: usize) -> PyFloorPlantProblem {
        let fpp = FloorPlantProblem::generate_new(n);
        PyFloorPlantProblem { fpp }
    }

    pub fn get_current_sp_objective(&self) -> PyResult<f32> {
        let aux_obj = self.fpp.get_wire_length_estimate_and_area(&self.fpp.best_sp);
        Ok(aux_obj.0 + aux_obj.1)
    }

    pub fn x(&self) -> PyResult<Vec<i32>> { Ok(self.fpp.best_sp.x.clone()) }
    pub fn y(&self) -> PyResult<Vec<i32>> { Ok(self.fpp.best_sp.y.clone()) }
    pub fn widths(&self) -> PyResult<Vec<i32>> { Ok(self.fpp.min_widths.clone()) }
    pub fn heights(&self) -> PyResult<Vec<i32>> { Ok(self.fpp.get_max_heights()) }
    pub fn connected_to(&self) -> PyResult<Vec<Vec<bool>>> { Ok(self.fpp.connected_to.clone()) }
    pub fn offset_heights(&self) -> PyResult<Vec<i32>> {
        Ok(self.fpp.get_base_heights(&self.fpp.best_sp).0)
    }
    pub fn offset_widths(&self) -> PyResult<Vec<i32>> {
        Ok(self.fpp.get_base_widths(&self.fpp.best_sp).0)
    }

    pub fn shuffle_sp(&mut self) { self.fpp.best_sp.shuffle() }

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