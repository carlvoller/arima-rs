use std::any::TypeId;

use anyhow::Result;
use ndarray::{ArrayBase, Data, Ix2, RawDataClone};

use super::sklearn::check_array;

pub fn check_exog<D: Data<Elem = f64> + RawDataClone>(
    y: ArrayBase<D, Ix2>,
    copy: bool,
    force_all_finite: bool,
) -> Result<ArrayBase<D, Ix2>> {
    let endog = check_array()
        .arr(y)
        .ensure_2d(true)
        .force_all_finite(force_all_finite)
        .copy(copy)
        .dtype(TypeId::of::<f64>())
        .call()?;

    Ok(endog)
}
