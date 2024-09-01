use std::any::TypeId;

use anyhow::Result;
use ndarray::{ArrayBase, Data, Ix1, RawDataClone};

use super::sklearn::check_array;

pub fn check_endog<D: Data<Elem = f64> + RawDataClone>(
    y: ArrayBase<D, Ix1>,
    copy: bool,
    force_all_finite: bool,
) -> Result<ArrayBase<D, Ix1>> {
    let endog = check_array()
        .arr(y)
        .force_all_finite(force_all_finite)
        .copy(copy)
        .dtype(TypeId::of::<f64>())
        .call()?;

    Ok(endog)
}
