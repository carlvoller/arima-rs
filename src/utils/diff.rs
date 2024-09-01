use std::any::TypeId;

use anyhow::{bail, Result};
use ndarray::{ArrayBase, Data, Ix2, RawDataClone};

use crate::errors::ArimaError;

use super::sklearn::check_array;

pub fn diff<D: Data<Elem = f64> + RawDataClone>(
    arr: ArrayBase<D, Ix2>,
    lag: Option<u32>,
    differences: Option<u32>,
) -> Result<ArrayBase<D, Ix2>> {
    let lag_unwrapped = lag.unwrap_or(1);
    let differences_unwrapped = differences.unwrap_or(1);

    if lag_unwrapped < 1 {
        bail!(ArimaError::ValueError {
            expected: String::from("lag > 0"),
            found: lag_unwrapped.to_string()
        })
    }

    if differences_unwrapped < 1 {
        bail!(ArimaError::ValueError {
            expected: String::from("differences > 0"),
            found: differences_unwrapped.to_string()
        })
    }

    let x = check_array()
        .arr(arr)
        .dtype(TypeId::of::<f64>())
        .call()?;

    Ok(x)
}
