use std::{any::TypeId, cmp::min};

use anyhow::{bail, Ok, Result};
use ndarray::{s, Array, ArrayBase, Data, DataMut, Dimension, Ix1, Ix2, RawDataClone};

use crate::errors::ArimaError;

use super::sklearn::check_array;

fn _diff_vector<D: Data<Elem = f64> + DataMut>(
    arr: &mut ArrayBase<D, Ix1>,
    lag: u32,
) -> Array<f64, Ix1> {
    let n = arr.shape()[0];
    let min_lag = min(lag as usize, n);

    // [min_lag: n] - arr[: n-min_lag]
    let operand1 = arr.slice(s![min_lag..n]);
    let operand2 = arr.slice(s![..n - min_lag]);
    let result = &operand1 - &operand2;

    result
}

pub fn diff<S: Data<Elem = f64> + RawDataClone, D: Dimension>(
    arr: Array<S, D>,
    lag: Option<u32>,
    differences: Option<u32>,
) -> Result<Array<S, D>> {
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

    let x = check_array().arr(arr).dtype(TypeId::of::<f64>()).call()?;

    if x.ndim() == 1 {
        let mut res = x.into_flat();

        for _ in 1..differences_unwrapped {
            res = _diff_vector(&mut res, lag_unwrapped);

            if res.shape()[0] == 0 {
                Ok(res)
            }
        }
    } else if x.ndim() == 2 {
    }

    Ok(res)
}
