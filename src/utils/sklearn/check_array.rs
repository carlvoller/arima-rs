use std::any::TypeId;

use anyhow::{bail, Result};
use bon::builder;
use ndarray::{ArrayBase, Data, Dimension, RawDataClone};
use num_traits::Float;

use crate::errors::ArimaError;

// TODO: Add ensure_min_samples, ensure_min_features support
#[builder]
pub fn check_array<T: 'static + Float, S: Data<Elem = T> + RawDataClone, D: Dimension>(
    arr: ArrayBase<S, D>,
    ensure_2d: Option<bool>,
    force_all_finite: Option<bool>,
    dtype: Option<TypeId>,
    copy: Option<bool>,
) -> Result<ArrayBase<S, D>> {
    let final_arr: ArrayBase<S, D> = {
        if copy.is_some_and(|x| x) {
            arr.clone()
        } else {
            arr
        }
    };

    if force_all_finite.is_some_and(|x| x) {
        let contains_any_invalid = final_arr.is_any_infinite() || final_arr.is_any_nan();

        if contains_any_invalid {
            bail!(ArimaError::InfiniteValueFound);
        }
    }

    if ensure_2d.is_some_and(|x| x) {
        if final_arr.ndim() != 2 {
            bail!(ArimaError::Non2DArrayFound);
        }
    }

    if dtype.is_some() {
        let dtype_unwrapped = dtype.unwrap();
        if TypeId::of::<T>() != dtype_unwrapped {
            bail!(ArimaError::ValueError {
                expected: String::from("dtype to match ndarray type"),
                found: String::from("dtype does not match ndarry type")
            })
        }
    }

    Ok(final_arr)
}
