use std::any::TypeId;

use anyhow::{bail, Result};
use bon::builder;
use ndarray::{Array, ArrayBase, Data, Dimension, RawDataClone};
use num_traits::Float;

use crate::errors::ArimaError;

#[builder]
pub fn check_array<T: 'static + Float, S: Data<Elem = T + RawDataClone> + RawDataClone, D: Dimension>(
    arr: Array<S, D>,
    ensure_2d: Option<bool>,
    force_all_finite: Option<bool>,
    dtype: Option<TypeId>,
    copy: Option<bool>,
) -> Result<Array<S, D>> {
    // Not exposing these options default options from sklearn
    // as pmdarima always uses these default values
    let ensure_min_samples = 1;
    let ensure_min_features = 1;

    let final_arr: Array<S, D> = {
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

    // I still do these checks despite ensure_min_samples and ensure_min_features being constant
    // in the event that I choose to expose these parameters in the future.
    if ensure_min_samples > 0 {
        let shape = final_arr.shape();
        if shape.len() < 1 || shape[0] < 1 {
            bail!(ArimaError::ValueError {
                expected: String::from("an array with > 0 samples"),
                found: String::from("array with < 1 samples")
            })
        }
    }

    if ensure_min_features > 0 && final_arr.ndim() == 2 {
        let shape = final_arr.shape();
        let features = shape[1];
        if features < ensure_min_features {
            bail!(ArimaError::ValueError {
                expected: format!("a 2d array with at least {ensure_min_features}"),
                found: format!("an array with {features}")
            })
        }
    }

    Ok(final_arr)
}
