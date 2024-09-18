use ndarray::{s, Array1};

pub fn diff(x: Array1<f64>, lag: Option<usize>, difference: Option<usize>) -> Array1<f64> {
    let lag_val = lag.unwrap_or(1);
    let difference_val = difference.unwrap_or(1);

    if lag_val == 0 || difference_val == 0 {
        panic!("diff(): lag and difference need to be >0");
    }

    let x_len = x.len();
    let eventual_shape = x_len.checked_sub(lag_val * difference_val);

    if eventual_shape.is_none() {
        return Array1::from(vec![]);
    }

    let shape = eventual_shape.unwrap();
    let mut temp_array = Array1::<f64>::from(x);

    for _ in 0..difference_val {
        for j in 0..x_len {
            if j + lag_val == x_len {
                break;
            }
            let curr = temp_array[j];
            let next = temp_array[j + lag_val];
            temp_array[j] = next - curr;
        }
    }

    temp_array.slice_mut(s![..shape]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_diff() {
        // Test lag == 1, diff == 1
        assert_eq!(
            diff(array![10., 4., 2., 9., 34.], Some(1), Some(1)),
            array![-6., -2., 7., 25.]
        );

        // Test lag == 3, diff == 1
        assert_eq!(
            diff(array![10., 4., 2., 9., 34.], Some(3), Some(1)),
            array![-1., 30.]
        );

        // Test lag == 1, diff == 2
        assert_eq!(
            diff(array![10., 4., 2., 9., 34.], Some(1), Some(2)),
            array![4., 9., 18.]
        );

        // Test lag == 2, diff == 2
        assert_eq!(
            diff(array![10., 4., 2., 9., 34.], Some(2), Some(2)),
            array![40.]
        );
    }
}
