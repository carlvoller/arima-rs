use arimars_utils::arrays::diff;
use faer::prelude::*;
use faer_ext::*;
use ndarray::{concatenate, s, Array1, Array2, ArrayView2, Axis, NewAxis};

pub struct AugmentedDickeyFullerTest {
    alpha: f64,
    k: Option<i32>,
}

impl AugmentedDickeyFullerTest {
    fn _embed(&self, x: &Array1<f64>, k: i32) -> Array2<f64> {
        let n = x.shape()[0];
        if k as usize > n {
            panic!("k cannot exceed y dim");
        }

        let mut rows = Array2::<f64>::zeros((k as usize, n - (k as usize - 1)));

        let mut iter = ((k - 1)..-1).enumerate();
        while let Some((i, j)) = iter.next() {
            rows.row_mut(i).as_mut_ptr();
            x.slice(s![j as usize..(n - i)]).assign_to(rows.row_mut(i));
        }

        rows
    }

    fn _ols(&self, x: &Array1<f64>, n: usize, z: ArrayView2<f64>, k: i32) -> Array1<f64> {
        let yt = z.column(0);
        let mut tt = Array1::range((k - 1) as f64, n as f64, 1.);
        let xt1 = x.select(Axis(0), &tt.mapv(|x| x as usize).to_vec());

        tt.mapv_inplace(|x| x + 1.);

        let _n = xt1.shape()[0];

        let mut x_stacked = concatenate![
            Axis(1),
            Array2::ones((_n, 1)),
            xt1.to_shape((_n, 1)).ok().unwrap(),
            tt.to_shape((_n, 1)).ok().unwrap()
        ];

        if k > 1 {
            let yt1 = z.slice(s![.., 1..k]);
            x_stacked = concatenate![Axis(1), x_stacked, yt1];
        }

        // Perform OLS using QR method
        let x_faer = x_stacked.view().into_faer();
        let y_faer = yt.slice(s![.., NewAxis]).into_faer();
        let coefficients = x_faer.col_piv_qr().solve_lstsq(&y_faer);
        coefficients
            .as_ref()
            .into_ndarray()
            .slice(s![.., 0])
            .to_owned()
    }

    pub fn should_diff(&self, x: &Array1<f64>) {
        let k = self
            .k
            .unwrap_or_else(|| (x.shape()[0] as f64 - 1.0).powf(1.0 / 3.0) as i32)
            + 1;

        let x_copied = x.to_owned();

        let y = diff(x_copied, Some(1), Some(1));
        let n = y.shape()[0];
        let z = self._embed(x, k);
        let z_transposed = z.t();

        let res = self._ols(x, n, z_transposed, k);
    }
}
