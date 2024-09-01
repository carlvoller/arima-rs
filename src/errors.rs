use thiserror::Error;

#[derive(Error, Debug)]
pub enum ArimaError {
    #[error("a non-finite value was found")]
    InfiniteValueFound,
    #[error("a 2d array is required")]
    Non2DArrayFound,
    #[error("a 1d array is required")]
    Non1DArrayFound,
    #[error("invalid value (expected {expected:?}, found {found:?})")]
    ValueError {
        expected: String,
        found: String,
    }
}
