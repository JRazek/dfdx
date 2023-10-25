use crate::prelude::*;

/// Calls [crate::tensor_ops::ln()].
#[derive(Default, Debug, Clone, Copy, CustomModule)]
pub struct Ln;
impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for Ln {
    type Output = Tensor<S, E, D, T>;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        x.try_ln()
    }
}
