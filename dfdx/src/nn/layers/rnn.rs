use crate::prelude::*;

pub struct LSTMState<H: ConstDim, Dt: Dtype, Dev: Device<Dt>, T: Tape<Dt, Dev>> {
    pub cell: Tensor<(H,), Dt, Dev, T>,
    pub hidden: Tensor<(H,), Dt, Dev, T>,
}

#[derive(Debug, Clone)]
pub struct LSTM<InDim: ConstDim, OutDim: ConstDim, Dt: Dtype, Dev: Device<Dt>> {
    pub forget_linear: Linear<InDim, OutDim, Dt, Dev>,
    pub update_linear: Linear<InDim, OutDim, Dt, Dev>,
    pub output_linear: Linear<InDim, OutDim, Dt, Dev>,
    pub candidate_linear: Linear<InDim, OutDim, Dt, Dev>,
}

impl<InputDim: ConstDim, HiddenDim: ConstDim, Dt: Dtype, Dev: Device<Dt>, T: Tape<Dt, Dev>>
    Module<(
        LSTMState<HiddenDim, Dt, Dev, T>,
        Tensor<(InputDim,), Dt, Dev, T>,
    )> for LSTM<Const<{ InputDim::SIZE + HiddenDim::SIZE }>, HiddenDim, Dt, Dev>
where
    ((HiddenDim,), (InputDim,)): TryConcatAlong<Axis<0>>,
    <((HiddenDim,), (InputDim,)) as TryConcatAlong<Axis<0>>>::Output: Shape,
    Linear<Const<{ InputDim::SIZE + HiddenDim::SIZE }>, HiddenDim, Dt, Dev>: Module<
        Tensor<<((HiddenDim,), (InputDim,)) as TryConcatAlong<Axis<0>>>::Output, Dt, Dev>,
        Output = Tensor<(HiddenDim,), Dt, Dev, T>,
    >,
{
    type Output = LSTMState<HiddenDim, Dt, Dev, T>;

    fn try_forward(
        &self,
        (LSTMState { cell, hidden }, input): (
            LSTMState<HiddenDim, Dt, Dev, T>,
            Tensor<(InputDim,), Dt, Dev, T>,
        ),
    ) -> Result<Self::Output, Error> {
        let concat = (hidden, input).concat_along(Axis::<0>);

        let forget_gate = self
            .forget_linear
            .try_forward(concat.retaped())?
            .try_sigmoid()?;
        let update_gate = self
            .update_linear
            .try_forward(concat.retaped())?
            .try_sigmoid()?;
        let output_gate = self
            .output_linear
            .try_forward(concat.retaped())?
            .try_sigmoid()?;

        let candidate_cell = self
            .candidate_linear
            .try_forward(concat.retaped())?
            .try_tanh()?;

        let cell = candidate_cell
            .try_mul(update_gate)?
            .try_add(cell.try_mul(forget_gate)?)?;

        let hidden = output_gate.try_mul(cell.retaped::<T>())?;

        let state = LSTMState { cell, hidden };

        Ok(state)
    }
}

use std::marker::PhantomData;

#[derive(Clone, Copy, Debug, Default)]
pub struct LSTMConfig<I: ConstDim, O: ConstDim> {
    inp: PhantomData<I>,
    out: PhantomData<O>,
}

impl<const IN_DIM: usize, const OUT_DIM: usize, Dt: Dtype, Dev: Device<Dt>> BuildOnDevice<Dt, Dev>
    for LSTMConfig<Const<IN_DIM>, Const<OUT_DIM>>
{
    type Built = LSTM<Const<IN_DIM>, Const<OUT_DIM>, Dt, Dev>;

    fn try_build_on_device(&self, device: &Dev) -> Result<Self::Built, dfdx_core::tensor::Error> {
        let linear_config = LinearConstConfig::<IN_DIM, OUT_DIM>::default();

        let zero_tensor = linear_config.try_build_on_device(device)?;

        let lstm = LSTM {
            forget_linear: zero_tensor.clone(),
            update_linear: zero_tensor.clone(),
            output_linear: zero_tensor.clone(),
            candidate_linear: zero_tensor.clone(),
        };

        Ok(lstm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm() {
        let dev = AutoDevice::default();

        let linear_config = LinearConstConfig::<4, 2>::default();

        let zero_tensor: Linear<_, _, f64, _> = linear_config.try_build_on_device(&dev).unwrap();

        let lstm = LSTM {
            forget_linear: zero_tensor.clone(),
            update_linear: zero_tensor.clone(),
            output_linear: zero_tensor.clone(),
            candidate_linear: zero_tensor.clone(),
        };

        let state = LSTMState {
            cell: dev.try_tensor(&[0.0, 0.0]).unwrap(),
            hidden: dev.try_tensor(&[0.0, 0.0]).unwrap(),
        };

        let input = dev.try_tensor(&[0.0, 0.0]).unwrap();

        let LSTMState { cell, hidden } = lstm.try_forward((state, input)).unwrap();

        println!("cell: {:?}, hidden: {:?}", cell, hidden);
    }
}
