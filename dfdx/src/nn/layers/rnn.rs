use crate::prelude::*;
use dfdx_core::tensor_ops::matmul;
use dfdx_core::tensor_ops::TryConv1D;

fn apply_gate<const N: usize, Dt: Dtype, Dev: Device<Dt>>(
    vec: Tensor<Rank1<N>, Dt, Dev>,
    gate: Tensor<Rank1<N>, Dt, Dev>,
) -> Result<Tensor<Rank1<N>, Dt, Dev>, Error> {
    vec.try_mul(gate)
}

type Vector<D, Dt, Dev> = Tensor<(D,), Dt, Dev>;

pub struct LSTMState<H: ConstDim, Dt: Dtype, Dev: Device<Dt>> {
    pub cell: Vector<H, Dt, Dev>,
    pub hidden: Vector<H, Dt, Dev>,
}

pub struct LSTM<InDim: ConstDim, OutDim: Dim, Dt: Dtype, Dev: Device<Dt>> {
    forget_linear: Linear<OutDim, InDim, Dt, Dev>,
    update_linear: Linear<OutDim, InDim, Dt, Dev>,
    output_linear: Linear<OutDim, InDim, Dt, Dev>,
    candidate_linear: Linear<OutDim, InDim, Dt, Dev>,
}

use typenum::Sum;

impl<InputDim: ConstDim, HiddenDim: ConstDim, Dt: Dtype, Dev: Device<Dt>>
    Module<(LSTMState<HiddenDim, Dt, Dev>, Vector<InputDim, Dt, Dev>)>
    for LSTM<InputDim, HiddenDim, Dt, Dev>
{
    type Output = (LSTMState<HiddenDim, Dt, Dev>, Vector<InputDim, Dt, Dev>);

    fn try_forward(
        &self,
        (LSTMState { cell, hidden }, input): (
            LSTMState<HiddenDim, Dt, Dev>,
            Vector<InputDim, Dt, Dev>,
        ),
    ) -> Result<Self::Output, Error> {
        let forget_gate = self
            .forget_linear
            .try_forward(hidden.clone())?
            .try_sigmoid()?;

        //        let update_gate = self
        //            .update_linear
        //            .try_forward(hidden.clone())?
        //            .try_sigmoid()?;
        //        let output_gate = self.output_linear.try_forward(hidden)?.try_sigmoid()?;
        //
        //        let candidate_cell = self.candidate_linear.try_forward(input)?.try_tanh()?;

        todo!()
    }

    //    fn forward(&self, input: Self::Input, _: ()) -> Result<Self::Output, Error> {
    //        let LSTMState { cell, hidden } = input;
    //        let forget_gate = self.forget_linear.forward(hidden.clone(), ())?;
    //        let update_gate = self.update_linear.forward(hidden, ())?;
    //        let output_gate = self.output_linear.forward(hidden, ())?;
    //        let forget_gate = forget_gate.sigmoid()?;
    //        let update_gate = update_gate.sigmoid()?;
    //        let output_gate = output_gate.tanh()?;
    //        let cell = apply_gate(cell, forget_gate)?;
    //        let cell = apply_gate(cell, update_gate)?;
    //        let hidden = cell.clone().tanh()?;
    //        let hidden = apply_gate(hidden, output_gate)?;
    //        Ok(LSTMState { cell, hidden })
    //    }
}
