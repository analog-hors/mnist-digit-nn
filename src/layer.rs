use std::any::Any;

use super::float_vec::FloatVec;

pub trait Layer {
    /// Evaluate the layer.
    fn forward(&self, input: &FloatVec) -> FloatVec;

    /// How much the input affects the output given some input and the output gradient.
    fn input_gradient(&self, input: &FloatVec, output_gradient: &FloatVec) -> FloatVec;

    /// How much the layer parameters affect the output given some input and the output gradient.
    fn param_gradient(&self, input: &FloatVec, output_gradient: &FloatVec) -> Box<dyn Any + Send>;

    /// Update the parameters given a list of parameter gradients.
    fn apply_gradients(&mut self, gradients: &[Box<dyn Any + Send>], lr: f32);

    fn serialize(&self, buffer: &mut Vec<u8>);

    fn deserialize(&mut self, buffer: &mut &[u8]);
}
