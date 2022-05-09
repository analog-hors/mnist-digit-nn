use std::any::Any;

use super::float_vec::FloatVec;
use super::layer::Layer;

#[derive(Debug, Clone)]
pub struct Sigmoid;

impl Layer for Sigmoid {
    fn forward(&self, input: &FloatVec) -> FloatVec {
        1.0 / (1.0 + (-input).exp())
    }

    fn input_gradient(&self, input: &FloatVec, output_gradient: &FloatVec) -> FloatVec {
        let sigmoid = self.forward(input);
        &sigmoid * (1.0 - &sigmoid) * output_gradient
    }

    fn param_gradient(&self, _input: &FloatVec, _output_gradient: &FloatVec) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn apply_gradients(&mut self, _gradients: &[Box<dyn Any + Send>], _lr: f32) {}

    fn serialize(&self, _buffer: &mut Vec<u8>) {}

    fn deserialize(&mut self, _buffer: &mut &[u8]) {}
}
