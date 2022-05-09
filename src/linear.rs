use std::any::Any;

use serde::{Serialize, Deserialize};

use super::layer::Layer;
use super::float_vec::FloatVec;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    inputs: usize,
    weights: Vec<FloatVec>,
    biases: FloatVec
}

impl Linear {
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    pub fn outputs(&self) -> usize {
        self.biases.len()
    }

    pub fn random(state: &mut u128, inputs: usize, outputs: usize) -> Self {
        let mut rand = || {
            *state = state.wrapping_mul(0xDA942042E4DD58B5);
            let next = (*state >> (u128::BITS - u64::BITS)) as u64;
            next as u32 as f32 / u32::MAX as f32 - 0.5
        };
        let mut weights = (0..outputs).map(|_| FloatVec::all(0.0, inputs)).collect::<Vec<_>>();
        for weights in &mut weights {
            for weight in weights.iter_mut() {
                *weight = rand();
            }
        }
        let mut biases = FloatVec::all(0.0, outputs);
        for bias in biases.iter_mut() {
            *bias = rand();
        }
        Self {
            inputs,
            weights,
            biases
        }
    }
}

type Gradient = (Vec<FloatVec>, FloatVec);

impl Layer for Linear {
    fn forward(&self, input: &FloatVec) -> FloatVec {
        assert_eq!(input.len(), self.inputs());
        let mut output = self.biases.clone();
        for (output, weights) in output.iter_mut().zip(self.weights.iter()) {
            *output += weights.dot(input);
        }
        output
    }

    fn input_gradient(&self, _input: &FloatVec, output_gradient: &FloatVec) -> FloatVec {
        let mut gradients = FloatVec::all(0.0, self.inputs());
        for (weights, gradient) in self.weights.iter().zip(output_gradient.iter()) {
            gradients += weights * gradient;
        }
        gradients
    }

    fn param_gradient(&self, input: &FloatVec, output_gradient: &FloatVec) -> Box<dyn Any + Send> {
        let mut weight_gradients = (0..self.outputs()).map(|_| input.clone()).collect::<Vec<_>>();
        for (gradient, output_gradient) in weight_gradients.iter_mut().zip(output_gradient.iter()) {
            *gradient *= output_gradient;
        }
        let bias_gradient = output_gradient.clone();
        Box::new((weight_gradients, bias_gradient))
    }

    fn apply_gradients(&mut self, gradients: &[Box<dyn Any + Send>], lr: f32) {
        let mut weight_gradients = (0..self.outputs()).map(|_| FloatVec::all(0.0, self.inputs())).collect::<Vec<_>>();
        let mut bias_gradient = FloatVec::all(0.0, self.outputs());
        for gradient in gradients {
            let (wg, bg) = gradient.downcast_ref::<Gradient>().unwrap();
            for (weight_gradient, wg) in weight_gradients.iter_mut().zip(wg) {
                *weight_gradient += wg;
            }
            bias_gradient += bg;
        }
        for (weight, weight_gradient) in self.weights.iter_mut().zip(weight_gradients) {
            *weight -= weight_gradient / gradients.len() as f32 * lr;
        }
        self.biases -= bias_gradient / gradients.len() as f32 * lr;
    }

    fn serialize(&self, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(&bincode::serialize(self).unwrap());
    }

    fn deserialize(&mut self, buffer: &mut &[u8]) {
        *self = bincode::deserialize_from(buffer).unwrap();
    }
}
