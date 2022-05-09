use std::any::Any;

use rayon::prelude::*;

use crate::float_vec::FloatVec;
use crate::layer::Layer;

pub struct Model {
    pub layers: Vec<Box<dyn Layer + Sync>>
}

impl Model {
    pub fn forward(&self, input: &FloatVec) -> FloatVec {
        self.layers.iter().fold(input.clone(), |i, l| l.forward(&i))
    }

    pub fn train_step(&mut self, batch: &[(FloatVec, FloatVec)], lr: f32) {
        let raw_gradients = batch.par_iter()
            .map(|(input, target)| {
                let mut gradients = Vec::with_capacity(self.layers.len());
                self.collect_gradients(0, &mut gradients, input, target);
                gradients
            })
            .collect::<Vec<_>>();

        let mut gradients = (0..self.layers.len())
            .map(|_| Vec::with_capacity(raw_gradients.len()))
            .collect::<Vec<_>>();
        for g in raw_gradients {
            for (dest, src) in gradients.iter_mut().zip(g.into_iter().rev()) {
                dest.push(src);
            }
        }

        for (layer, gradients) in self.layers.iter_mut().zip(&gradients) {
            layer.apply_gradients(gradients, lr);
        }
    }

    fn collect_gradients(&self, index: usize, gradients: &mut Vec<Box<dyn Any + Send>>, input: &FloatVec, target: &FloatVec) -> FloatVec {
        if let Some(layer) = self.layers.get(index) {
            let output = layer.forward(input);
            let output_gradient = self.collect_gradients(index + 1, gradients, &output, target);
            gradients.push(layer.param_gradient(input, &output_gradient));
            layer.input_gradient(input, &output_gradient)
        } else {
            input - target
        }
    }

    pub fn serialize(&self, buffer: &mut Vec<u8>) {
        for layer in &self.layers {
            layer.serialize(buffer);
        }
    }

    pub fn deserialize(&mut self, buffer: &mut &[u8]) {
        for layer in &mut self.layers {
            layer.deserialize(buffer);
        }
    }
}
