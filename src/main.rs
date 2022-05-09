use std::fs;
use std::env::args;
use std::io::Read;

mod float_vec;
mod layer;
mod sigmoid;
mod linear;
mod model;
mod dataset;

use float_vec::FloatVec;
use linear::Linear;
use sigmoid::Sigmoid;
use model::Model;

struct Checkpoint {
    model: Model,
    epoch: u32
}

impl Checkpoint {
    fn train_loop(&mut self) {
        let train_set = dataset::load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
        let test_set = dataset::load_mnist("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

        loop {
            for batch in train_set.chunks(32) {
                self.model.train_step(batch, 1.0);
            }

            let mut loss = 0.0;
            for (input, target) in &test_set {
                let pred = self.model.forward(input);
                loss += (pred - target).iter().map(|n| n * n).sum::<f32>();
            }
            loss /= test_set.len() as f32;

            self.epoch += 1;
            self.save();

            println!("[Epoch {}] Loss: {}", self.epoch, loss);
        }
    }

    fn infer(&self, image: FloatVec) {
        let output = self.model.forward(&image);
        let mut max_digit = 0;
        let mut max_output = -1.0;
        for (digit, output) in output.iter().enumerate() {
            println!("{}: {}", digit, output);
            if output > max_output {
                max_digit = digit;
                max_output = output;
            }
        }
        println!("Classified as {} ({}).", max_digit, max_output);
    }

    fn save(&self) {
        let mut buffer = Vec::new();
        self.model.serialize(&mut buffer);
        buffer.extend_from_slice(&self.epoch.to_le_bytes());
        fs::create_dir_all("checkpoints").unwrap();
        fs::write(format!("checkpoints/epoch-{}", self.epoch), &buffer).unwrap();
    }

    fn load(&mut self, path: &str) {
        let buffer = fs::read(path).unwrap();
        let mut buffer = &buffer[..];
        self.model.deserialize(&mut buffer);
        let mut epoch = 0u32.to_ne_bytes();
        buffer.read_exact(&mut epoch).unwrap();
        self.epoch = u32::from_le_bytes(epoch);
    }
}

macro_rules! model {
    ($($layer:expr),*) => {
        Model {
            layers: vec![$(Box::new($layer)),*]
        }
    }
}

fn main() {
    let mut state = 0xB57D6A35CFD25BDBD774231501440A94;
    let mut checkpoint = Checkpoint {
        model: model! {
            Linear::random(&mut state, 28 * 28, 128),
            Sigmoid,
            Linear::random(&mut state, 128, 10),
            Sigmoid
        },
        epoch: 0
    };

    let mut args = args().skip(1);
    let subcommand = args.next().expect("Expected subcommand (train, infer).");
    match &*subcommand {
        "train" => {
            if let Some(path) = args.next() {
                checkpoint.load(&path);
                println!("Loaded checkpoint \"{}\".", path);
            }
            checkpoint.train_loop();
        },
        "infer" => {
            checkpoint.load(&args.next().expect("Expected path to checkpoint."));
            let image = dataset::load_image(&args.next().expect("Expected path to image."));
            checkpoint.infer(image);
        }
        _ => panic!("Invalid subcommand \"{}\" (train, infer).", subcommand)
    }
}
