use std::fs;
use std::env::args;
use std::io::Read;

mod float_vec;
mod layer;
mod sigmoid;
mod linear;
mod model;
mod dataset;
mod rand;

use float_vec::FloatVec;
use linear::Linear;
use sigmoid::Sigmoid;
use model::Model;
use rand::Rng;
use dataset::MnistDataset;

struct Checkpoint {
    model: Model,
    epoch: u32
}

const BATCH_SIZE: usize = 64;
const LEARNING_RATE: f32 = 0.1;

impl Checkpoint {
    fn train_loop(&mut self, dataset: &MnistDataset) {
        let mut rng = Rng::default();
        loop {
            for _ in 0..dataset.train_set_len() / BATCH_SIZE {
                let batch = dataset.sample_batch(&mut rng, BATCH_SIZE);
                self.model.train_step(&batch, LEARNING_RATE);
            }

            let mut loss = 0.0;
            let mut correct_predictions = 0;
            for (input, digit) in dataset.test_set() {
                let target_vec = dataset.digit_to_label(digit);
                let pred_vec = self.model.forward(input);
                let (pred, _) = pred_vec.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                if pred == digit {
                    correct_predictions += 1;
                }
                loss += (pred_vec - target_vec).iter().map(|n| n * n).sum::<f32>();
            }
            let test_set_len = dataset.test_set_len();
            loss /= test_set_len as f32;
            let accuracy = correct_predictions as f32 / test_set_len as f32;

            self.epoch += 1;
            self.save();

            println!("[Epoch {}] Loss: {} ({:.1}% accuracy)", self.epoch, loss, accuracy * 100.0);
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
    let mut rng = Rng::default();
    let mut checkpoint = Checkpoint {
        model: model! {
            Linear::random(&mut rng, 28 * 28, 256),
            Sigmoid,
            Linear::random(&mut rng, 256, 10),
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
            let dataset = MnistDataset::load(
                "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"
            );
            checkpoint.train_loop(&dataset);
        },
        "infer" => {
            checkpoint.load(&args.next().expect("Expected path to checkpoint."));
            let image = dataset::load_image(&args.next().expect("Expected path to image."));
            checkpoint.infer(image);
        }
        _ => panic!("Invalid subcommand \"{}\" (train, infer).", subcommand)
    }
}
