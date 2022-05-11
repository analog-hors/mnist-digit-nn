use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;

use image::io::Reader as ImageReader;
use image::imageops::FilterType;

use crate::float_vec::FloatVec;
use crate::rand::Rng;

macro_rules! read {
    ($src:expr, $type:ty) => {{
        let mut buf = <$type>::to_ne_bytes(0);
        $src.read_exact(&mut buf).unwrap();
        <$type>::from_be_bytes(buf)
    }};
}

pub struct MnistDataset {
    train_set: [Vec<FloatVec>; 10],
    test_set: [Vec<FloatVec>; 10],
    labels: [FloatVec; 10]
}

impl MnistDataset {
    pub fn load(
        train_images: &str,
        train_labels: &str,
        test_images: &str,
        test_labels: &str
    ) -> Self {
        let mut labels = [(); 10].map(|_| FloatVec::all(0.0, 10));
        for i in 0..labels.len() {
            labels[i][i] = 1.0;
        }
        Self {
            train_set: load_mnist(train_images, train_labels),
            test_set: load_mnist(test_images, test_labels),
            labels
        }
    }

    pub fn digit_to_label(&self, digit: usize) -> &FloatVec {
        &self.labels[digit]
    }

    pub fn train_set_len(&self) -> usize {
        self.train_set.iter().map(Vec::len).sum()
    }

    pub fn test_set_len(&self) -> usize {
        self.test_set.iter().map(Vec::len).sum()
    }

    pub fn sample_batch(&self, rng: &mut Rng, size: usize) -> Vec<(&FloatVec, &FloatVec)> {
        (0..size)
            .map(|_| {
                let digit = (rng.next() * self.train_set.len() as f64) as usize;
                let label = self.digit_to_label(digit);
                let digits = &self.train_set[digit];
                let input = &digits[(rng.next() * digits.len() as f64) as usize];
                (input, label)
            })
            .collect()
    }

    pub fn test_set(&self) -> impl Iterator<Item=(&'_ FloatVec, usize)> {
        self.test_set
            .iter()
            .enumerate()
            .flat_map(|(digit, inputs)| {
                inputs.iter().map(move |input| (input, digit))
            })
    }
}

fn load_mnist(images: &str, labels: &str) -> [Vec<FloatVec>; 10] {
    let mut images = BufReader::new(File::open(images).unwrap());
    let mut labels = BufReader::new(File::open(labels).unwrap());
    assert_eq!(read!(images, u32), 2051);
    assert_eq!(read!(labels, u32), 2049);
    let image_count = read!(images, u32);
    assert_eq!(image_count, read!(labels, u32));
    let width = read!(images, u32);
    let height = read!(images, u32);
    let image_size = width * height;
    let mut dataset = [(); 10].map(|_| Vec::new());
    for _ in 0..image_count {
        let mut image = Vec::with_capacity(image_size as usize);
        for _ in 0..image_size {
            image.push(read!(images, u8) as f32 / u8::MAX as f32);
        }
        let label = read!(labels, u8) as usize;
        dataset[label].push(image.into());
    }
    dataset
}

pub fn load_image(path: &str) -> FloatVec {
    let image = ImageReader::open(path).unwrap().decode().unwrap();
    let image = image.resize_exact(28, 28, FilterType::Lanczos3);
    let image = image.to_luma8();
    image.pixels().map(|p| p.0[0] as f32 / u8::MAX as f32).collect::<Vec<_>>().into()
}
