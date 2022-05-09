use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;

use image::io::Reader as ImageReader;
use image::imageops::FilterType;

use crate::float_vec::FloatVec;

macro_rules! read {
    ($src:expr, $type:ty) => {{
        let mut buf = <$type>::to_ne_bytes(0);
        $src.read_exact(&mut buf).unwrap();
        <$type>::from_be_bytes(buf)
    }};
}

pub fn load_mnist(images: &str, labels: &str) -> Vec<(FloatVec, FloatVec)> {
    let mut images = BufReader::new(File::open(images).unwrap());
    let mut labels = BufReader::new(File::open(labels).unwrap());
    assert_eq!(read!(images, u32), 2051);
    assert_eq!(read!(labels, u32), 2049);
    let image_count = read!(images, u32);
    assert_eq!(image_count, read!(labels, u32));
    let width = read!(images, u32);
    let height = read!(images, u32);
    let image_size = width * height;
    let mut dataset = Vec::with_capacity(image_count as usize);
    for _ in 0..image_count {
        let mut image = Vec::with_capacity(image_size as usize);
        for _ in 0..image_size {
            image.push(read!(images, u8) as f32 / u8::MAX as f32);
        }
        let mut label = vec![0.0; 10];
        label[read!(labels, u8) as usize] = 1.0;
        dataset.push((image.into(), label.into()));
    }
    dataset
}

pub fn load_image(path: &str) -> FloatVec {
    let image = ImageReader::open(path).unwrap().decode().unwrap();
    let image = image.resize_exact(28, 28, FilterType::Lanczos3);
    let image = image.to_luma8();
    image.pixels().map(|p| p.0[0] as f32 / u8::MAX as f32).collect::<Vec<_>>().into()
}
