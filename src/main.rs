use anyhow::{Error, Result};
use chrono::Utc;
use ndarray::Array1;
use ndarray_rand::rand_distr::num_traits::{FromPrimitive, ToPrimitive};
use ndarray_stats::QuantileExt;
use spt_neural_net::{NeuralNetwork, NumericImage, IMAGE_HEIGHT, IMAGE_WIDTH};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::FromStr;

const TRAINING_DATA: &str = "mnist_dataset/mnist_train_100.csv";
const TEST_DATA: &str = "mnist_dataset/mnist_test_10.csv";

// The inputs will be numeric images of dimensions IMAGE_WIDTH x IMAGE_HEIGHT
const INPUT_NODES: usize = IMAGE_WIDTH * IMAGE_HEIGHT;

// Hidden nodes should be less than the input but not so small as to restrict the ability of the
// network
const HIDDEN_NODES: usize = 300;

// We are determining what digit the images represent, therefore there are 10 possibilities (or
// labels)
const OUTPUT_NODES: usize = 10;
const LEARNING_RATE: f64 = 0.1;

const EPOCHS: i32 = 10;

const TRAINING_DATA_ROTATION: f32 = 0.174_533; // 10 degrees

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let display_images = if args.len() > 1 {
        bool::from_str(&args[1]).unwrap_or(false)
    } else {
        false
    };

    let mut network = NeuralNetwork::new(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)?;

    println!("{:?} Training the network...", Utc::now().naive_local());

    // Train the neural network
    for _ in [..EPOCHS] {
        let training_data_file = BufReader::new(File::open(TRAINING_DATA)?);

        for record in training_data_file.lines().enumerate() {
            let image = NumericImage::parse(record.1?)?;

            // Target output values can't be 0 or 1.0, so we scale the zeros by adding 0.01 and set
            // the output node (label) that matches the value the image in the training data
            // represents to 0.99
            let mut targets = vec![0.01; OUTPUT_NODES];
            targets[image.numeric_value() as usize] = 0.99;

            let rotated_clockwise_image = image.rotate(TRAINING_DATA_ROTATION)?;
            let rotated_anti_clockwise_image = image.rotate(-TRAINING_DATA_ROTATION)?;

            if display_images {
                image.display()?;
                rotated_clockwise_image.display()?;
                rotated_anti_clockwise_image.display()?;
            }
            network.train(image.scaled_data(), targets.clone())?;
            network.train(rotated_clockwise_image.scaled_data(), targets.clone())?;
            network.train(rotated_anti_clockwise_image.scaled_data(), targets.clone())?;

            if (record.0 + 1) % 10000 == 0 {
                println!(
                    "{:?} Network trained with {} records...",
                    Utc::now().naive_local(),
                    record.0 + 1
                );
            }
        }
    }

    println!("{:?} Testing the network...", Utc::now().naive_local());

    let mut score_card = vec![];

    for record in BufReader::new(File::open(TEST_DATA)?).lines() {
        let image = NumericImage::parse(record?)?;

        if display_images {
            image.display()?;
        }

        let outputs = network.query(image.scaled_data())?;
        let result = Array1::from_vec(outputs).argmax()?.to_u8().unwrap();

        if result == image.numeric_value() {
            score_card.push(1.0);
        } else {
            score_card.push(0.0);
        }
    }

    println!(
        "{:?} Network performance: {:.2}",
        Utc::now().naive_local(),
        score_card.iter().sum::<f64>()
            / f64::from_usize(score_card.len()).ok_or_else(|| Error::msg(format!(
                "Failed to convert {} to f64",
                score_card.len()
            )))?
    );

    Ok(())
}
