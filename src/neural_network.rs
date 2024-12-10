use activation_functions::f64::sigmoid;
use anyhow::{Error, Result};
use ndarray::{Array, Array2, ShapeError};
use ndarray_rand::rand_distr::num_traits::FromPrimitive;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

/// Represents a neural network.
///
/// # Example
///
/// Training with a small dataset and querying the network.
///
/// ```
/// use ndarray::{Array1, Array2};
/// use ndarray_rand::rand_distr::num_traits::ToPrimitive;
/// use ndarray_stats::QuantileExt;
/// use spt_neural_net::NeuralNetwork;
/// use std::fs::File;
/// use std::io::{BufRead, BufReader};
///
/// // The inputs will be 28x28 images from the MNIST Dataset of Handwritten Numbers
/// let input_nodes = 28 * 28;
/// let hidden_nodes = 100;
/// let output_nodes = 10;
///
/// // Learning rate us 0.3
/// let learning_rate = 0.3;
///
/// let mut network = NeuralNetwork::new(input_nodes, hidden_nodes, output_nodes, learning_rate).unwrap();
///
/// let training_data_file = BufReader::new(File::open("mnist_dataset/mnist_train_100.csv").unwrap());
/// let training_data_list: Vec<String> = training_data_file.lines().collect::<Result<_, _>>().unwrap();
///
/// for record in training_data_list.iter().enumerate() {
///     // Get all values for one image by splitting the csv record values
///     let all_values: Vec<f64> = record
///         .1
///         .split(',')
///         .map(|s| s.parse::<f64>().unwrap())
///         .collect();
///
///     // Scale the input so that the colour values are transformed from 0-255, to 0.01 -> 1.0
///     let inputs = all_values[1..]
///         .iter()
///         .map(|v| (v / 255.0).mul_add(0.99, 0.01))
///         .collect();
///
///     // Target output values can't be 0 or 1.0, so we scale the zeros by adding 0.01 and set the
///     // output node (label) that matches the value the image in the training data represents to
///     // 0.99
///     let mut targets = vec![0.01; output_nodes];
///     targets[all_values[0] as usize] = 0.99;
///
///     network.train(inputs, targets).unwrap();
/// }
///
/// let test_data_file = BufReader::new(File::open("mnist_dataset/mnist_test_10.csv").unwrap());
/// let test_data_list: Vec<String> = test_data_file.lines().collect::<Result<_, _>>().unwrap();
///
/// let all_values: Vec<f64> = test_data_list[0]
///     .split(',')
///     .map(|s| s.parse::<f64>().unwrap())
///     .collect();
///
/// let outputs = network
///     .query(
///         all_values[1..]
///             .iter()
///             .map(|v| (v / 255.0).mul_add(0.99, 0.01))
///             .collect(),
///     )
///     .unwrap();
///
/// let result: i32 = Array1::from_vec(outputs).argmax().unwrap().to_i32().unwrap();
///
/// // The first value in the data set represents a 7
/// assert_eq!(result, 7);
/// ```
#[derive(Debug)]
pub struct NeuralNetwork {
    i_nodes: usize,
    o_nodes: usize,

    lr: f64,

    wih: Array2<f64>,
    who: Array2<f64>,

    activation_function: fn(&f64) -> f64,
}

impl NeuralNetwork {
    /// Creates a new untrained [`NeuralNetwork`].
    ///
    /// Parameters:
    ///
    /// - `i_nodes` the number of input nodes the network should have.
    /// - `h_nodes` the number of hidden node layers.
    /// - `o_nodes` the number of output node.
    /// - `lr` the learning rate.
    ///
    /// # Errors
    ///
    /// Returns `Err` for the following reasons:
    ///
    /// * unable to convert `i_nodes` to `f64`.
    /// * unable to convert `h_nodes` to `f64`.
    /// * unable to convert `o_nodes` to `f64`.
    /// * unable to create a `Normal` distribution.
    ///
    /// # Example
    ///
    /// Creation of a simple neural network.
    ///
    /// ```
    /// use ndarray::Array2;
    /// use spt_neural_net::NeuralNetwork;
    ///
    /// // Number of input, hidden and output nodes
    /// let input_nodes = 3;
    /// let hidden_nodes = 3;
    /// let output_nodes = 3;
    ///
    /// // Learning rate is 0.3
    /// let learning_rate = 0.3;
    ///
    /// let network = NeuralNetwork::new(input_nodes, hidden_nodes, output_nodes, learning_rate).unwrap();
    /// let outputs = network.query(vec![1.0, 0.5, -1.5]).unwrap();
    ///
    /// assert_eq!(outputs.len(), output_nodes);
    /// ```
    pub fn new(i_nodes: usize, h_nodes: usize, o_nodes: usize, lr: f64) -> Result<Self> {
        Ok(Self {
            i_nodes,
            o_nodes,
            lr,
            wih: Array::random(
                (h_nodes, i_nodes),
                Normal::new(
                    0.0,
                    f64::from_usize(i_nodes)
                        .ok_or_else(|| Error::msg(format!("Failed to convert {i_nodes} to f64")))?
                        .powf(-0.5),
                )?,
            ),
            who: Array::random(
                (o_nodes, h_nodes),
                Normal::new(
                    0.0,
                    f64::from_usize(h_nodes)
                        .ok_or_else(|| Error::msg(format!("Failed to convert {h_nodes} to f64")))?
                        .powf(-0.5),
                )?,
            ),
            activation_function: |f| sigmoid(*f),
        })
    }

    /// Provide the neural network with training data.
    ///
    /// - `inputs` the input data.
    /// - `targets` the expected outputs for the `inputs`.
    ///
    /// # Errors
    ///
    /// Returns `Err` for the following reasons:
    ///
    /// * the number of `inputs` does not correspond to the network's number of input nodes.
    /// * the number of `targets` does not correspond to the network's number if output nodes.
    pub fn train(&mut self, inputs: Vec<f64>, targets: Vec<f64>) -> Result<()> {
        let inputs = Array2::from_shape_vec((self.i_nodes, 1), inputs)?;
        let targets = Array2::from_shape_vec((self.o_nodes, 1), targets)?;

        let outputs = self.calculate_outputs(&inputs);

        let output_errors = targets - outputs.final_outputs.clone();
        let hidden_errors = self.who.t().dot(&output_errors);

        self.who += &(self.lr
            * (&output_errors * &outputs.final_outputs * (1.0 - &outputs.final_outputs))
                .dot(&outputs.hidden_outputs.t()));

        self.wih += &(self.lr
            * (&hidden_errors * &outputs.hidden_outputs * (1.0 - &outputs.hidden_outputs))
                .dot(&inputs.t()));

        Ok(())
    }

    /// Queries the neural network.
    ///
    /// - `input` the input data.
    ///
    /// # Errors
    ///
    /// Returns `Err` if inputs does not correspond to the networks number of input nodes.
    pub fn query(&self, inputs: Vec<f64>) -> Result<Vec<f64>, ShapeError> {
        Array2::from_shape_vec((self.i_nodes, 1), inputs)
            .map(|a| self.calculate_outputs(&a))
            .map(|o| o.final_outputs)
            .map(|o| o.into_raw_vec_and_offset().0)
    }

    fn calculate_outputs(&self, inputs: &Array2<f64>) -> Outputs {
        let hidden_inputs = self.wih.dot(inputs);
        let hidden_outputs = hidden_inputs.map(self.activation_function);

        let final_inputs = self.who.dot(&hidden_outputs);
        let final_outputs = final_inputs.map(self.activation_function);

        Outputs {
            hidden_outputs,
            final_outputs,
        }
    }
}

struct Outputs {
    hidden_outputs: Array2<f64>,
    final_outputs: Array2<f64>,
}

#[cfg(test)]
mod test {
    use crate::NeuralNetwork;
    use activation_functions::f64::sigmoid;
    use ndarray::Array2;

    #[test]
    fn train_should_update_the_network_weights() {
        // Given
        //   - a network with known initial weights
        let i_nodes = 3;
        let h_nodes = 1;
        let o_nodes = 3;
        let lr = 0.5;

        let mut target = NeuralNetwork {
            i_nodes,
            o_nodes,
            lr,
            wih: Array2::from_shape_vec((h_nodes, i_nodes), vec![0.9, 0.3, 0.4]).unwrap(),
            who: Array2::from_shape_vec((o_nodes, h_nodes), vec![0.3, 0.7, 0.5]).unwrap(),
            activation_function: |f| sigmoid(*f),
        };

        // When
        //   - the network is trained
        target
            .train(vec![0.9, 0.1, 0.8], vec![0.01, 0.01, 0.99])
            .unwrap();

        // Then
        //   - the weights are updated
        let wih: Vec<String> = target
            .wih
            .into_raw_vec_and_offset()
            .0
            .iter()
            .map(|f| format!("{f:.3}"))
            .collect();

        let who: Vec<String> = target
            .who
            .into_raw_vec_and_offset()
            .0
            .iter()
            .map(|f| format!("{f:.3}"))
            .collect();

        assert_eq!(wih, vec!["0.867", "0.296", "0.371"]);
        assert_eq!(who, vec!["0.249", "0.645", "0.536"]);
    }

    #[test]
    fn query_should_calculate_outputs() {
        // Given
        //   - a network with known weights
        let i_nodes = 3;
        let h_nodes = 3;
        let o_nodes = 3;
        let lr = 0.5;

        let target = NeuralNetwork {
            i_nodes,
            o_nodes,
            lr,
            wih: Array2::from_shape_vec(
                (h_nodes, i_nodes),
                vec![0.9, 0.3, 0.4, 0.2, 0.8, 0.2, 0.1, 0.5, 0.6],
            )
            .unwrap(),
            who: Array2::from_shape_vec(
                (o_nodes, h_nodes),
                vec![0.3, 0.7, 0.5, 0.6, 0.5, 0.2, 0.8, 0.1, 0.9],
            )
            .unwrap(),
            activation_function: |f| sigmoid(*f),
        };

        // When
        //   - the network is queried
        let result: Vec<String> = target
            .query(vec![0.9, 0.1, 0.8])
            .unwrap()
            .iter()
            .map(|f| format!("{f:.3}"))
            .collect();

        // Then
        //   - it returns the expected output values
        assert_eq!(result, vec!["0.726", "0.709", "0.778"]);
    }
}
