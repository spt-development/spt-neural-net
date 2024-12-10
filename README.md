# spt-neural-net

Native Rust implementation of the Neural Network defined in Tariq Rashid's excellent book
[\"Make Your Own Neural Network\"](https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/tree/master).

## Pre-requisites

The application takes a single (optional) boolean argument that results in each of the images used
for training and testing the [NeuralNetwork](src/neural_network.rs) being displayed - this is
useful when debugging the application. The display of the images uses the
[matplotlib](https://crates.io/crates/matplotlib) crate, which simply wraps the Python library of
the same name. In order therefore, for the image display to work, the following Python library
pre-requisites must be installed.

```shell
$ pip install --user matplotlib
$ pip install --user PyQt5
```

## Building and running the application

The standard `cargo` commands can be used for building, testing and running the application.

```shell
$ cargo clippy
$ cargo test
$ cargo build --release
$ cargo run --release
```

Example output when running the application.

```shell
$  cargo run --release
    Finished `release` profile [optimized] target(s) in 0.13s
     Running `target/release/spt-neural-net`
2024-12-15T17:13:51.391128 Training the network...
2024-12-15T17:14:13.786799 Network trained with 10000 records...
2024-12-15T17:14:35.126296 Network trained with 20000 records...
2024-12-15T17:14:55.012176 Network trained with 30000 records...
2024-12-15T17:15:15.502282 Network trained with 40000 records...
2024-12-15T17:15:35.603023 Network trained with 50000 records...
2024-12-15T17:15:55.829038 Network trained with 60000 records...
2024-12-15T17:15:55.829095 Testing the network...
2024-12-15T17:15:57.872578 Network performance: 0.97
```

**NOTE** In order to get results in the region of 97% as shown in the example above, it will be
necessary to train the neural network with a larger data set than
[mnist_train_100.csv](mnist_dataset/mnist_train_100.csv); the full dataset is available
[here](https://pjreddie.com/projects/mnist-in-csv/).