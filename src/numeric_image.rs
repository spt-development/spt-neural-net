use anyhow::{Error, Result};
use imageproc::geometric_transformations::{rotate, Interpolation};
use imageproc::image::{GrayImage, Luma};
use matplotlib::{commands as c, MatplotlibOpts, Mpl, Run};
use ndarray::Array;
use ndarray_rand::rand_distr::num_traits::FromPrimitive;

pub const IMAGE_WIDTH: usize = 28;
pub const IMAGE_HEIGHT: usize = 28;

#[derive(Debug)]
pub struct NumericImage {
    numeric_value: u8,
    raw_data: Vec<f64>,
}

impl NumericImage {
    /// Parses a CSV record representing an 28 X 28 pixel image of a numeric value where the first
    /// column of the CSV is the value that the image represents.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if the CSV record can not be parsed.
    pub fn parse<T>(csv: T) -> Result<Self>
    where
        T: Into<String>,
    {
        let all_values: Vec<f64> = csv
            .into()
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()?;

        if all_values.len() != IMAGE_WIDTH * IMAGE_HEIGHT + 1 {
            return Err(Error::msg(format!(
                "CSV record does not represent a {IMAGE_WIDTH} x {IMAGE_HEIGHT} image"
            )));
        }

        let numeric_value = all_values[0];

        Ok(Self {
            numeric_value: u8::from_f64(numeric_value)
                .ok_or_else(|| Error::msg(format!("Failed to convert {numeric_value} to u8")))?,
            raw_data: all_values[1..].to_vec(),
        })
    }

    // Returns the image dimensions.
    #[must_use]
    pub const fn dimensions(&self) -> (usize, usize) {
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    }

    /// Returns the numeric value represented by the image data.
    #[must_use]
    pub const fn numeric_value(&self) -> u8 {
        self.numeric_value
    }

    /// Returns the raw image data.
    #[must_use]
    pub fn raw_data(&self) -> Vec<f64> {
        self.raw_data.clone()
    }

    /// Returns the image data transformed from 0-255, to 0.01 -> 1.0, for use with the
    /// `NeuralNetwork`.
    #[must_use]
    pub fn scaled_data(&self) -> Vec<f64> {
        self.raw_data
            .iter()
            .map(|v| (v / 255.0).mul_add(0.99, 0.01))
            .collect()
    }

    /// Returns a new `NumericImageData` with the image data representing the original image,
    /// rotated by theta.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if the `raw_data` can not be converted to a `GrayImage` or any of the
    /// conversions between numeric types fail.
    pub fn rotate(&self, theta: f32) -> Result<Self> {
        let image = GrayImage::from_raw(
            u32::from_usize(IMAGE_WIDTH)
                .ok_or_else(|| Error::msg(format!("Failed to convert {IMAGE_WIDTH} to u32")))?,
            u32::from_usize(IMAGE_HEIGHT)
                .ok_or_else(|| Error::msg(format!("Failed to convert {IMAGE_HEIGHT} to u32")))?,
            self.raw_data()
                .iter()
                .map(|p| {
                    u8::from_f64(*p).ok_or_else(|| {
                        Error::msg(format!("Failed to convert RGB value: {p} to u8"))
                    })
                })
                .collect::<Result<_, _>>()?,
        )
        .ok_or_else(|| Error::msg("Failed to convert image data to GrayImage"))?;

        let rotated_image = rotate(
            &image,
            (
                f32::from_usize(IMAGE_WIDTH)
                    .ok_or_else(|| Error::msg(format!("Failed to convert {IMAGE_WIDTH} to f32")))?
                    / 2.0,
                f32::from_usize(IMAGE_HEIGHT).ok_or_else(|| {
                    Error::msg(format!("Failed to convert {IMAGE_HEIGHT} to f32"))
                })? / 2.0,
            ),
            theta,
            Interpolation::Bilinear,
            Luma::from([0]),
        );

        Ok(Self {
            numeric_value: self.numeric_value,
            raw_data: rotated_image
                .to_vec()
                .iter()
                .map(|i| {
                    f64::from_u8(*i)
                        .ok_or_else(|| Error::msg(format!("Failed to convert {} to f64", *i)))
                })
                .collect::<Result<_, _>>()?,
        })
    }

    /// Displays the image using a wrapper around Python matplotlib.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if the `raw_data` can not be converted to a `IMAGE_HEIGHT` x `IMAGE_WIDTH`
    /// `Array`.
    pub fn display(&self) -> Result<()> {
        let image_array = Array::from_vec(self.raw_data.clone())
            .into_shape_with_order((IMAGE_HEIGHT, IMAGE_WIDTH))?;

        let image_vec: Vec<Vec<f64>> = image_array
            .rows()
            .into_iter()
            .map(|a| a.iter().copied().collect())
            .collect();

        Mpl::new()
            & c::imshow(image_vec)
                .o("cmap", "Greys")
                .o("interpolation", "none")
            | Run::Show;

        Ok(())
    }
}
