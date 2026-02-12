use burn::tensor::Tensor;
use tracing::error;

pub(crate) fn to_samples(file: &str) -> (Vec<f64>, f64) {
    let (samples, sample_rate): (Vec<f64>, f64) = match hound::WavReader::open(file) {
        Ok(mut reader) => {
            let samples: Vec<f64> = reader
                .samples()
                .filter_map(|s| s.ok())
                .map(|s: f32| s as f64)
                .collect::<Vec<f64>>();
            let sample_rate: f64 = reader.spec().sample_rate as f64;
            (samples.to_vec(), sample_rate.into())
        }
        Err(_) => (Vec::new(), 0.0),
    };
    (samples, sample_rate)
}

pub(crate) fn to_wav_file<B: burn::tensor::backend::Backend, const D: usize>(
    file_path: &str,
    tensor: Tensor<B, D>,
    sample_rate: i32,
    n_channels: u16,
) {
    let samples: Vec<f64> = match tensor.to_data().to_vec() {
        Ok(data) => data,
        Err(_) => Vec::new(),
    };

    let wav_spec = hound::WavSpec {
        channels: 1,
        sample_rate: 1024,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = match hound::WavWriter::create(file_path, wav_spec) {
        Ok(writer) => writer,
        Err(e) => {
            error!("Failed to create wav file: {}", e);
            return;
        }
    };
    for sample in samples {
        if let Err(e) = writer.write_sample(sample as i16) {
            error!("Failed to write sample to wav file: {}", e);
            return;
        }
    }
    if let Err(e) = writer.finalize() {
        error!("Failed to finalize wav file: {}", e);
    }
}
