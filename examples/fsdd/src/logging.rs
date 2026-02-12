use burn::tensor::{self, Tensor};
use chrono::{DateTime, Utc};
use std::{env, path::PathBuf};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::prelude::*;

use crate::convert::to_wav_file;

pub fn rolling_appender() -> tracing_appender::non_blocking::WorkerGuard {
    let log_location = match env::var("LOG_LOCATION") {
        Ok(folder) => folder,
        Err(_) => String::from("./log"),
    };
    let file_appender = RollingFileAppender::new(Rotation::DAILY, log_location, "fsdd.log");
    let (non_blocking_appender, _guard) = tracing_appender::non_blocking(file_appender);
    let filter = tracing_subscriber::filter::Targets::new()
        .with_target("fsdd::logging", tracing::Level::INFO);
    let file_layer = tracing_subscriber::fmt::Layer::new()
        .with_writer(non_blocking_appender)
        .with_filter(filter);
    tracing_subscriber::registry().with(file_layer).init();
    return _guard;
}

pub fn log_tensor_state<B: burn::tensor::backend::Backend, const D: usize>(
    message: &str,
    tensor: Tensor<B, D>,
) {
    tracing::info!("{message}\nTensor state: {:?}", tensor.to_data());
}

pub fn log_wav<B: burn::tensor::backend::Backend>(tensor: Tensor<B, 1>, message: &str) {
    tracing::info!("{message}\nLogging tensor as wav file.");
    let current_time: DateTime<Utc> = Utc::now();
    let mut log_to_location = match env::var("LOG_LOCATION") {
        Ok(folder) => PathBuf::from(folder),
        Err(_) => PathBuf::from("./log"),
    };
    log_to_location.push(format!("{}.wav", current_time.timestamp()));
    let log_to_location = match log_to_location.to_str() {
        Some(path) => path,
        None => {
            tracing::error!("Failed to convert log location to string. Logging failed.");
            return;
        }
    };
    let num_channels = 1;
    let sample_rate = 1024;
    to_wav_file(&log_to_location, tensor, num_channels, sample_rate);
}
