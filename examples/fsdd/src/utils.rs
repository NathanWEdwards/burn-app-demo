use std::env;
use tracing_subscriber::prelude::*;
use tracing_appender::rolling::{
    RollingFileAppender,
    Rotation
};



pub fn rolling_appender() -> tracing_appender::non_blocking::WorkerGuard {
    let log_location = match env::var("LOG_LOCATION") {
        Ok(folder) => folder,
        Err(_) => String::from("./log")
    };
    let file_appender = RollingFileAppender::new(
        Rotation::DAILY,
        log_location,
        "fsdd.log"
    );
    let (non_blocking_appender, _guard) = tracing_appender::non_blocking(file_appender);
    let file_layer = tracing_subscriber::fmt::Layer::new()
        .with_writer(non_blocking_appender)
        .with_filter(tracing_subscriber::filter::LevelFilter::DEBUG);
    tracing_subscriber::registry()
        .with(file_layer)
        .init();
    return _guard;
}