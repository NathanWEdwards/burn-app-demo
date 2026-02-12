use burn::{
    backend::{Autodiff, Wgpu},
    config::Config,
    data::dataloader::DataLoaderBuilder,
    nn::{
        RotaryEncodingConfig,
        // loss::{MseLoss, Reduction::Mean},
        loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Float, Int, Tensor, backend::AutodiffBackend, s},
};
use clap::Parser;
use std::path::PathBuf;
use tracing_subscriber::filter::targets;

use app::model::{KeyValueCache, TransformerConfig};
use fsdd::{
    dataset::{FsddBatcher, FsddDataset},
    logging::{log_tensor_state, log_wav, rolling_appender},
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    dir_channel_capacity: usize,
    #[arg(long)]
    file_channel_capacity: usize,
    #[arg(long)]
    num_threads: usize,
    #[arg(long)]
    path: String,
    // Training configurations
    #[arg(long)]
    pub num_epochs: usize,
    #[arg(long)]
    pub batch_size: usize,
    #[arg(long)]
    pub num_workers: usize,
    #[arg(long)]
    pub seed: u64,
    // Model configurations
    #[arg(long)]
    pub embedding_size: usize,
    #[arg(long)]
    pub model_dim: usize,
    #[arg(long)]
    pub num_layers: usize,
    #[arg(long)]
    pub intermediate_size: usize,
    #[arg(long)]
    pub num_heads: usize,
    #[arg(long)]
    pub num_kv_heads: usize,
    #[arg(long)]
    pub beta_1: f32,
    #[arg(long)]
    pub beta_2: f32,
    #[arg(long)]
    pub epsilon: f32,
    #[arg(long)]
    pub learning_rate: f64,
    #[arg(long)]
    pub tolerance: f64,
}

fn main() {
    let _guard = rolling_appender();
    type Backend = Wgpu<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;
    let args = Args::parse();
    let device = burn::backend::wgpu::WgpuDevice::default();
    let model_config = TransformerConfig::new(
        args.embedding_size,
        args.model_dim,
        args.num_layers,
        args.intermediate_size,
        args.num_heads,
        args.num_kv_heads,
    )
    .with_epsilon(args.epsilon);
    let optimizer_config = AdamConfig::new()
        .with_beta_1(args.beta_1)
        .with_beta_2(args.beta_2)
        .with_epsilon(args.epsilon.clone());
    let fsdd_config = FsddConfig::new(
        args.num_threads,
        PathBuf::from(&args.path),
        args.num_epochs,
        args.batch_size,
        args.num_workers,
        args.seed,
        model_config,
        optimizer_config,
    )
    .with_dir_channel_capacity(args.dir_channel_capacity)
    .with_file_channel_capacity(args.file_channel_capacity)
    .with_learning_rate(args.learning_rate)
    .with_tolerance(args.tolerance);
    run::<AutodiffBackend>(device, fsdd_config);
}

#[derive(Config, Debug)]
struct FsddConfig {
    // Data loading configurations
    #[config(default = "10")]
    dir_channel_capacity: usize,
    #[config(default = "10")]
    file_channel_capacity: usize,
    num_threads: usize,
    path: PathBuf,
    // Training configurations
    pub num_epochs: usize,
    pub batch_size: usize,
    pub num_workers: usize,
    pub seed: u64,
    // Optimizer configurations
    #[config(default = "1e-5")]
    pub epsilon: f64,
    #[config(default = "1e-3")]
    pub learning_rate: f64,
    #[config(default = "1e-1")]
    pub tolerance: f64,
    // Additional configurations
    pub model: TransformerConfig,
    pub optimizer: AdamConfig,
}

fn run<B: AutodiffBackend>(device: B::Device, fsdd_config: FsddConfig) {
    B::seed(&device, fsdd_config.seed);

    let dataset = FsddDataset::new(
        fsdd_config.path,
        fsdd_config.dir_channel_capacity,
        fsdd_config.file_channel_capacity,
        fsdd_config.num_threads,
    );

    let mut cache = (0..fsdd_config.model.num_layers)
        .map(|_| {
            KeyValueCache::new(
                fsdd_config.batch_size,
                fsdd_config.model.num_kv_heads,
                fsdd_config.model.embedding_size,
                fsdd_config.model.model_dim,
                &device,
            )
        })
        .collect::<Vec<_>>();

    let rope = RotaryEncodingConfig::new(
        fsdd_config.model.embedding_size * 2,
        fsdd_config.model.model_dim / fsdd_config.model.num_heads,
    );
    let rope = rope.init(&device);

    let batcher = FsddBatcher::new(device.clone(), fsdd_config.model.embedding_size);

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(fsdd_config.batch_size)
        .shuffle(fsdd_config.seed)
        .num_workers(fsdd_config.num_workers)
        .build(dataset);

    let model = fsdd_config.model.init::<B>(&device);
    let mut optimizer = fsdd_config.optimizer.init();

    for epoch in 1..=fsdd_config.num_epochs {
        for (_iteration, batch) in dataloader_train.iter().enumerate() {
            let inputs: Tensor<B, 2, Float> = batch.inputs.to_device(&device);
            let targets: Tensor<B, 2, Int> = batch.targets.to_device(&device).int();
            let targets: Tensor<B, 1, Int> = targets.slice(s![.., -1]).squeeze_dims(&[1]);
            let output = model.forward(inputs.clone(), &mut cache, &rope);
            if epoch % 2 == 1 {
                log_wav(
                    output.clone().mean_dim(1).squeeze(),
                    &format!("output_{}.wav", _iteration),
                );
            }
            let loss = CrossEntropyLossConfig::new()
                .init(&device)
                .forward(output, targets);
            let gradients = loss.backward();
            let gradients = GradientsParams::from_grads(gradients, &model);
            let _model = optimizer.step(fsdd_config.learning_rate, model.clone(), gradients);
        }
    }
}
