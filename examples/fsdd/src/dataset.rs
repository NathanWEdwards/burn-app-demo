use core::task::Poll;
use std::path::{
    Path,
    PathBuf
};
use async_channel::{
    Receiver,
    Sender,
    TryRecvError,
    bounded
};
use burn_dataset::{
    Dataset,
    InMemDataset
};
use tokio::{
        fs::read_dir,
        runtime::Runtime,
        task::JoinSet,
        task::spawn
};
use crate::convert::to_samples;

struct FsddItem {
    pub samples: Vec<f64>
}

impl FsddItem {
    pub fn new(samples: Vec<f64>) -> Self {
        FsddItem {
            samples
        }
    }
}

struct Loader {
    dir_channel_capacity: usize,
    file_channel_capacity: usize,
    num_threads: usize,
    path: PathBuf
}

impl Loader {
    fn new(
        dir_channel_capacity: usize,
        file_channel_capacity: usize,
        num_threads: usize,
        path: PathBuf
    ) -> Self {
        Loader {
            dir_channel_capacity,
            file_channel_capacity,
            num_threads,
            path
        }
    }

    pub fn load(&self) -> Vec<FsddItem> {
        let (fs_channel_send, mut fs_channel_receive) = bounded::<PathBuf>(self.dir_channel_capacity);
        let fs_channel_receive_clone_for_discovery = fs_channel_receive.clone();
        let (wav_file_channel_send, mut wav_file_channel_receive) = bounded::<PathBuf>(self.file_channel_capacity);
        let cloned_path = self.path.clone();
        // Declarations for asynchronous use.
        let mut joinset = JoinSet::new();
        let runtime = Runtime::new();
        if let Ok(rt) = runtime {
            rt.block_on(async {
                let mut data: Vec<FsddItem> = Vec::new();
                // Add the initial path to start discovery.
                fs_channel_send.send(cloned_path).await;
                // Spawn tasks to handle discovered .wav files.
                for _ in 0..self.num_threads {
                    let wav_file_channel_clone = wav_file_channel_receive.clone();
                    joinset.spawn(async move {
                            let mut thread_items: Vec<FsddItem> = Vec::new();
                            while let Ok(wav_file) = wav_file_channel_clone.recv().await {
                                if let Some(file) = wav_file.to_str() {
                                    let (samples, _) = to_samples(file);
                                    let mut item: FsddItem = FsddItem::new(samples); 
                                    thread_items.push(item);
                                }
                            }
                            thread_items
                    });
                }
                // Drop original sender to close channels when done.
                drop(fs_channel_send);
                // Spawn a task to discover .wav files.
                discover_wav_files(
                    fs_channel_receive_clone_for_discovery,
                    wav_file_channel_send
                ).await;
                while let Some(result) = joinset.join_next().await {
                    if let Ok(mut items) = result {
                        data.append(&mut items);
                    }
                }
                data
            })
        } else {
            return Vec::new();
        }
    }
}

async fn discover_wav_files(
    fs_receiver: Receiver<PathBuf>,
    wav_file_sender: Sender<PathBuf>,
) {
    // Start processing directories.
    if let Ok(directory) = fs_receiver.recv().await {
        if let Ok(mut read_directory) = read_dir(&directory).await {
            // Read contents of a directory from the channel.
            while let Ok(Some(entry)) = read_directory.next_entry().await {
                // Send .wav files over the wav_file_channel
                if let Some(ext) = entry.path().extension() {
                    if ext == "wav" {
                        wav_file_sender.send(entry.path()).await;
                    }
                }
            }
        }
    }
}

pub struct FsddDataset {
    dataset: InMemDataset<FsddItem>
}

impl FsddDataset {
    pub fn new(
        path: PathBuf,
        dir_channel_capacity: usize,
        file_channel_capacity: usize,
        num_threads: usize
    ) -> Self {
        let loader = Loader::new(
            dir_channel_capacity,
            file_channel_capacity,
            num_threads,
            path
        );
        let items = loader.load();
        let dataset = InMemDataset::new(items);
        FsddDataset { dataset }
    }
}