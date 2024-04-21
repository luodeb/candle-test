use anyhow::Ok;
use candle_core::{DType, Device, D};
use candle_datasets::vision::Dataset;
use candle_nn::{loss, ops, Optimizer, VarBuilder, VarMap};
use candle_test::models::{Config, Model, CNN};
use candle_test::utils::decompress_dataset;
use rand::prelude::{thread_rng, SliceRandom};

fn train(dataset: Dataset, config: Config, device: Device) -> anyhow::Result<()> {
    let bsize: usize = config.get_batch_size();

    let train_images = dataset.train_images.to_device(&device)?;
    let train_labels = dataset
        .train_labels
        .to_dtype(DType::U32)?
        .to_device(&device)?;

    let test_images = dataset.test_images.to_device(&device)?;
    let test_labels = dataset
        .test_labels
        .to_dtype(DType::U32)?
        .to_device(&device)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = CNN::new(vs.clone(), 10)?;

    let mut sgd = candle_nn::optim::SGD::new(varmap.all_vars(), config.get_lr())?;
    let n_batches = train_images.dim(0)? / bsize;
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();

    for epoch in 0..config.get_num_epochs() {
        batch_idxs.shuffle(&mut thread_rng()); // 打乱顺序
        let mut sum_loss = 0f32;

        for batch_idx in batch_idxs.iter() {
            let train_images = train_images.narrow(0, batch_idx * bsize, bsize)?;
            let train_labels = train_labels.narrow(0, batch_idx * bsize, bsize)?;
            let logits = model.forward(&train_images, &config)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &train_labels)?;
            sgd.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;
        }

        let avg_loss = sum_loss / n_batches as f32;

        let test_logits = model.forward(&test_images, &config)?;

        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;

        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
            avg_loss,
            100. * test_accuracy
        );
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;

    let dataset_dir = "datasets/mnist";
    decompress_dataset(dataset_dir);

    let dataset = candle_datasets::vision::mnist::load_dir(dataset_dir)?;

    println!("train-images: {:?}", dataset.train_images.shape());
    println!("train-labels: {:?}", dataset.train_labels.shape());
    println!("test-images: {:?}", dataset.test_images.shape());
    println!("test-labels: {:?}", dataset.test_labels.shape());

    let config = Config::new(
        0.05, // 学习率
        None, // 加载模型
        None, // 保存模型
        1,    // epoch
        true, // 是否训练
        1024, // batch_size
    );

    train(dataset, config, device)?;

    Ok(())
}
