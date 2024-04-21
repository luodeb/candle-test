最近看到一个非常有意思的项目Candle，使用Rust做机器学习开发。

# 前言
AI的成本来自哪里？数据、算法还是资源？现如今，AI的开发还处于上升期，这个时候，最需要的是数据的快速获取，以及算法快速落地，所以Python成为了机器学习的首选语言。但是随着AI应用的不断完善，占成本最高的是算力资源和电力资源，这个时候Python的劣势就暴露出来了，Python的解释性语言导致了性能的不足，以及资源的浪费。所以，一门性能更好，更加适合模型推理服务，更“省电”的语言——Rust将会是后AI时代的首选。
“下一个短缺的将是电力。” —— Elon Musk

# Candle简介
Candle is a minimalist ML framework for Rust with a focus on performance (including GPU support) and ease of use.
Candle是一个专注于性能（包括GPU支持）和易用性的Rust的最小化ML框架，由Hugging Face公司开发。Candle 使我们能够使用一个类似 torch 的 API 在 Rust 中构建健壮且轻量级的模型推理服务。基于 Candle 的推理服务将容易扩展，快速引导，并且以极快的速度处理请求，这使它更适合应对规模和韧性挑战的云原生无服务器环境。

# Candle体验
使用Candle来做一个经典的mnist手写数字识别。

## Cargo.toml
``` toml
[package]
name = "candle-test"
version = "0.1.0"
edition = "2021"

# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]

candle-nn = "0.4.1"
candle-core = "0.4.1"
candle-datasets = "0.4.1"
anyhow = "1.0.75"
flate2 = "1.0.28"
rand = "0.8.5"

[features]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
```

## mnist 手写数字数据集
这个数据集是一个非常经典的数据集，用于训练模型，这里我们使用Candle提供的数据集。
训练集图片：60000\*28\*28
训练集标签：60000\*1
测试集图片：10000\*28\*28
测试集标签：10000\*1

加载数据集：
``` rust
let dataset_dir = "datasets/mnist";
decompress_dataset(dataset_dir);

let dataset = candle_datasets::vision::mnist::load_dir(dataset_dir)?;

println!("train-images: {:?}", dataset.train_images.shape());
println!("train-labels: {:?}", dataset.train_labels.shape());
println!("test-images: {:?}", dataset.test_images.shape());
println!("test-labels: {:?}", dataset.test_labels.shape());
/* Output:
train-images: [60000, 784]
train-labels: [60000]
test-images: [10000, 784]
test-labels: [10000]
*/
```

## CNN模型
定义一个简单的CNN模型，由两个卷积层和；两个全连接层，一个Dropout层组成。
``` rust
pub trait Model : Sized {
    fn forward(&self, input: &Tensor, config: &Config) 
        -> Result<Tensor, Error>;
    // TODO: Make this accept a config
    fn new(vars: VarBuilder, labels: usize) -> Result<Self, Error>;
}

pub struct CNN {
    conv1: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    dropout: candle_nn::Dropout,
}

impl Model for CNN {
    fn new(vs: VarBuilder, labels: usize) -> Result<Self, Error> {
        let conv1 = candle_nn::conv2d(1, 4, 5, Default::default(), vs.pp("c1"))?;
        let conv2 = candle_nn::conv2d(4, 8, 5, Default::default(), vs.pp("c2"))?;
        let fc1 = candle_nn::linear(128, 64, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(64, labels, vs.pp("fc2"))?;
        let dropout = candle_nn::Dropout::new(0.5);
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
            dropout,
        })
    }
    fn forward(&self, xs: &Tensor, config: &Config) 
        -> Result<Tensor, Error> {
        // Get the batch and image dimensions from the tensor
        let (b_sz, _img_dim) = xs.dims2()?;
        let mut varmap = VarMap::new();
        if let Some(load) = &config.load {
            println!("loading weights from {load}");
            varmap.load(load)?
        }
        let xs = xs
            .reshape((b_sz, 1, 28, 28))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?;
        let x = self.dropout.forward_t(&xs, config.train)?.apply(&self.fc2)?;
        if let Some(save) = &config.save {
            println!("saving trained weights in {save}");
            varmap.save(save)?
        }
        Ok(x)
    }
}
```

有一说一，跟PyTorch相比，还挺像那么回事的。

## 训练参数
``` rust
pub struct Config {
    lr: f64,    // 学习率
    load: Option<String>, // 加载模型
    save: Option<String>, // 保存模型
    epochs: usize, // epoch
    train: bool, // 是否训练
    batch_size: usize, // batch_size
}

impl Config {
    pub fn new(lr: f64, load: Option<String>, save: Option<String>,
        epochs: usize, train: bool, batch_size: usize) -> Self {
        Config {
            lr: lr, 
            load: load,
            save: save,
            epochs: epochs,
            train: train,
            batch_size: batch_size,
        }
    }
    pub fn get_lr(&self) -> f64 {
        self.lr
    }

    pub fn get_num_epochs(&self) -> usize {
        self.epochs
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }
}

let config = Config::new(
    0.05, // 学习率
    None, // 加载模型
    None, // 保存模型
    10,    // epoch
    true, // 是否训练
    1024, // batch_size
);
```

这些参数设置的还是比较简单的。没有装CUDA，比较慢。

## 训练模型
``` rust
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

/*
   0 train loss  1.96967 test acc: 47.35%
   1 train loss  1.37902 test acc: 57.75%
   2 train loss  1.19515 test acc: 63.13%
   3 train loss  1.09978 test acc: 64.90%
   4 train loss  1.04852 test acc: 67.16%
   5 train loss  1.02437 test acc: 67.40%
   6 train loss  0.98418 test acc: 70.13%
   7 train loss  0.94210 test acc: 71.01%
   8 train loss  0.88540 test acc: 70.75%
   9 train loss  0.96413 test acc: 74.40%
*/
```

老样子，加载训练数据，加载测试数据，加载模型，定义优化器，开始训练，测试。

# 总结
总体使用下来，感觉还行，可以看出Hugging Face公司已经在尽力向PyTorch靠拢了，整体的使用体验还是不错的。
不过怎么感觉同样是CPU训练，好像性能没有提升多少，没有具体测试，感兴趣的同学可以自己测试一下。

Github地址：
[Candle](https://github.com/huggingface/candle)
[代码](https://github.com/luodeb/candle-test)
