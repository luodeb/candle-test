use std::{
    fs::File,
    io::{copy, BufReader},
    path::PathBuf,
    str::FromStr,
};
use flate2::bufread;

pub fn decompress_dataset(path: &str) {
    let dir = PathBuf::from_str(path).unwrap();
    if !dir.exists() {
        panic!("dataset path not found!");
    }

    let file_list = vec![
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
    ];
    for file in file_list {
        let mut temp = dir.clone();
        temp.push(file);
        if !temp.exists() {
            println!("decompress gz file:{}.gz", file);
            let output = temp.clone();
            let mut output = File::create(output.as_path()).unwrap();
            let mut input = dir.clone();
            input.push(format!("{file}.gz"));
            let input = BufReader::new(File::open(input.as_path()).unwrap());
            let mut decoder = bufread::GzDecoder::new(input);
            copy(&mut decoder, &mut output).unwrap();
        } 
    }
}
