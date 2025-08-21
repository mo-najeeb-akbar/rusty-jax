use burn::tensor::Tensor;
use burn::backend::ndarray::{NdArray, NdArrayDevice};

mod model;
use model::vae::Model;

type Backend = NdArray<f32>;

fn main() {
    println!("Initializing CPU (NdArray) backend...");
    let device = NdArrayDevice::Cpu;
    println!("Using CPU device");

    println!("Loading VAE model...");
    let model: Model<Backend> = Model::default();
    
    // let model = model.load_file("src/model/vae.mpk", &recorder, &device)
    //     .expect("Failed to load model weights");

    let input = Tensor::<Backend, 4>::random([1, 1, 256, 256], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    
    println!("Input tensor shape: {:?}", input.dims());
    println!("Running inference on CPU...");
    
    let start = std::time::Instant::now();
    let output = model.forward(input);
    let duration = start.elapsed();
    
    println!("Inference completed in {:?}", duration);
}

