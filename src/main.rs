use packing_spectral::*;

fn main() {
    println!("Hello, world!");
    let opt = Opt::from_args();
    spectral_density(&opt);    
}
