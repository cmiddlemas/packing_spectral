use packing_spectral::*;
use chrono::prelude::*;

fn main() {
    let opt = Opt::from_args();
    println!("Beginning calculation at: {}", Utc::now());
    spectral_density(&opt);
    println!("Exiting successfully at time: {}", Utc::now());
}
