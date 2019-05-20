use packing_spectral::*;
use structopt::StructOpt;
use chrono::prelude::*;

fn main() {
    let opt = Opt::from_args();
    println!("Beginning calculation at: {}", Utc::now());
    spectral_density_cmdline(&opt);
    println!("Exiting successfully at time: {}", Utc::now());
}
