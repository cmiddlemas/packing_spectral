use packing_spectral::*;
use structopt::StructOpt;
use chrono::prelude::*;

// From 
// https://stackoverflow.com/questions/39204908/how-to-check-release-debug-builds-using-cfg-in-rust
// and
// https://vallentin.io/2019/06/06/versioning
#[cfg(debug_assertions)]
const BUILD_MODE: &str = "debug";
#[cfg(not(debug_assertions))]
const BUILD_MODE: &str = "release";

#[cfg(feature = "using_make")]
const USING_MAKE: &str = "true";
#[cfg(not(feature = "using_make"))]
const USING_MAKE: &str = "false";

fn main() {
// Print out build time information
    println!("Build info for packing_spectral:");
    println!("Using make: {}", USING_MAKE);
    println!("Cargo version: {}", env!("C_VER"));
    println!("Commit SHA: {}", env!("VERGEN_SHA"));
    println!("Commit date: {}", env!("VERGEN_COMMIT_DATE"));
    println!("Version: {}", env!("VERGEN_SEMVER"));
    println!("Build: {}", BUILD_MODE);
    println!("Build time: {}", env!("VERGEN_BUILD_TIMESTAMP"));
    println!("Compiler version: {}", env!("V_RUSTC"));
    println!("RUSTFLAGS: {}", env!("S_RUSTFLAGS"));
    println!("Target: {}", env!("VERGEN_TARGET_TRIPLE"));
    println!("Clean working directory for build: {}", env!("WD_IS_CLEAN"));
    println!("Start program:\n");

    let opt = Opt::from_args();
    println!("Beginning calculation at: {}", Utc::now());
    spectral_density_cmdline(&opt);
    println!("Exiting successfully at time: {}", Utc::now());
}
