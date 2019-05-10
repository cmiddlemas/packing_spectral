pub use structopt::StructOpt;
use rayon::prelude::*;
use std::path::PathBuf;
use std::fs::File;
use std::io;
use std::io::{Read, BufRead, BufReader};
use nalgebra::Matrix3;

#[derive(StructOpt, Debug)]
#[structopt(name = "Spectral density for packings")]
pub struct Opt {
    /// Max number of lattice vectors to go out to
    #[structopt(short = "m",
                long = "maxvec",
                default_value = "10")]
    pub max_vec: usize,

    /// Input filename, file must follow Donev convention
    #[structopt(name = "infile", parse(from_os_str))]
    pub infile: PathBuf,

    /// Output filename, file will be given as
    /// idx1 idx2 ... kx ky ... S(k)
    /// for every k vector on its own line
    #[structopt(name = "outfile", parse(from_os_str))]
    pub outfile: PathBuf,
}

#[derive(Debug)]
struct Config {
    dim: usize,
    n_points: usize,
    // Coordinates + radius, flat storage:
    // x1 y1 z1 R1 x2 y2 z2 R2 ...
    config: Vec<f64>,
    // Unit cell, flat storage:
    // u11 u12 u13 u21 u22 u23 ...
    unit_cell: Vec<f64>,
}

impl Config {
    // Thanks to 
    // https://users.rust-lang.org/t/read-a-file-line-by-line/1585
    fn from_file(path: &PathBuf) -> Config {
        let mut bufr = BufReader::new(File::open(path)
                                  .expect("io error")
        );
        let mut bufs = String::new();

        // Get dimension
        bufr.read_line(&mut bufs).expect("io line error");
        let dim: usize = bufs.split_whitespace()
            .collect::<Vec<&str>>()[0]
            .parse()
            .expect("parse dim failed");

        // Get number of points
        bufr.read_line(&mut bufs).expect("io line error");
        bufs.clear();
        bufr.read_line(&mut bufs).expect("io line error");
        let n_points: usize = bufs.trim().parse()
            .expect("parse n_points failed");

        // Get unit cell
        bufs.clear();
        for _ in 0..dim {
            bufr.read_line(&mut bufs).expect("io line error");
        }
        let unit_cell: Vec<f64> = bufs.split_whitespace()
                            .map(|x| x.parse()
                                .expect("parse unit_cell failed")
                                )
                            .collect();
        
        // Get config
        bufr.read_line(&mut bufs).expect("io line error");
        bufs.clear();
        bufr.read_to_string(&mut bufs)
            .expect("io read to string error");
        let config: Vec<f64> = bufs.split_whitespace()
            .map(|x| x.parse()
                 .expect("parse config failed")
                 )
            .collect();

        // Consistency check
        // Config is coordinates + radius
        assert!(config.len() == (dim+1)*n_points);
        assert!(unit_cell.len() == dim*dim);

        Config {dim, n_points, config, unit_cell} 
    }
}

fn reciprocal_lattice(unit_cell: &Vec<f64>, dim: usize) 
    -> Vec<f64> 
{
    match dim {
        3 => {
            let u = Matrix3::from_row_slice(unit_cell);
            let k = u.lu()
        }
        _ => {
            panic!("That dimension is not implemented for
                reciprocal_lattice");
        }
    }
}

pub fn spectral_density(opt: &Opt) {
    let config = Config::from_file(&opt.infile);
    println!("config:\n {:?}", config);
}
