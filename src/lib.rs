pub use structopt::StructOpt;
use rayon::prelude::*;
use std::path::PathBuf;
use std::fs::File;
use std::io::{Read, BufRead, BufReader, Write, BufWriter};
use nalgebra::{Matrix3, LU};

//Value from Mathematica 11.2.0.0
const PI: f64 = 3.141592653589793;

#[derive(StructOpt, Debug)]
#[structopt(name = "packing_spectral", about = "Computes spectral density for polydisperse sphere packings.\nOutputs data on stdout if -o is not specified.")]
pub struct Opt {
    /// Max number of reciprocal lattice vectors to go out to
    #[structopt(short = "m",
                long = "maxvec",
                default_value = "10")]
    pub max_vec: usize,
    
    /// Output filename, file will be given as{n}
    /// max_vec{n}
    /// reciprocal_vector1{n}
    /// reciprocal_vector2{n}
    /// ...{n}
    /// kx ky ... S(k){n}
    /// for every k vector on its own line
    #[structopt(short = "o", long = "outfile",
                parse(from_os_str))]
    pub outfile: Option<PathBuf>,

    /// Compute structure factor instead
    /// of spectral density
    #[structopt(short = "s", long = "structure")]
    pub compute_structure_factor: bool,

    /// Input filenames, files must follow Donev convention
    /// and be mutually compatible (i.e. same dimension,
    /// unit cells, and number of spheres)
    #[structopt(name = "infile", parse(from_os_str))]
    pub infiles: Vec<PathBuf>,
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
    reciprocal_cell: Vec<f64>,
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

        println!("Read {} as dim", dim);

        // Get number of points
        bufr.read_line(&mut bufs).expect("io line error");
        bufs.clear();
        bufr.read_line(&mut bufs).expect("io line error");
        let n_points: usize = bufs.trim().parse()
            .expect("parse n_points failed");

        println!("Read {} as number of points", n_points);

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

        // Make reciprocal cell
        let reciprocal_cell = reciprocal_lattice3(
                &unit_cell
        );
        
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

        Config {dim, n_points, config, unit_cell, reciprocal_cell} 
    }
}

// Reciprocal lattice is computed as
// q = 2Pi(u^(-1))^T
fn reciprocal_lattice3(unit_cell: &Vec<f64>)
    -> Vec<f64> 
{
    let u = Matrix3::from_row_slice(unit_cell);
    let q = 2.0*PI
        *u.lu()
        .try_inverse()
        .expect("unit cell matrix must be invertible")
        .transpose();
    Vec::from(q.as_slice())
}

fn make_support3(q_lat: &Vec<f64>, max_idx: usize)
    -> Vec<(f64, f64, f64)> 
{
    let mut support = Vec::new();
    let s_idx = max_idx as isize;
    for i in -s_idx..(s_idx+1) {
        for j in -s_idx..(s_idx+1) {
            for k in -s_idx..(s_idx+1) {
                let m = i as f64;
                let n = j as f64;
                let o = k as f64;
                support.push(
                    (m*q_lat[0] + n*q_lat[3] + o*q_lat[6],
                     m*q_lat[1] + n*q_lat[4] + o*q_lat[7],
                     m*q_lat[2] + n*q_lat[5] + o*q_lat[8])
                );
            }
        }
    }
    return support;
}

// Gives the form factor for a sphere of radius
// r evaluated at radial wavenumber q
// Reference:
// Zachary et. al. PRE 83, 051308 (2011)
// and Mathematica 11.2.0.0
fn form_factor3(q: f64, r: f64) -> f64 {
    0.0
}

fn one_wavevector3(q: (f64, f64, f64),
                   val: &mut f64,
                   config: &Config)
{
}

fn sf_one_wavevector3(q: (f64, f64, f64),
                      val: &mut f64,
                   config: &Config)
{
    //println!("Working on wavevector: {:?}", q);
    let mut real: f64 = 0.0;
    let mut imag: f64 = 0.0;
    for p in config.config.chunks(4) {
        let dot = -(q.0*p[0] + q.1*p[1] + q.2*p[2]);
        real += dot.cos();
        imag += dot.sin();
    }
    *val += (real*real + imag*imag)/(config.n_points as f64);
}

fn one_configuration3(config: &Config,
                     spectral_density: &mut [f64],
                     support: &[(f64, f64, f64)],
                     opt: &Opt)
{
    if opt.compute_structure_factor {
        support.par_iter().zip(spectral_density.par_iter_mut())
            .for_each(|(q, val)| sf_one_wavevector3(*q, val, &config));
    } else {
        support.par_iter().zip(spectral_density.par_iter_mut())
            .for_each(|(q, val)| one_wavevector3(*q, val, &config));
    }
}

fn write_results3(config: &Config,
                      spectral_density: &[f64],
                      support: &[(f64,f64,f64)],
                      opt: &Opt
)
{
    let q_lat = &config.reciprocal_cell;
    // See 
    // https://stackoverflow.com/questions/25278248/rust-structs-with-nullable-option-fields
    // for borrow checker discussion
    match &opt.outfile {
        Some(p) => {
            let mut outfile = BufWriter::new(
                File::create(p)
                .expect("Couldn't make outfile")
            );
            writeln!(&mut outfile, "{}", opt.max_vec)
                .expect("Couldn't write to outfile");
            for i in 0..3 {
                writeln!(&mut outfile, "{}\t{}\t{}",
                    q_lat[3*i], q_lat[3*i+1], q_lat[3*i+2])
                    .expect("Couldn't write to outfile");
            }
            for (q, chi) in support.iter()
                .zip(spectral_density.iter()) 
            {
                writeln!(&mut outfile,
                         "{}\t{}\t{}\t{}",
                         q.0, q.1, q.2, chi)
                .expect("Couldn't write to outfile");
            }
        }
        None => { // Default to stdout
            println!("{}", opt.max_vec);
            for i in 0..3 {
                println!("{}\t{}\t{}",
                    q_lat[3*i], q_lat[3*i+1], q_lat[3*i+2]);
            }
            for (q, chi) in support.iter()
                .zip(spectral_density.iter()) 
            {
            println!("{}\t{}\t{}\t{}", 
                 q.0, q.1, q.2, chi);
            }
        }
    }
}

pub fn spectral_density(opt: &Opt) {
    let first_config = Config::from_file(&opt.infiles[0]);
    match first_config.dim {
        3 => {
            // Initialize data structures, assume
            // first file defines the correct
            // reciprocal lattice and number
            // of spheres
            let support = make_support3(
                &first_config.reciprocal_cell,
                opt.max_vec
            );
            let mut spectral_density = vec![0.0; support.len()];
            
            // Ensemble sum
            for (i, path) in opt.infiles.iter().enumerate() {
                println!("Processing file: {}", i);
                let config = Config::from_file(path);
                one_configuration3(&config,
                                   &mut spectral_density,
                                   &support,
                                   opt
                );
            }
            
            // Normalize by number of realizations
            let n_ens = opt.infiles.len() as f64;
            for val in &mut spectral_density {
                *val /= n_ens;
            }

            write_results3(&first_config,
                           &spectral_density,
                           &support,
                           opt
            );

        }
        _ => {
            panic!("That dimension is not implemented")
        }
    }
}
