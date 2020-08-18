use structopt::StructOpt;
use std::path::PathBuf;
use nalgebra::{Matrix2, Vector2, Matrix3, Vector3};
use std::fs::File;
use std::io::{BufRead, BufReader, Write, BufWriter};
use rayon::prelude::*;
use nalgebra::{U2, U3};
use itertools::izip;

mod sphere;
mod ellipsoid;

use sphere::SphereConfig;
use ellipsoid::EllipsoidConfig;

//Value from Mathematica 11.2.0.0
const PI: f64 = 3.141592653589793;

#[derive(StructOpt, Debug)]
#[structopt(name = "packing_spectral",
            about = "Computes spectral density for 
            polydisperse sphere packings.\n
            Outputs data on stdout if -o is not specified."
            )
]
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

    /// Turns off parallelism and runs timing output
    #[structopt(short = "t", long = "timing")]
    pub run_timing: bool,

    /// Input filenames, files must follow Donev convention
    /// and be mutually compatible (i.e. same dimension,
    /// unit cells, and number of spheres)
    #[structopt(name = "infile", parse(from_os_str))]
    pub infiles: Vec<PathBuf>,

    /// If given, will assume input file is of the asc_monte_carlo type.
    /// If not given, will assume input file is of the Donev type
    #[structopt(long)]
    pub asc: bool,
}

// Traits --------------------------------------------------------
// https://github.com/rust-lang/rfcs/issues/2035
// Common methods to all configurations
trait Config {
    fn get_reciprocal_cell(&self) -> &[f64];
    fn one_wavevector(&self, q: &[f64]) -> f64;
    fn sf_one_wavevector(&self, q: &[f64]) -> f64;
    fn get_dimension(&self) -> usize;
    fn get_n_points(&self) -> usize;
    fn get_vol(&self) -> f64;
}

// dimension specific functions ------------------------------------------

// Triple product algorithm, see Wikipedia
fn volume3(uc: &[f64]) -> f64
{
    let v1 = Vector3::new(uc[0], uc[1], uc[2]);
    let v2 = Vector3::new(uc[3], uc[4], uc[5]);
    let v3 = Vector3::new(uc[6], uc[7], uc[8]);
    (v1.dot(&v2.cross(&v3))).abs()
}

fn volume2(uc: &[f64]) -> f64 {
    let v1 = Vector2::new(uc[0], uc[1]);
    let v2 = Vector2::new(uc[2], uc[3]);
    v1.perp(&v2).abs()
}

// Reciprocal lattice is computed as
// q = 2Pi(u^(-1))^T
fn reciprocal_lattice3(unit_cell: &[f64])
    -> Vec<f64> 
{
    let u = Matrix3::from_row_slice(unit_cell);
    let q = 2.0*PI
        *u.lu()
        .try_inverse()
        .expect("unit cell matrix must be invertible");
    Vec::from(q.as_slice())
}

fn reciprocal_lattice2(unit_cell: &[f64]) -> Vec<f64> {
    let u = Matrix2::from_row_slice(unit_cell);
    let q = 2.0*PI
        *u.lu()
        .try_inverse()
        .expect("unit cell matrix must be invertible");
    Vec::from(q.as_slice())
}

// Gives the form factor for a sphere of radius
// r evaluated at radial wavenumber q
// Reference:
// Zachary et. al. PRE 83, 051308 (2011)
// and Mathematica 11.2.0.0
// Also notes for Lecture 1 of CHM 510, by S. Torquato
fn form_factor3(q: f64, r: f64) -> f64 {
    4.0*PI*((q*r).sin() - q*r*(q*r).cos())/(q*q*q)
}

// Generic helper functions ---------------------------------------

// After looking at Ge's code, realized that this should exclude
// the S(k) = S(-k) symmetry for correct error bars
fn make_support(dim: usize, q_lat: &[f64], max_idx: usize)
    -> Vec<Vec<f64>> 
{
    let mut support = Vec::new();
    match dim {
        2 => {
            let s_idx = max_idx as isize;
            // Handle vertical half line segment out of origin
            for j in 0..(s_idx+1) {
                let n = j as f64;
                support.push(
                    vec![n*q_lat[2], n*q_lat[3]]
                );
            }
            // Handle verticle full line segments offset from origin
            for i in 1..(s_idx+1) {
                for j in -s_idx..(s_idx+1) {
                    let m = i as f64;
                    let n = j as f64;
                    support.push(
                        vec![m*q_lat[0] + n*q_lat[2],
                         m*q_lat[1] + n*q_lat[3]]
                    );
                }
            }
        }
        3 => {
            let s_idx = max_idx as isize;
            // Handle boundary
            // i = j = 0, half line
            for k in 0..(s_idx+1) {
                let o = k as f64;
                support.push(
                    vec![o*q_lat[6], o*q_lat[7], o*q_lat[8]]
                );
            }
            
            // i = 0, bulk of 2d plane
            for j in 1..(s_idx+1) {
                for k in -s_idx..(s_idx+1) {
                    let n = j as f64;
                    let o = k as f64;
                    support.push(
                        vec![n*q_lat[3] + o*q_lat[6],
                             n*q_lat[4] + o*q_lat[7],
                             n*q_lat[5] + o*q_lat[8]
                        ]
                    );
                }
            }
            
            // Handle bulk of 3d
            for i in 1..(s_idx+1) {
                for j in -s_idx..(s_idx+1) {
                    for k in -s_idx..(s_idx+1) {
                        let m = i as f64;
                        let n = j as f64;
                        let o = k as f64;
                        support.push(
                            vec![m*q_lat[0] + n*q_lat[3] + o*q_lat[6],
                             m*q_lat[1] + n*q_lat[4] + o*q_lat[7],
                             m*q_lat[2] + n*q_lat[5] + o*q_lat[8]]
                        );
                    }
                }
            }
        }
        _ => panic!("Haven't implemented that dimension!"),
    }
    return support;
}

//https://stackoverflow.com/questions/50056778/how-can-you-easily-borrow-a-vecvect-as-a-t
fn one_configuration(
                    config: &(dyn Config + Sync),
                     spectral_density: &mut [f64],
                     spectral_density_squared: &mut [f64],
                     support: &[Vec<f64>],
                     opt: &Opt)
{
    if opt.run_timing { // Don't parallelize to get a timing estimate
        if opt.compute_structure_factor {
            for (i, q, val, val2) in izip!(0..support.len(),
                                           support,
                                           spectral_density,
                                           spectral_density_squared
                                          )
            {
                if i%1000 == 0 {
                    println!("working on iteration {}", i);
                }
                let s = config.sf_one_wavevector(q);
                *val += s;
                *val2 += s*s;
            }
        } else {
            for (i, q, val, val2) in izip!(0..support.len(),
                                           support,
                                           spectral_density,
                                           spectral_density_squared
                                          )
            {
                if i%1000 == 0 {
                    println!("working on iteration {}", i);
                }
                let s = config.one_wavevector(q);
                *val += s;
                *val2 += s*s;
            }
        }
    } else { // Parallelize for speed
        if opt.compute_structure_factor {
            (support, spectral_density, spectral_density_squared).into_par_iter()
                .for_each(|(q, val, val2)| {
                        let s = config.sf_one_wavevector(q);
                        *val += s;
                        *val2 += s*s; 
                    }
                ); 
        } else {
            (support, spectral_density, spectral_density_squared).into_par_iter()
                .for_each(|(q, val, val2)| {
                        let s = config.one_wavevector(q);
                        *val += s;
                        *val2 += s*s;
                    }
                );
        }    
    }
}

fn write_results(
                        config: &dyn Config,
                        spectral_density: &[f64],
                        sigma: &[f64],
                        support: &[Vec<f64>],
                        opt: &Opt
)
{
    let q_lat = config.get_reciprocal_cell();
    let dim = config.get_dimension();
    // See 
    // https://stackoverflow.com/questions/25278248/rust-structs-with-nullable-option-fields
    // for borrow checker discussion
    // https://stackoverflow.com/questions/31567708/difference-in-mutability-between-reference-and-box
    let mut out_dest: Box<dyn Write> = match &opt.outfile {
        Some(p) => {Box::new(
                        BufWriter::new(
                            File::create(p)
                            .expect("Couldn't make outfile")
                            )
                        )
                    }
        None => Box::new(std::io::stdout()),
    };
    writeln!(&mut *out_dest, "{}", opt.max_vec)
        .expect("Couldn't write to outfile");
    
    match dim {
        2 => {
            for i in 0..2 {
                writeln!(&mut *out_dest, "{}\t{}",
                    q_lat[2*i], q_lat[2*i+1])
                    .expect("Couldn't write to outfile");
            }
            for (q, chi, o) in izip!(support, spectral_density, sigma) {
                writeln!(&mut *out_dest,
                         "{}\t{}\t{}\t{}",
                         q[0], q[1], chi, o)
                .expect("Couldn't write to outfile");
            }
        }
        3 => {
            for i in 0..3 {
                writeln!(&mut *out_dest, "{}\t{}\t{}",
                    q_lat[3*i], q_lat[3*i+1], q_lat[3*i+2])
                    .expect("Couldn't write to outfile");
            }
            for (q, chi, o) in izip!(support, spectral_density, sigma) {
                writeln!(&mut *out_dest,
                         "{}\t{}\t{}\t{}\t{}",
                         q[0], q[1], q[2], chi, o)
                .expect("Couldn't write to outfile");
            }
        }
        _ => panic!("Haven't implemented output for that dimension yet!"),
    }
}
       
//https://users.rust-lang.org/t/sending-trait-objects-between-threads/2374
fn parse_config(path: &PathBuf, opt: &Opt) -> Box<dyn Config + Sync> {
    // Get first line of file
    let mut bufs = String::new();

    {   
        let mut bufr = BufReader::new(
            File::open(path).expect("io error")
        );

        bufr.read_line(&mut bufs).expect("io line error");
    }

    let tokens: Vec<&str> = bufs.split_whitespace().collect();

    // Determine if file is asc type or Donev type
    if opt.asc {
        if tokens[2] == "Sphere" {
            Box::new(<SphereConfig<U3>>::from_file_asc(path))
        } else if tokens[2] == "Disk" {
            Box::new(<SphereConfig<U2>>::from_file_asc(path))
        } else {
            unimplemented!()
        }
    } else {
        if tokens[0] == "3" && tokens[1] == "HS" {
            //https://stackoverflow.com/questions/44483876/update-to-rust-1-18-broke-compilation-e0034-multiple-applicable-items-in-scope
            //https://doc.rust-lang.org/book/ch19-03-advanced-traits.html
            Box::new(<SphereConfig<U3>>::from_file(path))
        } else if tokens[0] == "2" && tokens[1] == "HS" {
            Box::new(<SphereConfig<U2>>::from_file(path))
        } else if tokens[0] == "3" && tokens[1] == "HE" {
            Box::new(<EllipsoidConfig<U3>>::from_file(path))
        } else {
            unimplemented!()
        }
    }
}

// Called by main() -------------------------------------------------------

pub fn spectral_density_cmdline(opt: &Opt) {
    // Decide the type and dimension
    // of given input files

    let first_config = parse_config(&opt.infiles[0], opt);
            
    // Initialize data structure
    let support = make_support(
        first_config.get_dimension(),
        first_config.get_reciprocal_cell(),
        opt.max_vec
    );
    
    let mut spectral_density = vec![0.0; support.len()];
    let mut spectral_density_squared = vec![0.0; support.len()];
                    
    // Ensemble sum
    for (i, path) in opt.infiles.iter().enumerate() {
        println!("Processing file: {}", i);
        let config = parse_config(path, opt);
        one_configuration(&*config,
                           &mut spectral_density,
                           &mut spectral_density_squared,
                           &support,
                           &opt
        );
    }
    
    // Normalize by extensiveness of the system
    if opt.compute_structure_factor {
        for (val, val2) in izip!(&mut spectral_density, &mut spectral_density_squared) {
            let n_points = first_config.get_n_points() as f64;
            *val /= n_points;
            *val2 /= n_points*n_points;
        }
    } else {
        for (val, val2) in izip!(&mut spectral_density, &mut spectral_density_squared) {
            let vol = first_config.get_vol();
            *val /= first_config.get_vol();
            *val2 /= vol*vol;
        }
    }


    // Normalize by number of realizations
    // Spectral density vector now contains final value
    let n_ens = opt.infiles.len() as f64;
    for val in &mut spectral_density {
        *val /= n_ens;
    }

    // Compute std error of mean
    // Similar to what Ge uses, but
    // a little different once angularly averaged
    let mut sigma = vec![0.0; support.len()];
    for (s, val2, o) in izip!(&spectral_density, &spectral_density_squared, &mut sigma) {
        *o = ((*val2 - n_ens*s*s)/(n_ens - 1.0)).sqrt();
        *o /= n_ens.sqrt();
    }

    write_results(&*first_config,
                  &spectral_density,
                  &sigma,
                  &support,
                  &opt
    );
}
