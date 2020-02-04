use structopt::StructOpt;
use std::path::PathBuf;
use nalgebra::{Matrix2, Vector2, Matrix3, Vector3};
use std::fs::File;
use std::io::{BufRead, BufReader, Write, BufWriter};
use rayon::prelude::*;
use typenum::{U2, U3};

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
}

// Traits --------------------------------------------------------
// https://github.com/rust-lang/rfcs/issues/2035
// Common methods to all configurations
trait Config {
    fn get_reciprocal_cell(&self) -> &[f64];
    fn one_wavevector(&self, q: &[f64], val: &mut f64);
    fn sf_one_wavevector(&self, q: &[f64], val: &mut f64);
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

fn make_support(dim: usize, q_lat: &[f64], max_idx: usize)
    -> Vec<Vec<f64>> 
{
    let mut support = Vec::new();
    match dim {
        2 => {
            let s_idx = max_idx as isize;
            for i in -s_idx..(s_idx+1) {
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
            for i in -s_idx..(s_idx+1) {
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
                     support: &[Vec<f64>],
                     opt: &Opt)
{
    if opt.run_timing { // Don't parallelize to get a timing estimate
        if opt.compute_structure_factor {
            for (i, (q, val)) in support.iter()
                .zip(spectral_density.iter_mut())
                .enumerate() 
            {
                if i%1000 == 0 {
                    println!("working on iteration {}", i);
                }
                config.sf_one_wavevector(q, val);
            }
        } else {
            for (i, (q, val)) in support.iter()
                .zip(spectral_density.iter_mut())
                .enumerate() 
            {
                if i%1000 == 0 {
                    println!("working on iteration {}", i);
                }
                config.one_wavevector(q, val);
            }
        }
    } else { // Parallelize for speed
        if opt.compute_structure_factor {
            support.par_iter().zip(spectral_density.par_iter_mut())
                .for_each(|(q, val)| config.sf_one_wavevector(q, val));
        } else {
            support.par_iter().zip(spectral_density.par_iter_mut())
                .for_each(|(q, val)| config.one_wavevector(q, val));
        }    
    }
}

fn write_results(
                        config: &dyn Config,
                        spectral_density: &[f64],
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
            for (q, chi) in support.iter()
                .zip(spectral_density.iter()) 
            {
                writeln!(&mut *out_dest,
                         "{}\t{}\t{}",
                         q[0], q[1], chi)
                .expect("Couldn't write to outfile");
            }
        }
        3 => {
            for i in 0..3 {
                writeln!(&mut *out_dest, "{}\t{}\t{}",
                    q_lat[3*i], q_lat[3*i+1], q_lat[3*i+2])
                    .expect("Couldn't write to outfile");
            }
            for (q, chi) in support.iter()
                .zip(spectral_density.iter()) 
            {
                writeln!(&mut *out_dest,
                         "{}\t{}\t{}\t{}",
                         q[0], q[1], q[2], chi)
                .expect("Couldn't write to outfile");
            }
        }
        _ => panic!("Haven't implemented output for that dimension yet!"),
    }
}
       
//https://users.rust-lang.org/t/sending-trait-objects-between-threads/2374
fn parse_config(path: &PathBuf) -> Box<dyn Config + Sync> {
    let mut bufs = String::new();

    {   
        let mut bufr = BufReader::new(
            File::open(path).expect("io error")
        );

        bufr.read_line(&mut bufs).expect("io line error");
    }

    let tokens: Vec<&str> = bufs.split_whitespace().collect();

    if tokens[0] == "3" && tokens[1] == "HS" {
        //https://stackoverflow.com/questions/44483876/update-to-rust-1-18-broke-compilation-e0034-multiple-applicable-items-in-scope
        //https://doc.rust-lang.org/book/ch19-03-advanced-traits.html
        Box::new(<SphereConfig<U3>>::from_file(path))
    } else if tokens[0] == "2" && tokens[1] == "HS" {
        Box::new(<SphereConfig<U2>>::from_file(path))
    } else if tokens[0] == "3" && tokens[1] == "HE" {
        Box::new(<EllipsoidConfig<U3>>::from_file(path))
    } else {
        panic!("Haven't implemented that dimension/shape");
    }
}

// Called by main() -------------------------------------------------------

pub fn spectral_density_cmdline(opt: &Opt) {
    // Decide the type and dimension
    // of given input files

    let first_config = parse_config(&opt.infiles[0]);
            
    // Initialize data structure
    let support = make_support(
        first_config.get_dimension(),
        first_config.get_reciprocal_cell(),
        opt.max_vec
    );
    
    let mut spectral_density = vec![0.0; support.len()];
                    
    // Ensemble sum
    for (i, path) in opt.infiles.iter().enumerate() {
        println!("Processing file: {}", i);
        let config = parse_config(path);
        one_configuration(&*config,
                           &mut spectral_density,
                           &support,
                           &opt
        );
    }
    
    // Normalize by extensiveness of the system
    if opt.compute_structure_factor {
        for val in &mut spectral_density {
            *val /= first_config.get_n_points() as f64;
        }
    } else {
        for val in &mut spectral_density {
            *val /= first_config.get_vol();
        }
    }


    // Normalize by number of realizations
    let n_ens = opt.infiles.len() as f64;
    for val in &mut spectral_density {
        *val /= n_ens;
    }

    write_results(&*first_config,
                  &spectral_density,
                  &support,
                  &opt
    );
}
