use std::path::PathBuf;
use std::fs::File;
use std::io::{Read, BufRead, BufReader};
//https://stackoverflow.com/questions/54289071/what-are-the-valid-path-roots-in-the-use-keyword
use crate::{
        form_factor3, reciprocal_lattice3, volume3,
        reciprocal_lattice2, volume2,
        Config
};
use typenum::{U2, U3, Unsigned, NonZero};
use std::marker::PhantomData;

#[derive(Debug)]
pub struct SphereConfig<T: Unsigned + NonZero> { // T is dimension
    pub n_points: usize,
    // Coordinates + radius, flat storage:
    // x1 y1 z1 R1 x2 y2 z2 R2 ...
    pub config: Vec<f64>,
    // Unit cell, flat storage:
    // u11 u12 u13 u21 u22 u23 ...
    pub unit_cell: Vec<f64>,
    pub vol: f64,
    pub reciprocal_cell: Vec<f64>,
    phantom: PhantomData<T>,
}

// 2D Code --------------------------------------------------------

impl SphereConfig<U2> {
    pub fn from_file(path: &PathBuf) -> SphereConfig<U2> {
        let dim = U2::to_usize();
        let mut bufr = BufReader::new(File::open(path)
                                  .expect("io error")
        );
        let mut bufs = String::new();

        // check dimension
        bufr.read_line(&mut bufs).expect("io line error");
        let dim_check: usize = bufs.split_whitespace()
            .next().unwrap()
            .parse()
            .expect("parse dim failed")
        ;
        assert!(dim_check == dim);

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

        // Compute volume
        let vol = volume2(&unit_cell);
        
        // Make reciprocal cell
        let reciprocal_cell = reciprocal_lattice2(&unit_cell);
        
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

        SphereConfig {n_points, config, unit_cell, vol, reciprocal_cell, phantom: PhantomData} 
    }
}

impl Config for SphereConfig<U2> {
    fn get_reciprocal_cell(&self) -> &[f64] { &self.reciprocal_cell }
    
    fn one_wavevector(&self, _q: &[f64],) -> f64 {unimplemented!()}
    
    fn sf_one_wavevector(&self, q: &[f64]) -> f64 {
        let mut real: f64 = 0.0;
        let mut imag: f64 = 0.0;
        for p in self.config.chunks_exact(3) {
            let dot = -(q[0]*p[0] + q[1]*p[1]);
            real += dot.cos();
            imag += dot.sin();
        }
        real*real + imag*imag
    }
    
    fn get_dimension(&self) -> usize { U2::to_usize() }
    
    fn get_n_points(&self) -> usize { self.n_points }
    
    fn get_vol(&self) -> f64 { self.vol }
}

// 3D Code --------------------------------------------------------

impl SphereConfig<U3> {
    // Thanks to 
    // https://users.rust-lang.org/t/read-a-file-line-by-line/1585
    pub fn from_file(path: &PathBuf) -> SphereConfig<U3> {
        let dim = U3::to_usize();
        let mut bufr = BufReader::new(File::open(path)
                                  .expect("io error")
        );
        let mut bufs = String::new();

        // check dimension
        bufr.read_line(&mut bufs).expect("io line error");
        let dim_check: usize = bufs.split_whitespace()
            .next().unwrap()
            .parse()
            .expect("parse dim failed")
        ;
        assert!(dim_check == dim);

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

        // Compute volume
        let vol = volume3(&unit_cell);
        
        // Make reciprocal cell
        let reciprocal_cell = reciprocal_lattice3(&unit_cell);
        
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

        SphereConfig {n_points, config, unit_cell, vol, reciprocal_cell, phantom: PhantomData} 
    }
}

impl Config for SphereConfig<U3> {

    fn get_reciprocal_cell(&self) -> &[f64] {
        &self.reciprocal_cell
    }

    fn one_wavevector(&self, q: &[f64]) -> f64
    {
        //println!("Working on wavevector: {:?}", q);
        let mut real: f64 = 0.0;
        let mut imag: f64 = 0.0;
        let q_amp = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2]).sqrt();
        
        for p in self.config.chunks_exact(4) {
            let dot = -(q[0]*p[0] 
                        + q[1]*p[1] 
                        + q[2]*p[2]);
            let m = form_factor3( q_amp, p[3] );
            real += m*dot.cos();
            imag += m*dot.sin();
        }
        
        real*real + imag*imag
    }

    fn sf_one_wavevector(&self, q: &[f64]) -> f64
    {
        //println!("Working on wavevector: {:?}", q);
        let mut real: f64 = 0.0;
        let mut imag: f64 = 0.0;
        for p in self.config.chunks_exact(4) {
            let dot = -(q[0]*p[0] + q[1]*p[1] + q[2]*p[2]);
            real += dot.cos();
            imag += dot.sin();
        }
        real*real + imag*imag
    }

    fn get_dimension(&self) -> usize {
        U3::to_usize()
    }

    fn get_n_points(&self) -> usize {
        self.n_points
    }

    fn get_vol(&self) -> f64 {
        self.vol
    }

}
