use std::path::PathBuf;
use std::fs::File;
use std::io::{Read, BufRead, BufReader};
//https://stackoverflow.com/questions/54289071/what-are-the-valid-path-roots-in-the-use-keyword
use crate::{
        form_factor3, reciprocal_lattice3, volume3,
        reciprocal_lattice2, volume2,
        Config
};
use nalgebra::{U2, U3, DimName};
use std::marker::PhantomData;
use nalgebra::{Matrix2, Vector2, Matrix3, Vector3};

#[derive(Debug)]
pub struct SphereConfig<T: DimName> { // T is dimension
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

// General code ---------------------------------------------------

impl<T: DimName> SphereConfig<T> {
    pub fn from_file_asc(path: &PathBuf) -> SphereConfig<T> {
        // Get the dimension from type info
        let dim = T::dim();
        
        // Prepare to read file line by line
        let mut bufr = BufReader::new(File::open(path)
                                  .expect("io error")
        );
        let mut bufs = String::new();

        // Read and check dimension against type info
        bufr.read_line(&mut bufs).unwrap();
        let dim_check: usize = bufs.trim().split_whitespace()
                      .next().unwrap()
                      .parse().unwrap();
        assert!(dim_check == dim);
        bufs.clear();
        
        // Read unit cell
        bufr.read_line(&mut bufs).unwrap();
        let unit_cell: Vec<f64> = bufs.trim().split_whitespace()
                            .map(|x| x.parse().unwrap())
                            .collect();
        bufs.clear();
        
        // Read particle list
        let mut config: Vec<f64> = Vec::new();
        let mut n_points = 0;
        // Decided to do it this way because generics are hard
        // https://stackoverflow.com/questions/53225972/how-do-i-create-a-struct-containing-a-nalgebra-matrix-with-higher-dimensions
        // https://github.com/rustsim/nalgebra/issues/580
        // https://discourse.nphysics.org/t/using-nalgebra-in-generics/90
        match dim {
            2 => {
                let cell_mat = Matrix2::from_column_slice(&unit_cell);
                for line in bufr.lines() {
                    n_points += 1;
                    let rel_coord: Vec<f64> = line.unwrap()
                                                  .trim()
                                                  .split_whitespace()
                                                  .map(|x| x.parse().unwrap())
                                                  .collect();
                    let rel_coord_vect = Vector2::from_column_slice(&rel_coord[0..dim]);
                    let global_coord_vect = cell_mat*rel_coord_vect;
                    config.extend_from_slice(global_coord_vect.as_slice());
                    config.push(rel_coord[dim]);
                }
            }
            3 => {
                let cell_mat = Matrix3::from_column_slice(&unit_cell);
                for line in bufr.lines() {
                    n_points += 1;
                    let rel_coord: Vec<f64> = line.unwrap()
                                                  .trim()
                                                  .split_whitespace()
                                                  .map(|x| x.parse().unwrap())
                                                  .collect();
                    let rel_coord_vect = Vector3::from_column_slice(&rel_coord[0..dim]);
                    let global_coord_vect = cell_mat*rel_coord_vect;
                    config.extend_from_slice(global_coord_vect.as_slice());
                    config.push(rel_coord[dim]);
                }
            }
            _ => unimplemented!(),
        }

        // Compute other fields
        let vol = match dim {
            2 => volume2(&unit_cell),
            3 => volume3(&unit_cell),
            _ => unimplemented!(),
        };
        
        let reciprocal_cell = match dim {
            2 => reciprocal_lattice2(&unit_cell),
            3 => reciprocal_lattice3(&unit_cell),
            _ => unimplemented!(),
        };

        SphereConfig { n_points, config, unit_cell, vol, reciprocal_cell, phantom: PhantomData }
    }
}

// 2D Code --------------------------------------------------------

impl SphereConfig<U2> {
    pub fn from_file(path: &PathBuf) -> SphereConfig<U2> {
        let dim = U2::dim();
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
    
    fn get_dimension(&self) -> usize { U2::dim() }
    
    fn get_n_points(&self) -> usize { self.n_points }
    
    fn get_vol(&self) -> f64 { self.vol }
}

// 3D Code --------------------------------------------------------

impl SphereConfig<U3> {
    // Thanks to 
    // https://users.rust-lang.org/t/read-a-file-line-by-line/1585
    pub fn from_file(path: &PathBuf) -> SphereConfig<U3> {
        let dim = U3::dim();
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
        U3::dim()
    }

    fn get_n_points(&self) -> usize {
        self.n_points
    }

    fn get_vol(&self) -> f64 {
        self.vol
    }

}
