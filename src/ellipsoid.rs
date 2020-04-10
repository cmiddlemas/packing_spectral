use std::path::PathBuf;
use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::{form_factor3, reciprocal_lattice3, volume3, Config};
use typenum::{U3, Unsigned, NonZero};
use std::marker::PhantomData;
use nalgebra::{Quaternion, UnitQuaternion, Point3};

// Holds arbitrary ellipsoids
#[derive(Debug)]
pub struct EllipsoidConfig<T: Unsigned + NonZero> {
    pub n_points: usize,
    // Storage convention:
    // coord    rotation    axes
    // x1 y1 z1 q11 q21 q31 q41 a1 b1 c1 x2 ...
    pub config: Vec<f64>,
    pub unit_cell: Vec<f64>,
    pub vol: f64,
    pub reciprocal_cell: Vec<f64>,
    phantom: PhantomData<T>,
}

// 3D code --------------------------------------------------------

impl EllipsoidConfig<U3> {
    pub fn from_file(path: &PathBuf) -> EllipsoidConfig<U3> {
        let mut bufr = BufReader::new(File::open(path)
                                      .expect("io error")
        );
        let mut bufs = String::new();

        // Check dimension
        bufr.read_line(&mut bufs).expect("io line error");
        let dim: usize = bufs.split_whitespace()
            .next().unwrap()
            .parse()
            .expect("parse dim failed");
        assert!(dim == U3::to_usize());

        // Get number of points
        bufr.read_line(&mut bufs).expect("io line error");
        bufs.clear();
        bufr.read_line(&mut bufs).expect("io line error");
        let n_points: usize = bufs.trim().parse()
            .expect("parse n_points failed");

        println!("Read {} as number of points", n_points);

        // Get ellipsoid axes
        bufs.clear();
        bufr.read_line(&mut bufs).expect("io line error");
        let mut axes: Vec<f64> = bufs.split_whitespace()
                                .map(|x| x.parse()
                                    .expect("Parse ellipsoid axes failed")
                                )
                                .collect()
        ;
        axes.resize(3, 0.0);

        // Get unit cell
        bufs.clear();
        let mut unit_cell: Vec<f64> = Vec::new();
        while unit_cell.len() < dim*dim {
            bufr.read_line(&mut bufs).expect("io line error");
            unit_cell.extend(
                bufs.split_whitespace()
                    .map(|x| x.parse::<f64>()
                        .expect("parse unit_cell failed")
                    )
            );
        }
        // Check that we read exactly the right number of entries
        assert!(unit_cell.len() == dim*dim);

        let vol = volume3(&unit_cell);
        
        let reciprocal_cell = reciprocal_lattice3(&unit_cell);

        // Get config
        bufr.read_line(&mut bufs).expect("io line error");
        bufs.clear();
        let mut config: Vec<f64> = Vec::new();
        for line in bufr.lines() {
            config.extend(
                line.expect("io line error")
                    .split_whitespace()
                    .map(|x| x.parse::<f64>()
                        .expect("error in parsing config")
                    )
            );
            config.extend_from_slice(&axes);
        }

        // Consistency check
        // Config is coordinates + quaterion + axes
        assert!(config.len() == 10*n_points);

        EllipsoidConfig { 
            n_points,
            config,
            unit_cell,
            vol,
            reciprocal_cell,
            phantom: PhantomData
        }

    }
}

// q: [q_x q_y q_z]
// t: [a b c d a_x a_y a_z] (quaternion + axes)
fn ellipsoid_factor(q: &[f64], t: &[f64]) -> f64 {
    let quat: UnitQuaternion<f64> = UnitQuaternion::from_quaternion(
                            Quaternion::new(t[0], t[1], t[2], t[3])
    );
    let point_q: Point3<f64> = Point3::new(q[0], q[1], q[2]);
    let rotate_q = quat.inverse_transform_point(&point_q);
    let q_amp = (
                    (t[4]*rotate_q[0]).powi(2)
                +   (t[5]*rotate_q[1]).powi(2)
                +   (t[6]*rotate_q[2]).powi(2)
    ).sqrt();
    let scale_prefactor = t[4]*t[5]*t[6];
    scale_prefactor*form_factor3(q_amp, 0.5)
}

impl Config for EllipsoidConfig<U3> {
    
    fn get_reciprocal_cell(&self) -> &[f64] {
        &self.reciprocal_cell
    }
    
    fn get_dimension(&self) -> usize {
        U3::to_usize()
    }

    fn one_wavevector(&self, q: &[f64]) -> f64
    {
        let mut real: f64 = 0.0;
        let mut imag: f64 = 0.0;
        for p in self.config.chunks_exact(10) {
            let dot = -( q[0]*p[0]
                        +q[1]*p[1]
                        +q[2]*p[2]);
            real += dot.cos() * ellipsoid_factor(q, &p[3..10]);
            imag += dot.sin() * ellipsoid_factor(q, &p[3..10]);
        }
        real*real + imag*imag
    }

    fn sf_one_wavevector(&self, q: &[f64]) -> f64
    {
        let mut real: f64 = 0.0;
        let mut imag: f64 = 0.0;
        for p in self.config.chunks_exact(10) {
            let dot = -( q[0]*p[0]
                        +q[1]*p[1]
                        +q[2]*p[2] );
            real += dot.cos();
            imag += dot.sin();
        }
        real*real + imag*imag
    }

    fn get_n_points(&self) -> usize {
        self.n_points
    }

    fn get_vol(&self) -> f64 {
        self.vol
    }

}

#[cfg(test)]
mod tests{
    use super::*;

    // Following two tests computed with help of 
    // Mathematica 11.2.0.0 Student Edition
    #[test]
    fn test_ellipsoid_factor() {
        let trig: f64 = 1.0/(2.0_f64.sqrt());
        let val = ellipsoid_factor(
            &[1.0, 0.0, 0.0],
            &[trig, 0.0, trig, 0.0, 2.0, 1.0, 0.5]
        );
        if (val - 0.520334).abs() > 0.00001 {
            panic!()
        }
    }

    #[test]
    fn test_ellipsoid_factor2() {
        let val = ellipsoid_factor(
            &[1.0, 0.0, 0.0],
            &[0.9238795325112867, 0.0, 0.3826834323650898, 0.0, 2.0, 1.0, 0.5]
        );
        if (val - 0.496305).abs() > 0.00001 {
            println!("{}", val);
            panic!()
        }
    }

    #[test]
    fn test_ellipsoid_factor3() {
        let val = ellipsoid_factor(
            &[1.0, 0.0, 1.0],
            &[0.9659258262890682, 0.0, 0.2588190451025207, 0.0, 2.0, 1.0, 0.5]
        );
        if (val - 0.510594).abs() > 0.00001 {
            println!("{}", val);
            panic!()
        }
    }
    #[test]
    fn test_ellipsoid_factor4() {
        let val = ellipsoid_factor(
            &[3.0, 2.0, 1.0],
            &[0.9659258262890682, 0.0, 0.2588190451025207, 0.0, 2.0, 1.0, 0.5]
        );
        if (val - 0.278123).abs() > 0.00001 {
            println!("{}", val);
            panic!()
        }
    }

    #[test]
    fn test_ellipsoid_factor5() {
        let val = ellipsoid_factor(
            &[-3.0, 2.0, 5.0],
            &[0.9510565162951535, 0.30901699437494745, 0.0, 0.0, 2.0, 1.0, 0.5]
        );
        if (val - 0.0643571).abs() > 0.000001 {
            println!("{}", val);
            panic!()
        }
    }
}
