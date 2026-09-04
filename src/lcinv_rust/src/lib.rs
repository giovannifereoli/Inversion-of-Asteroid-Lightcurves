//! Rust kernels for `lcinv`.
//!
//! Only the two hot spots of Section 2's ray tracer live here.  Everything else
//! is fast enough in NumPy, and the Minkowski solve is dominated by Qhull,
//! which is already C.
//!
//! * [`build_blockers`] finds, for each facet, "which vertices are above [the]
//!   facet's local horizon and which facets connected to these vertices are
//!   facing this facet".  In Python this is a loop over facets.
//! * [`trace_fractions`] runs the Moller-Trumbore occlusion test over every
//!   (observation, facet, test point, blocker) combination.  This dominates
//!   nonconvex inversion, where one Levenberg-Marquardt Jacobian needs about
//!   fifty full traces.
//!
//! Both mirror the Python implementations exactly, tolerances included, so the
//! two can be compared element by element - which the test suite does.

use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline(always)]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn row(a: &ArrayView2<'_, f64>, i: usize) -> [f64; 3] {
    [a[[i, 0]], a[[i, 1]], a[[i, 2]]]
}

/// Local blockers for every facet - Section 2's precomputation.
///
/// Returns `(pair_facet, pair_blocker, hull_mask, blocker_height)`.  The first
/// two form the flattened (facet, blocker) pair list; `hull_mask` marks facets
/// with no vertex above their local horizon, which "belong to the convex hull"
/// and can never be blocked; `blocker_height` is the mean height of the
/// above-horizon vertices, which Section 4's regularisation needs.
#[pyfunction]
fn build_blockers<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    facets: PyReadonlyArray2<'py, i64>,
    normals: PyReadonlyArray2<'py, f64>,
    centroids: PyReadonlyArray2<'py, f64>,
    eps: f64,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let verts = vertices.as_array();
    let faces = facets.as_array();
    let nor = normals.as_array();
    let cen = centroids.as_array();
    let n_f = faces.shape()[0];
    let n_v = verts.shape()[0];

    // vertex -> incident facets, as CSR.
    let mut offsets = vec![0usize; n_v + 1];
    for f in 0..n_f {
        for k in 0..3 {
            offsets[faces[[f, k]] as usize + 1] += 1;
        }
    }
    for i in 0..n_v {
        offsets[i + 1] += offsets[i];
    }
    let mut fill = offsets.clone();
    let mut incident = vec![0i64; offsets[n_v]];
    for f in 0..n_f {
        for k in 0..3 {
            let v = faces[[f, k]] as usize;
            incident[fill[v]] = f as i64;
            fill[v] += 1;
        }
    }

    let per_facet: Vec<(Vec<i64>, bool, f64)> = py.detach(|| {
        (0..n_f)
            .into_par_iter()
            .map(|j| {
                let nj = row(&nor, j);
                let plane = dot(nj, row(&cen, j));
                let mut above: Vec<usize> = Vec::new();
                let mut height_sum = 0.0;
                for v in 0..n_v {
                    let h = dot(nj, row(&verts, v)) - plane;
                    if h > eps {
                        above.push(v);
                        height_sum += h;
                    }
                }
                if above.is_empty() {
                    // Section 2: no vertex above the horizon => a hull facet.
                    return (Vec::new(), true, 0.0);
                }
                let mut cand: Vec<i64> = Vec::new();
                for &v in &above {
                    cand.extend_from_slice(&incident[offsets[v]..offsets[v + 1]]);
                }
                cand.sort_unstable();
                cand.dedup();
                let cj = row(&cen, j);
                let blockers: Vec<i64> = cand
                    .into_iter()
                    .filter(|&k| {
                        let k = k as usize;
                        if k == j {
                            return false;
                        }
                        // "... and which facets connected to these vertices are
                        // facing this facet".
                        let ck = row(&cen, k);
                        let d = [cj[0] - ck[0], cj[1] - ck[1], cj[2] - ck[2]];
                        dot(d, row(&nor, k)) > eps
                    })
                    .collect();
                (blockers, false, height_sum / above.len() as f64)
            })
            .collect()
    });

    let mut pair_facet: Vec<i64> = Vec::new();
    let mut pair_blocker: Vec<i64> = Vec::new();
    let mut hull = Array1::<bool>::from_elem(n_f, false);
    let mut height = Array1::<f64>::zeros(n_f);
    for (j, (blockers, is_hull, h)) in per_facet.into_iter().enumerate() {
        hull[j] = is_hull;
        height[j] = h;
        for b in blockers {
            pair_facet.push(j as i64);
            pair_blocker.push(b);
        }
    }
    Ok((
        Array1::from(pair_facet).into_pyarray(py),
        Array1::from(pair_blocker).into_pyarray(py),
        hull.into_pyarray(py),
        height.into_pyarray(py),
    ))
}

/// Moller-Trumbore: does the ray `orig + t * dir` meet the triangle?
#[inline(always)]
fn hits(orig: [f64; 3], dir: [f64; 3], v0: [f64; 3], e1: [f64; 3], e2: [f64; 3], eps: f64) -> bool {
    let pvec = cross(dir, e2);
    let det = dot(e1, pvec);
    if det.abs() <= 1e-14 {
        return false;
    }
    let inv = 1.0 / det;
    let tvec = [orig[0] - v0[0], orig[1] - v0[1], orig[2] - v0[2]];
    let u = dot(tvec, pvec) * inv;
    if u < 0.0 {
        return false;
    }
    let qvec = cross(tvec, e1);
    let v = dot(qvec, dir) * inv;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    dot(qvec, e2) * inv > eps
}

/// Visible-and-illuminated fraction of every facet, for every observation.
///
/// Returns an `(N, F)` array in `[0, 1]`: zero where `mu <= 0` or `mu0 <= 0`,
/// one for convex-hull facets, and otherwise the fraction of test points
/// unshadowed towards *both* the Earth and the Sun.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn trace_fractions<'py>(
    py: Python<'py>,
    earth: PyReadonlyArray2<'py, f64>,
    sun: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    v0: PyReadonlyArray2<'py, f64>,
    e1: PyReadonlyArray2<'py, f64>,
    e2: PyReadonlyArray2<'py, f64>,
    samples: PyReadonlyArray3<'py, f64>,
    pair_start: PyReadonlyArray1<'py, i64>,
    pair_blocker: PyReadonlyArray1<'py, i64>,
    hull_mask: PyReadonlyArray1<'py, bool>,
    eps: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let e = earth.as_array();
    let s = sun.as_array();
    let nor = normals.as_array();
    let av0 = v0.as_array();
    let ae1 = e1.as_array();
    let ae2 = e2.as_array();
    let smp: ArrayView3<f64> = samples.as_array();
    let starts: ArrayView1<i64> = pair_start.as_array();
    let blockers: ArrayView1<i64> = pair_blocker.as_array();
    let hull: ArrayView1<bool> = hull_mask.as_array();

    let n_obs = e.shape()[0];
    let n_f = nor.shape()[0];
    let n_s = smp.shape()[1];

    let flat: Vec<f64> = py.detach(|| {
        (0..n_obs)
            .into_par_iter()
            .flat_map(|i| {
                let ed = row(&e, i);
                let sd = row(&s, i);
                let mut out = vec![0.0f64; n_f];
                for j in 0..n_f {
                    let nj = row(&nor, j);
                    if dot(nj, ed) <= 0.0 || dot(nj, sd) <= 0.0 {
                        continue;
                    }
                    if hull[j] {
                        out[j] = 1.0;
                        continue;
                    }
                    let lo = starts[j] as usize;
                    let hi = starts[j + 1] as usize;
                    let mut free = 0usize;
                    for t in 0..n_s {
                        let orig = [smp[[j, t, 0]], smp[[j, t, 1]], smp[[j, t, 2]]];
                        let mut blocked = false;
                        for dir in [ed, sd] {
                            for p in lo..hi {
                                let k = blockers[p] as usize;
                                // A ray leaving a closed surface can only be
                                // stopped where it re-enters, so a blocker has
                                // to face back towards it.
                                if dot(row(&nor, k), dir) >= 0.0 {
                                    continue;
                                }
                                if hits(orig, dir, row(&av0, k), row(&ae1, k), row(&ae2, k), eps) {
                                    blocked = true;
                                    break;
                                }
                            }
                            if blocked {
                                break;
                            }
                        }
                        if !blocked {
                            free += 1;
                        }
                    }
                    out[j] = free as f64 / n_s as f64;
                }
                out
            })
            .collect()
    });

    let result = Array2::from_shape_vec((n_obs, n_f), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(result.into_pyarray(py))
}


/// Eq. (4) for the Lommel-Seeliger + Lambert law, fused.
///
/// `A[i, j] = f(alpha_i) * mu * mu0 * (1 / (mu + mu0) + c) * varpi_j`, zero
/// wherever `mu <= 0` or `mu0 <= 0`, with `mu = E_i . n_j` and
/// `mu0 = E0_i . n_j`.
///
/// The NumPy version materialises half a dozen `(N, M)` temporaries - two dot
/// products, a mask, a reciprocal, a product and a `where` - and building this
/// matrix is where a convex inversion with a free pole spends most of its time,
/// because every trial pole needs a fresh one.  Here it is one pass, in
/// parallel over observations.
///
/// `phase` is `[a, d, k]` of `f(alpha) = a exp(-alpha/d) + k alpha + 1`;
/// pass `use_phase = false` to drop it.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn design_matrix_lsl<'py>(
    py: Python<'py>,
    earth: PyReadonlyArray2<'py, f64>,
    sun: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    albedo: PyReadonlyArray1<'py, f64>,
    alpha: PyReadonlyArray1<'py, f64>,
    lambert_weight: f64,
    phase: [f64; 3],
    use_phase: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let e = earth.as_array();
    let s = sun.as_array();
    let nor = normals.as_array();
    let alb: ArrayView1<f64> = albedo.as_array();
    let al: ArrayView1<f64> = alpha.as_array();
    let n_obs = e.shape()[0];
    let n_f = nor.shape()[0];

    // One allocation for the whole matrix, filled in parallel by row: a
    // Vec-per-observation would spend as long allocating as computing.
    let mut flat = vec![0.0f64; n_obs * n_f];
    py.detach(|| {
        flat.par_chunks_mut(n_f).enumerate().for_each(|(i, out)| {
            let ed = row(&e, i);
            let sd = row(&s, i);
            let scale = if use_phase {
                let a = al[i];
                phase[0] * (-a / phase[1]).exp() + phase[2] * a + 1.0
            } else {
                1.0
            };
            for (j, cell) in out.iter_mut().enumerate() {
                let nj = row(&nor, j);
                let mu = dot(nj, ed);
                if mu <= 0.0 {
                    continue;
                }
                let mu0 = dot(nj, sd);
                if mu0 <= 0.0 {
                    continue;
                }
                *cell = scale * mu * mu0 * (1.0 / (mu + mu0) + lambert_weight) * alb[j];
            }
        });
    });

    let result = Array2::from_shape_vec((n_obs, n_f), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(result.into_pyarray(py))
}


/// Eq. (13) renormalisation and its derivative, for a whole lightcurve set.
///
/// For each lightcurve `i`, with `m = A g` over that curve's rows,
///
/// ```text
///   y_k        = m_k / mean(m)
///   dy_k/dg_j  = (A_kj - y_k * mean_l A_lj) / mean(m)
/// ```
///
/// which "renormalises both the observed and the model lightcurves to mean
/// brightnesses of unity", discarding every per-curve scale factor.
///
/// In Python this is a loop over lightcurves doing two means, an outer product
/// and a subtraction per curve.  For a period scan - thousands of tiny
/// Levenberg-Marquardt fits over fifty-odd curves - the per-call NumPy overhead
/// dominates everything else in the package.
///
/// `offsets` holds the `C + 1` row boundaries of the curves.  Returns
/// `(model, jacobian)`; the Jacobian is empty when `want_jac` is false.
#[pyfunction]
fn normalise_relative<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray1<'py, f64>,
    design: PyReadonlyArray2<'py, f64>,
    offsets: PyReadonlyArray1<'py, i64>,
    want_jac: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let m = raw.as_array();
    let a = design.as_array();
    let off: ArrayView1<i64> = offsets.as_array();
    let n = m.len();
    let n_f = a.shape()[1];
    let n_curves = off.len() - 1;

    let mut model = vec![0.0f64; n];
    let mut jac = vec![0.0f64; if want_jac { n * n_f } else { 0 }];

    py.detach(|| {
        // Each curve owns a disjoint block of rows, so they can be done in
        // parallel with no synchronisation.
        let mut model_blocks: Vec<&mut [f64]> = Vec::with_capacity(n_curves);
        let mut rest = model.as_mut_slice();
        for c in 0..n_curves {
            let len = (off[c + 1] - off[c]) as usize;
            let (head, tail) = rest.split_at_mut(len);
            model_blocks.push(head);
            rest = tail;
        }
        let mut jac_blocks: Vec<&mut [f64]> = Vec::with_capacity(n_curves);
        if want_jac {
            let mut rest = jac.as_mut_slice();
            for c in 0..n_curves {
                let len = (off[c + 1] - off[c]) as usize * n_f;
                let (head, tail) = rest.split_at_mut(len);
                jac_blocks.push(head);
                rest = tail;
            }
        }

        let work: Vec<usize> = (0..n_curves).collect();
        if want_jac {
            model_blocks
                .into_par_iter()
                .zip(jac_blocks.into_par_iter())
                .zip(work.into_par_iter())
                .for_each(|((mb, jb), c)| {
                    let lo = off[c] as usize;
                    let len = mb.len();
                    let mut mean = 0.0;
                    for k in 0..len {
                        mean += m[lo + k];
                    }
                    mean /= len as f64;
                    if mean <= 0.0 {
                        mean = 1e-300;
                    }
                    for k in 0..len {
                        mb[k] = m[lo + k] / mean;
                    }
                    // Column means of this curve's block of the design matrix.
                    let mut col_mean = vec![0.0f64; n_f];
                    for k in 0..len {
                        for (j, cm) in col_mean.iter_mut().enumerate() {
                            *cm += a[[lo + k, j]];
                        }
                    }
                    let inv_len = 1.0 / len as f64;
                    for cm in col_mean.iter_mut() {
                        *cm *= inv_len;
                    }
                    let inv_mean = 1.0 / mean;
                    for k in 0..len {
                        let yk = mb[k];
                        let row_out = &mut jb[k * n_f..(k + 1) * n_f];
                        for (j, cell) in row_out.iter_mut().enumerate() {
                            *cell = (a[[lo + k, j]] - yk * col_mean[j]) * inv_mean;
                        }
                    }
                });
        } else {
            model_blocks
                .into_par_iter()
                .zip(work.into_par_iter())
                .for_each(|(mb, c)| {
                    let lo = off[c] as usize;
                    let len = mb.len();
                    let mut mean = 0.0;
                    for k in 0..len {
                        mean += m[lo + k];
                    }
                    mean /= len as f64;
                    if mean <= 0.0 {
                        mean = 1e-300;
                    }
                    for k in 0..len {
                        mb[k] = m[lo + k] / mean;
                    }
                });
        }
    });

    let model_out = Array1::from(model).into_pyarray(py);
    let jac_out = if want_jac {
        Array2::from_shape_vec((n, n_f), jac)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    } else {
        Array2::<f64>::zeros((0, 0))
    };
    Ok((model_out, jac_out.into_pyarray(py)))
}

#[pymodule]
fn lcinv_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_blockers, m)?)?;
    m.add_function(wrap_pyfunction!(trace_fractions, m)?)?;
    m.add_function(wrap_pyfunction!(design_matrix_lsl, m)?)?;
    m.add_function(wrap_pyfunction!(normalise_relative, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
