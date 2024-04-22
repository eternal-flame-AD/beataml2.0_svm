use lazy_static::lazy_static;
use linfa::prelude::*;
use linfa_svm::{error::Result, Svm};
use ndarray::{ArrayBase, Dim, Ix1, OwnedRepr};

use crate::{roc, RocCurve};

pub struct CombinedSVM<F: Float> {
    svms: Vec<Svm<F, Pr>>,
}

impl Predict<ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>, f64> for CombinedSVM<f64> {
    fn predict(&self, x: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>) -> f64 {
        let mut sum = 0.0f64;
        for svm in &self.svms {
            sum += *svm.predict(x.clone()) as f64;
        }
        sum / self.svms.len() as f64
    }
}

impl<F: Float> CombinedSVM<F> {
    pub fn new(svms: Vec<Svm<F, Pr>>) -> Self {
        CombinedSVM { svms }
    }
}

impl CombinedSVM<f64> {
    pub fn roc_curve(&self, dataset: &Dataset<f64, bool, Ix1>) -> RocCurve {
        let mut probs = vec![0.0; dataset.nsamples()];
        for svm in &self.svms {
            let pred = svm.predict(dataset.records());
            for (i, p) in pred.iter().enumerate() {
                probs[i] += **p as f64;
            }
        }
        roc(&probs, dataset.targets().as_slice().unwrap())
    }
    pub fn confusion_matrix(&self, dataset: &Dataset<f64, bool, Ix1>) -> ConfusionMatrix<bool> {
        let mut preds = vec![0.0; dataset.nsamples()];
        for svm in &self.svms {
            let pred = svm.predict(dataset.records());
            for (i, p) in pred.iter().enumerate() {
                preds[i] += **p as f64;
            }
        }

        let preds = preds
            .iter()
            .map(|x| *x / self.svms.len() as f64)
            .map(|x| x > 0.5)
            .collect::<Vec<bool>>();

        ArrayBase::<OwnedRepr<bool>, Dim<[usize; 1]>>::from(preds)
            .confusion_matrix(dataset)
            .unwrap()
    }
}

pub fn model_svm<F: Float>(
    dataset: Dataset<F, bool, Ix1>,
    tv_ratio: f32,
    params: &SvmHyperParams<F>,
) -> Result<(Svm<F, Pr>, RocCurve, ConfusionMatrix<bool>)> {
    let (train, valid) = dataset.split_with_ratio(tv_ratio);

    let model = match params.kernel {
        Kernel::Linear => Svm::<F, Pr>::params()
            .pos_neg_weights(params.c_pos, params.c_neg)
            .linear_kernel()
            .fit(&train)?,
        Kernel::Gaussian(gamma) => Svm::<F, Pr>::params()
            .pos_neg_weights(params.c_pos, params.c_neg)
            .gaussian_kernel(gamma)
            .fit(&train)?,
        Kernel::Polynomial(gamma, degree) => Svm::<F, Pr>::params()
            .pos_neg_weights(params.c_pos, params.c_neg)
            .polynomial_kernel(gamma, degree)
            .fit(&train)?,
    };

    let valid_pred = model.predict(&valid);
    let valid_pred_probs = valid_pred.iter().map(|x| **x).collect::<Vec<f32>>();
    let valid_pred_bin = valid_pred.map(|x| x > &Pr::new(0.5));

    let cm = valid_pred_bin.confusion_matrix(&valid)?;

    Ok((
        model,
        roc(
            valid_pred_probs.as_slice(),
            valid.targets.as_slice().unwrap(),
        ),
        cm,
    ))
}

pub struct SvmHyperParams<F: Float> {
    pub c_pos: F,
    pub c_neg: F,
    pub kernel: Kernel<F>,
}

pub enum Kernel<F: Float> {
    Linear,
    Gaussian(F),
    Polynomial(F, F),
}
/*pub const C_OPTIONS: [f64; 23] = [
    0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    10.0, 50.0, 100.0, 500.0,
];*/

lazy_static! {
    static ref C_OPTIONS: Vec<f64> = (-5..=15).map(|x| 2f64.powi(x)).collect::<Vec<f64>>();
}

pub fn svm_hyper_params_grid<'a, 'b>(
    kernels: &'b [String],
) -> impl Iterator<Item = SvmHyperParams<f64>> + 'a
where
    'b: 'a,
{
    C_OPTIONS.iter().flat_map(move |c_pos| {
        C_OPTIONS.iter().flat_map(move |c_neg| {
            let linear_kernel = Kernel::Linear;

            let gaussian_kernels = (10..100).map(move |gamma| Kernel::Gaussian(gamma as f64));

            let polynomial_kernels =
                (-8..=8)
                    .map(|x| 2f64.powi(x))
                    .into_iter()
                    .flat_map(move |gamma| {
                        vec![1., 2., 3., 4., 5.]
                            .into_iter()
                            .map(move |degree| Kernel::Polynomial(gamma, degree))
                    });

            std::iter::once(linear_kernel)
                .chain(gaussian_kernels)
                .chain(polynomial_kernels)
                .filter(move |kernel| {
                    kernels.is_empty()
                        || match kernel {
                            Kernel::Linear => kernels.contains(&"linear".to_string()),
                            Kernel::Gaussian(_) => {
                                c_neg == c_pos && kernels.contains(&"gaussian".to_string())
                            }
                            Kernel::Polynomial(_, _) => {
                                c_neg == c_pos && kernels.contains(&"polynomial".to_string())
                            }
                        }
                })
                .map(move |kernel| SvmHyperParams {
                    c_pos: *c_pos,
                    c_neg: *c_neg,
                    kernel,
                })
        })
    })
}
