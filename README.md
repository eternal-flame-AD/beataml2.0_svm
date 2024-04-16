# BeatAML2.0_SVM

This repository contains the code and presentation for my personal course project in STAT5353 at the University of Dallas.

I anm not affiliated with the [BeatAML2.0 project](https://biodev.github.io/BeatAML2/).

## Project Description

The project is a machine learning project that uses the BeatAML2.0 dataset to predict the 12-month survival of patients with acute myeloid leukemia (AML) using a support vector machine (SVM) model. The dataset contains gene expression data, and clinical outcome data for 571 patients with AML. The project uses the gene expression data and drug response data to predict the 12-month survival of the patients.

See my [project proposal](./project_proposal_public.out.pdf) for more details.

## Reproducing the Project

1. Clone the repository
2. `git submodule update --init` to download the BeatAML2.0 dataset
3. Run [data_preparation.R](./data_preparation.R) to prepare the data for feeding into the SVM model
4. Go to [stat5353_project](./stat5353_project) and run `cargo run --release -- --help` to see the available options for running the SVM model. Run the following to reproduce the results:
   - `cargo run --release -- svm-grid-opt --kernel gaussian --input ../svm_data/svm_input_boruta.csv --output ../svm_data/svm_grid_opt_boruta_gaussian.csv --roc-output ../svm_data/svm_grid_opt_boruta_gaussian_roc.csv`
   - `cargo run --release -- svm-compute-importance --kernel gaussian --input ../svm_data/svm_input_boruta.csv --gaussian-eps 47 --c-pos 1 --c-neg 1 --output ../svm_data/svm_importance_boruta.csv --roc-output ../svm_data/svm_importance_boruta_roc.csv`
   - `cargo run --release -- svm-k-fold --kernel gaussian --input ../svm_data/svm_input_boruta.csv --gaussian-eps 47 --c-pos 1 --c-neg 1 --output ../svm_data/svm_kfold_boruta.csv --roc-output ../svm_data/svm_kfold_boruta_roc.csv --folds 10`
5. Go to the [presentation](./presentation) directory and run `quarto render stat5353_project.qmd` to reproduce the presentation slides.


## License

- BeatAML2.0 Data is used under the CC-BY-4.0 license. 
  Bottomly, D., Long, N., Schultz, A. R., Kurtz, S. E., Tognon, C. E., Johnson, K., … & Tyner, J. W. (2022). Integrative analysis of drug response and clinical outcome in acute myeloid leukemia. Cancer Cell, 40(8), 850-864.
- [Linfa](https://github.com/rust-ml/linfa), a Rust machine learning framework, is used under the [MIT](https://github.com/rust-ml/linfa) license. 

The code and presentation portion of this repository is licensed under the Apache-2.0 license.