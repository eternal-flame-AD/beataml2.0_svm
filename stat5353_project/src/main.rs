use std::{fs::OpenOptions, path::Path, sync::Mutex};

use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use linfa::Dataset;
use log::info;
use rand::prelude::*;
use rayon::prelude::*;

use ndarray::Array;
use stat5353_project::{
    kfold, pivot_data,
    svm::{svm_hyper_params_grid, CombinedSVM, SvmHyperParams, C_OPTIONS},
};

#[derive(Debug, Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    PivotData(PivotDataArgs),
    DropNACols(DropNAColsArgs),
    SvmGridOpt(SvmGridOptArgs),
    SvmComputeImportance(SvmComputeImportanceArgs),
    SvmKFold(SvmKFoldArgs),
}

#[derive(Debug, Parser)]
struct PivotDataArgs {
    #[clap(short, long, default_value = ",")]
    delim: char,
    #[clap(short, long)]
    input: String,
    #[clap(short, long)]
    output: String,
    #[clap(long)]
    overwrite: bool,
    #[clap(long)]
    id_column: String,
    #[clap(long)]
    new_id: String,
    #[clap(long)]
    negate: bool,
    #[clap(long)]
    data_column: Vec<String>,
}

#[derive(Debug, Parser)]
struct DropNAColsArgs {
    #[clap(short, long, default_value = ",")]
    delim: char,
    #[clap(short, long)]
    input: String,
    #[clap(short, long)]
    output: String,
}

#[derive(Debug, Parser)]
struct SvmGridOptArgs {
    #[clap(short, long, default_value = ",")]
    delim: char,
    #[clap(short, long)]
    kernel: Vec<String>,
    #[clap(short, long)]
    input: String,
    #[clap(short, long, default_value = "svm_grid_opt.csv")]
    output: String,
    #[clap(short, long)]
    roc_output: Option<String>,
}

#[derive(Debug, Parser)]
struct SvmHyperParamArgs {
    #[clap(short, long)]
    c_pos: f64,
    #[clap(short, long)]
    c_neg: f64,
    #[clap(short, long)]
    kernel: String,
    #[clap(short, long)]
    gaussian_eps: Option<f64>,
    #[clap(short, long)]
    polynomial_c: Option<f64>,
    #[clap(short, long)]
    polynomial_degree: Option<f64>,
}

impl Into<SvmHyperParams<f64>> for SvmHyperParamArgs {
    fn into(self) -> SvmHyperParams<f64> {
        match self.kernel.as_str() {
            "linear" => SvmHyperParams {
                c_pos: self.c_pos,
                c_neg: self.c_neg,
                kernel: stat5353_project::svm::Kernel::Linear,
            },
            "gaussian" => SvmHyperParams {
                c_pos: self.c_pos,
                c_neg: self.c_neg,
                kernel: stat5353_project::svm::Kernel::Gaussian(self.gaussian_eps.unwrap()),
            },
            "polynomial" => SvmHyperParams {
                c_pos: self.c_pos,
                c_neg: self.c_neg,
                kernel: stat5353_project::svm::Kernel::Polynomial(
                    self.polynomial_c.unwrap(),
                    self.polynomial_degree.unwrap(),
                ),
            },
            _ => panic!("Invalid kernel: {}", self.kernel),
        }
    }
}

#[derive(Debug, Parser)]
struct SvmComputeImportanceArgs {
    #[clap(short, long, default_value = ",")]
    delim: char,
    #[clap(short, long)]
    input: String,
    #[clap(short, long, default_value = "svm_stats.csv")]
    output: String,
    #[clap(short, long)]
    c_pos: f64,
    #[clap(short, long)]
    c_neg: f64,
    #[clap(short, long)]
    kernel: String,
    #[clap(short, long)]
    gaussian_eps: Option<f64>,
    #[clap(short, long)]
    polynomial_c: Option<f64>,
    #[clap(short, long)]
    polynomial_degree: Option<f64>,
    #[clap(short, long)]
    roc_output: Option<String>,
}

#[derive(Debug, Parser)]
struct SvmKFoldArgs {
    #[clap(short, long, default_value = ",")]
    delim: char,
    #[clap(short, long)]
    input: String,
    #[clap(short, long, default_value = "svm_stats.csv")]
    output: String,
    #[clap(short, long)]
    c_pos: f64,
    #[clap(short, long)]
    c_neg: f64,
    #[clap(short, long)]
    kernel: String,
    #[clap(short, long)]
    gaussian_eps: Option<f64>,
    #[clap(short, long)]
    polynomial_c: Option<f64>,
    #[clap(short, long)]
    polynomial_degree: Option<f64>,
    #[clap(short, long)]
    roc_output: Option<String>,

    #[clap(short, long)]
    folds: usize,
}

fn parse_r_bool(s: &str) -> bool {
    match s {
        "TRUE" => true,
        "FALSE" => false,
        _ => panic!("Invalid boolean value: {}", s),
    }
}

fn svm_grid_opt(args: SvmGridOptArgs) {
    let input = OpenOptions::new().read(true).open(args.input).unwrap();
    let mut input_csv = csv::ReaderBuilder::new()
        .delimiter(args.delim as u8)
        .from_reader(input);
    let headers = input_csv.headers().unwrap();
    let ncols = headers.len();

    let mut records: Vec<f64> = Vec::new();
    let mut targets: Vec<bool> = Vec::new();

    let mut nrows = 0;
    for record in input_csv.records() {
        nrows += 1;
        let record = record.unwrap();
        for (i, field) in record.iter().enumerate() {
            if i == 0 {
                targets.push(parse_r_bool(field));
            } else {
                records.push(field.parse().unwrap());
            }
        }
    }

    let dataset = Dataset::new(
        Array::from(records).into_shape((nrows, ncols - 1)).unwrap(),
        Array::from(targets),
    );

    let csv_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(args.output)
        .unwrap();

    let mut csv_writer = csv::WriterBuilder::new()
        .delimiter(args.delim as u8)
        .from_writer(csv_file);

    csv_writer
        .write_record(&[
            "kernel",
            "c_pos",
            "c_neg",
            "gaussian_eps",
            "polynomial_c",
            "polynomial_degree",
            "accuracy",
            "precision",
            "recall",
            "mcc",
            "f1",
        ])
        .unwrap();

    let csv_writer = Mutex::new(csv_writer);

    let params_count = svm_hyper_params_grid(&args.kernel).count() as u64;
    let params = svm_hyper_params_grid(&args.kernel)
        .par_bridge()
        .into_par_iter();

    let roc_output = args.roc_output.map(|s| {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(s)
            .unwrap();
        let mut f = csv::WriterBuilder::new().from_writer(file);
        f.write_record(&[
            "kernel",
            "c_pos",
            "c_neg",
            "gaussian_eps",
            "polynomial_c",
            "polynomial_degree",
            "x",
            "y",
        ])
        .unwrap();
        Mutex::new(f)
    });

    let bar = ProgressBar::new(params_count);

    bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise} ETA:{eta_precise}] {wide_bar} {pos:>7}/{len:7} {msg}",
        )
        .unwrap(),
    );

    params.for_each(|c| {
        let (_model, roc, cm) = stat5353_project::svm::model_svm(dataset.clone(), 0.8, &c).unwrap();
        let mut csv_writer = csv_writer.lock().unwrap();
        csv_writer
            .write_record(&[
                match c.kernel {
                    stat5353_project::svm::Kernel::Linear => "linear".to_string(),
                    stat5353_project::svm::Kernel::Gaussian(_) => "gaussian".to_string(),
                    stat5353_project::svm::Kernel::Polynomial(_, _) => "polynomial".to_string(),
                },
                c.c_pos.to_string(),
                c.c_neg.to_string(),
                match c.kernel {
                    stat5353_project::svm::Kernel::Gaussian(eps) => eps.to_string(),
                    _ => "".to_string(),
                },
                match c.kernel {
                    stat5353_project::svm::Kernel::Polynomial(c, _) => c.to_string(),
                    _ => "".to_string(),
                },
                match c.kernel {
                    stat5353_project::svm::Kernel::Polynomial(_, degree) => degree.to_string(),
                    _ => "".to_string(),
                },
                cm.accuracy().to_string(),
                cm.precision().to_string(),
                cm.recall().to_string(),
                cm.mcc().to_string(),
                cm.f1_score().to_string(),
            ])
            .unwrap();

        if let Some(roc_output) = &roc_output {
            let mut roc_output = roc_output.lock().unwrap();
            for (x, y) in roc.0 {
                roc_output
                    .write_record(&[
                        match c.kernel {
                            stat5353_project::svm::Kernel::Linear => "linear".to_string(),
                            stat5353_project::svm::Kernel::Gaussian(_) => "gaussian".to_string(),
                            stat5353_project::svm::Kernel::Polynomial(_, _) => {
                                "polynomial".to_string()
                            }
                        },
                        c.c_pos.to_string(),
                        c.c_neg.to_string(),
                        match c.kernel {
                            stat5353_project::svm::Kernel::Gaussian(eps) => eps.to_string(),
                            _ => "".to_string(),
                        },
                        match c.kernel {
                            stat5353_project::svm::Kernel::Polynomial(c, _) => c.to_string(),
                            _ => "".to_string(),
                        },
                        match c.kernel {
                            stat5353_project::svm::Kernel::Polynomial(_, degree) => {
                                degree.to_string()
                            }
                            _ => "".to_string(),
                        },
                        x.to_string(),
                        y.to_string(),
                    ])
                    .unwrap();
            }
        }

        bar.inc(1);
    });

    bar.finish();
}

fn svm_compute_importance(args: SvmComputeImportanceArgs) {
    let input = OpenOptions::new().read(true).open(args.input).unwrap();
    let mut input_csv = csv::ReaderBuilder::new()
        .delimiter(args.delim as u8)
        .from_reader(input);
    let headers = input_csv.headers().unwrap().clone();
    let ncols = headers.len();
    let record_dim = ncols - 1;

    let mut records: Vec<f64> = Vec::new();
    let mut targets: Vec<bool> = Vec::new();

    let params = SvmHyperParams {
        c_pos: args.c_pos,
        c_neg: args.c_neg,
        kernel: match args.kernel.as_str() {
            "linear" => stat5353_project::svm::Kernel::Linear,
            "gaussian" => stat5353_project::svm::Kernel::Gaussian(args.gaussian_eps.unwrap()),
            "polynomial" => stat5353_project::svm::Kernel::Polynomial(
                args.polynomial_c.unwrap(),
                args.polynomial_degree.unwrap(),
            ),
            _ => panic!("Invalid kernel: {}", args.kernel),
        },
    };

    let mut nrows = 0;
    for record in input_csv.records() {
        nrows += 1;
        let record = record.unwrap();
        for (i, field) in record.iter().enumerate() {
            if i == 0 {
                targets.push(parse_r_bool(field));
            } else {
                records.push(field.parse().unwrap());
            }
        }
    }

    let dataset_orig = Dataset::new(
        Array::from(records.clone())
            .into_shape((nrows, record_dim))
            .unwrap(),
        Array::from(targets.clone()),
    );

    let roc_output = args.roc_output.map(|s| {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(s)
            .unwrap();
        let mut f = csv::WriterBuilder::new().from_writer(file);
        f.write_record(&["feature", "x", "y"]).unwrap();
        Mutex::new(f)
    });

    let (_model, roc, cm) = stat5353_project::svm::model_svm(dataset_orig, 0.8, &params).unwrap();

    if let Some(roc_output) = &roc_output {
        let mut roc_output = roc_output.lock().unwrap();
        for (x, y) in roc.0 {
            roc_output
                .write_record(&["NONE", &x.to_string(), &y.to_string()])
                .unwrap();
        }
    }

    let csv_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(args.output)
        .unwrap();

    let mut csv_writer = csv::WriterBuilder::new()
        .delimiter(args.delim as u8)
        .from_writer(csv_file);

    csv_writer
        .write_record(&[
            "feature",
            "delta_accuracy",
            "delta_precision",
            "delta_recall",
            "delta_mcc",
            "delta_f1",
        ])
        .unwrap();

    let csv_writer = Mutex::new(csv_writer);

    let bar = ProgressBar::new(ncols as u64 - 1);

    bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise} ETA:{eta_precise}] {wide_bar} {pos:>7}/{len:7} {msg}",
        )
        .unwrap(),
    );

    (0..record_dim).into_par_iter().for_each_init(
        || thread_rng(),
        |rng, col_to_drop| {
            let mut dataset_perm = |i, rng| {
                let mut res = Array::from(records.clone())
                    .into_shape((nrows, record_dim))
                    .unwrap();

                let mut perm = res.column(i).to_vec();
                perm.shuffle(rng);

                res.column_mut(i).assign(&Array::from(perm));

                Dataset::new(res, Array::from(targets.clone()))
            };

            let dataset = dataset_perm(col_to_drop, rng);
            let (_model_drop, roc_drop, cm_drop) =
                stat5353_project::svm::model_svm(dataset, 0.8, &params).unwrap();
            let mut csv_writer = csv_writer.lock().unwrap();
            csv_writer
                .write_record(&[
                    headers[col_to_drop + 1].to_string(),
                    (cm_drop.accuracy() - cm.accuracy()).to_string(),
                    (cm_drop.precision() - cm.precision()).to_string(),
                    (cm_drop.recall() - cm.recall()).to_string(),
                    (cm_drop.mcc() - cm.mcc()).to_string(),
                    (cm_drop.f1_score() - cm.f1_score()).to_string(),
                ])
                .unwrap();
            if let Some(roc_output) = &roc_output {
                let mut roc_output = roc_output.lock().unwrap();
                for (x, y) in roc_drop.0 {
                    roc_output
                        .write_record(&[
                            headers[col_to_drop].to_string(),
                            x.to_string(),
                            y.to_string(),
                        ])
                        .unwrap();
                }
            }
            bar.inc(1);
        },
    );

    bar.finish();
}

fn svm_kfold(args: SvmKFoldArgs) {
    let mut records: Vec<Vec<f64>> = Vec::new();
    let mut targets = Vec::new();

    let input = OpenOptions::new().read(true).open(args.input).unwrap();
    let mut input_csv = csv::ReaderBuilder::new()
        .delimiter(args.delim as u8)
        .from_reader(input);

    let headers = input_csv.headers().unwrap().clone();
    let ncols = headers.len();
    let record_dim = ncols - 1;

    let mut nrows = 0;

    for record in input_csv.records() {
        nrows += 1;
        let record = record.unwrap();

        let mut row = Vec::new();
        for (i, field) in record.iter().enumerate() {
            if i == 0 {
                targets.push(parse_r_bool(field));
            } else {
                row.push(field.parse().unwrap());
            }
        }
        records.push(row);
    }

    let params = SvmHyperParams {
        c_pos: args.c_pos,
        c_neg: args.c_neg,
        kernel: match args.kernel.as_str() {
            "linear" => stat5353_project::svm::Kernel::Linear,
            "gaussian" => stat5353_project::svm::Kernel::Gaussian(args.gaussian_eps.unwrap()),
            "polynomial" => stat5353_project::svm::Kernel::Polynomial(
                args.polynomial_c.unwrap(),
                args.polynomial_degree.unwrap(),
            ),
            _ => panic!("Invalid kernel: {}", args.kernel),
        },
    };

    let joined_ds = records.iter().zip(targets.iter()).collect::<Vec<_>>();
    let folds = kfold(&joined_ds, args.folds)
        .into_iter()
        .map(|(train, valid)| {
            let train_pred = train.iter().map(|(r, _)| r.clone()).collect::<Vec<_>>();
            let train_target = train.iter().map(|(_, t)| *t).collect::<Vec<_>>();
            let valid_pred = valid.iter().map(|(r, _)| r.clone()).collect::<Vec<_>>();
            let valid_target = valid.iter().map(|(_, t)| *t).collect::<Vec<_>>();
            (train_pred, train_target, valid_pred, valid_target)
        })
        .collect::<Vec<_>>();

    let models = folds
        .iter()
        .map(|(train_pred, train_target, _, _)| {
            let dataset = Dataset::new(
                Array::from_shape_vec(
                    (train_pred.len(), train_pred[0].len()),
                    train_pred
                        .iter()
                        .map(|x| x.iter())
                        .flatten()
                        .cloned()
                        .collect::<Vec<f64>>(),
                )
                .unwrap(),
                Array::from(train_target.iter().map(|x| **x).collect::<Vec<bool>>()),
            );

            let (model, _, _) = stat5353_project::svm::model_svm(dataset, 1.0, &params).unwrap();

            model
        })
        .collect::<Vec<_>>();
    let models = CombinedSVM::new(models);

    let dataset = Dataset::new(
        Array::from_shape_vec(
            (records.len(), records[0].len()),
            records.iter().flatten().cloned().collect::<Vec<f64>>(),
        )
        .unwrap(),
        Array::from(targets),
    );

    let roc = models.roc_curve(&dataset);
    let cm = models.confusion_matrix(&dataset);

    let roc_output = args.roc_output.map(|s| {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(s)
            .unwrap();
        let mut f = csv::WriterBuilder::new().from_writer(file);
        f.write_record(&["x", "y"]).unwrap();
        Mutex::new(f)
    });

    if let Some(roc_output) = &roc_output {
        let mut roc_output = roc_output.lock().unwrap();
        for (x, y) in roc.0 {
            roc_output
                .write_record(&[&x.to_string(), &y.to_string()])
                .unwrap();
        }
    }

    let csv_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(args.output)
        .unwrap();

    let mut csv_writer = csv::WriterBuilder::new()
        .delimiter(args.delim as u8)
        .from_writer(csv_file);

    csv_writer
        .write_record(&["accuracy", "precision", "recall", "mcc", "f1"])
        .unwrap();

    csv_writer
        .write_record(&[
            cm.accuracy().to_string(),
            cm.precision().to_string(),
            cm.recall().to_string(),
            cm.mcc().to_string(),
            cm.f1_score().to_string(),
        ])
        .unwrap();
}

fn main() {
    simple_logger::init().unwrap();
    let cli = Cli::parse();
    info!("cli: {:?}", cli);
    match cli.command {
        Commands::PivotData(c) => {
            let input = OpenOptions::new().read(true).open(c.input).unwrap();
            if Path::new(&c.output).exists() {
                if !c.overwrite {
                    panic!("Not overwriting {}. Emergency stop.", c.output)
                }
            }
            let output = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(c.output)
                .unwrap();
            let input_csv = csv::ReaderBuilder::new()
                .delimiter(c.delim as u8)
                .from_reader(input);
            let mut output_csv = csv::WriterBuilder::new()
                .delimiter(c.delim as u8)
                .from_writer(output);
            pivot_data(
                input_csv,
                &mut output_csv,
                &c.id_column,
                c.new_id.clone(),
                |(_, name)| c.data_column.iter().find(|s| *s == name).is_some() != c.negate,
            )
            .unwrap();
        }
        Commands::DropNACols(c) => {
            let input = OpenOptions::new().read(true).open(c.input).unwrap();
            let output = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(c.output)
                .unwrap();
            let input_csv = csv::ReaderBuilder::new()
                .delimiter(c.delim as u8)
                .from_reader(input);
            let mut output_csv = csv::WriterBuilder::new()
                .delimiter(c.delim as u8)
                .from_writer(output);
            stat5353_project::drop_na_cols(input_csv, &mut output_csv).unwrap();
        }
        Commands::SvmGridOpt(c) => svm_grid_opt(c),
        Commands::SvmComputeImportance(c) => svm_compute_importance(c),
        Commands::SvmKFold(c) => svm_kfold(c),
    }
}
