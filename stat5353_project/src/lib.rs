use std::{
    collections::HashSet,
    io::{Read, Seek, Write},
    iter,
};

use linfa::Float;
use rand::prelude::*;

pub mod reader;
pub mod svm;

pub fn kfold<T: Clone>(data: &[T], k: usize) -> Vec<(Vec<T>, Vec<T>)> {
    let mut data = data.to_vec();
    data.shuffle(&mut thread_rng());
    let mut result = Vec::new();
    let n = data.len();
    let fold_size = n / k;
    for i in 0..k {
        let start = i * fold_size;
        let end = if i == k - 1 { n } else { (i + 1) * fold_size };
        let mut train = data[..start].to_vec();
        train.extend_from_slice(&data[end..]);
        result.push((train, data[start..end].to_vec()));
    }
    result
}

pub struct RocCurve(pub Vec<(f64, f64)>);

pub fn roc<F: Float>(probs: &[F], truth: &[bool]) -> RocCurve {
    let mut probs_sorted = probs
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (*p, *t))
        .collect::<Vec<_>>();
    probs_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let x_scale_fact = probs_sorted.iter().filter(|(_, t)| !*t).count() as f64;
    let y_scale_fact = probs_sorted.iter().filter(|(_, t)| *t).count() as f64;

    let tpr_tnr = |cutoff: F| {
        let tp = probs_sorted
            .iter()
            .filter(|(p, t)| *p >= cutoff && *t)
            .count() as f64
            / y_scale_fact;
        let tn = probs_sorted
            .iter()
            .filter(|(p, t)| *p < cutoff && !*t)
            .count() as f64
            / x_scale_fact;
        (1. - tp, tn)
    };

    RocCurve(
        iter::once((0.0, 0.0))
            .chain(probs_sorted.iter().map(|(p, _)| tpr_tnr(*p)))
            .chain(iter::once((1.0, 1.0)))
            .collect(),
    )
}

pub fn pivot_data<R: Read, W: Write, F: Fn((usize, &str)) -> bool>(
    mut rdr: csv::Reader<R>,
    wtr: &mut csv::Writer<W>,
    id_column: &str,
    new_id_name: String,
    select_column: F,
) -> csv::Result<()> {
    let header = rdr.headers()?;

    // the position of input columns that are selected as data
    let input_data_columns = header
        .iter()
        .enumerate()
        .filter(|(i, h)| h != &id_column && select_column((*i, h)))
        .map(|(i, h)| (i, h.to_string()))
        .collect::<Vec<_>>();

    let id_column_idx = header.iter().position(|h| h == id_column).unwrap();

    let mut output_data = vec![Vec::new(); input_data_columns.len() + 1];

    output_data[0].push(new_id_name);
    for (i, (_, c)) in input_data_columns.iter().enumerate() {
        output_data[i + 1].push(c.to_string());
    }

    for record in rdr.records() {
        let record = record.unwrap();
        output_data[0].push(record[id_column_idx].to_string());
        for (i, (pos, _)) in input_data_columns.iter().enumerate() {
            output_data[i + 1].push(record[*pos].to_string());
        }
    }

    for d in output_data {
        wtr.write_record(d)?;
    }

    wtr.flush()?;

    Ok(())
}

pub fn drop_na_cols<R: Read + Seek, W: Write>(
    mut rdr: csv::Reader<R>,
    wtr: &mut csv::Writer<W>,
) -> csv::Result<()> {
    let header = rdr
        .headers()?
        .iter()
        .map(|h| h.to_string())
        .collect::<Vec<_>>();

    let data_start = rdr.position().clone();
    let mut na_cols = HashSet::new();
    for (i, record) in rdr.records().enumerate() {
        let record = record?;
        for (j, field) in record.iter().enumerate() {
            if field == "NA" {
                na_cols.insert(j);
            }
        }
        if i == 0 {
            break;
        }
    }
    let mut output_header = vec![];
    for (i, h) in header.iter().enumerate() {
        if !na_cols.contains(&i) {
            output_header.push(h.to_string());
        }
    }
    wtr.write_record(output_header)?;
    rdr.seek(data_start)?;
    for record in rdr.records() {
        let record = record.unwrap();
        let mut output_record = vec![];
        for (i, h) in header.iter().enumerate() {
            if !na_cols.contains(&i) {
                output_record.push(record[i].to_string());
            }
        }
        wtr.write_record(output_record)?;
    }

    wtr.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn test_pivot_data() {
        let input = [
            "in_id,nil,a,b,c",
            "A,nil1,1,2,3",
            "B,nil2,4,5,6",
            "C,nil3,7,8,9",
        ]
        .join("\n");

        let mut output = Vec::new();
        pivot_data(
            csv::ReaderBuilder::new().from_reader(Cursor::new(input.as_bytes())),
            &mut csv::WriterBuilder::new().from_writer(&mut output),
            "in_id",
            "out_id".to_string(),
            |(_, col)| col != "nil",
        )
        .unwrap();
        let output = String::from_utf8_lossy(&output);
        assert_eq!(
            output.trim(),
            ["out_id,A,B,C", "a,1,4,7", "b,2,5,8", "c,3,6,9"].join("\n")
        );
    }
}
