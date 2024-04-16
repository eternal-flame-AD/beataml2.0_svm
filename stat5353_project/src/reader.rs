use csv::ReaderBuilder;
use std::{io::Read, path::Path};

pub fn read_tsv<R: Read>(r: R) -> csv::Reader<R> {
    let rdr = ReaderBuilder::new().delimiter(b'\t').from_reader(r);
    return rdr;
}

pub fn open_tsv(p: &Path) -> Result<csv::Reader<std::fs::File>, std::io::Error> {
    let f = std::fs::File::open(p)?;
    return Ok(read_tsv(f));
}
