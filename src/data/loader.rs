use ndarray::{Array1, Array2};
use polars::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct Dataset {
    pub features: Array2<f64>,
    pub labels: Array1<f64>,
    pub feature_names: Vec<String>,
}

type LoadResult<T> = Result<T, Box<dyn std::error::Error>>;

fn read_dataframe(file_path: &str, skiprows: usize) -> LoadResult<DataFrame> {
    let has_header = skiprows > 0;
    let df = CsvReadOptions::default()
        .with_has_header(has_header)
        .try_into_reader_with_file_path(Some(file_path.into()))?
        .finish()?;

    if skiprows > 1 {
        let drop_count = skiprows - 1;
        let remaining = df.height().saturating_sub(drop_count);
        Ok(df.slice(drop_count as i64, remaining))
    } else {
        Ok(df)
    }
}

fn column_to_values(column: &Column) -> LoadResult<Vec<f64>> {
    if column.dtype() == &DataType::String {
        let str_ca = column.str()?;
        let values: Vec<Option<&str>> = str_ca.iter().collect();
        let mut classes: Vec<String> = values
            .iter()
            .filter_map(|opt_s| opt_s.filter(|s| !s.is_empty()).map(|s| s.to_string()))
            .collect();
        classes.sort_unstable();
        classes.dedup();
        Ok(values
            .iter()
            .map(|opt_s| match opt_s {
                Some(s) if !s.is_empty() => {
                    classes.iter().position(|c| c.as_str() == *s).unwrap() as f64
                }
                _ => f64::default(),
            })
            .collect())
    } else {
        Ok(column
            .cast(&DataType::Float64)?
            .f64()?
            .iter()
            .map(|opt| opt.unwrap_or_default())
            .collect())
    }
}

pub fn load_dataset(
    file_path: &str,
    skiprows: usize,
    names: Vec<String>,
    label_col: usize,
) -> LoadResult<Dataset> {
    let df = read_dataframe(file_path, skiprows)?;
    let n_rows = df.height();

    let columns: Vec<Vec<f64>> = df
        .get_column_names()
        .into_iter()
        .map(|col_name| column_to_values(df.column(col_name.as_str())?))
        .collect::<LoadResult<_>>()?;

    let labels_vec: Vec<f64> = columns[label_col].clone();
    let feature_cols: Vec<&Vec<f64>> = columns
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != label_col)
        .map(|(_, col)| col)
        .collect();

    let num_features = feature_cols.len();
    let features_flat: Vec<f64> = (0..n_rows)
        .flat_map(|row| feature_cols.iter().map(move |col| col[row]))
        .collect();

    let features_array = Array2::from_shape_vec((n_rows, num_features), features_flat)?;
    let labels_array = Array1::from_vec(labels_vec);

    Ok(Dataset {
        features: features_array,
        labels: labels_array,
        feature_names: names,
    })
}
