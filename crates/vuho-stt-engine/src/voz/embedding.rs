//! The prediction network's token embedding table, shipped as
//! `embedding.f16`: raw little-endian half floats, one row per token id,
//! with the blank's row last.

use std::path::Path;

use crate::voz::meta::Meta;
use crate::EngineError;

/// Bytes per stored element.
const F16_BYTES: usize = 2;

/// The embedding table: `vocab_size + 1` rows of `pred_hidden` values.
pub(crate) struct Embedding {
    table: Vec<half::f16>,
    width: usize,
}

impl Embedding {
    /// Read `path` and check its size against `meta`.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::LoadFailed` if the file cannot be read or its
    /// length is not exactly `(vocab_size + 1) * pred_hidden` halves.
    pub(crate) fn load(path: &Path, meta: &Meta) -> Result<Self, EngineError> {
        let bytes = std::fs::read(path).map_err(|e| {
            EngineError::LoadFailed(format!("failed to read {}: {e}", path.display()))
        })?;
        Self::from_le_bytes(&bytes, meta)
    }

    fn from_le_bytes(bytes: &[u8], meta: &Meta) -> Result<Self, EngineError> {
        let expected = (meta.vocab_size + 1) * meta.pred_hidden * F16_BYTES;
        if bytes.len() != expected {
            return Err(EngineError::LoadFailed(format!(
                "embedding table is {} bytes, meta.json implies {expected}",
                bytes.len()
            )));
        }
        let (pairs, no_remainder) = bytes.as_chunks::<F16_BYTES>();
        debug_assert!(
            no_remainder.is_empty(),
            "the length check leaves no odd byte"
        );
        let table = pairs
            .iter()
            .copied()
            .map(half::f16::from_le_bytes)
            .collect();
        Ok(Self {
            table,
            width: meta.pred_hidden,
        })
    }

    /// Write the embedding of token `id` into `out` (cleared first).
    ///
    /// # Errors
    ///
    /// Returns `EngineError::Transcribe` if `id` is beyond the table — the
    /// decoder emitted an id the table does not cover.
    pub(crate) fn row_into(&self, id: u32, out: &mut Vec<f32>) -> Result<(), EngineError> {
        let start = id as usize * self.width;
        let row = self
            .table
            .get(start..start + self.width)
            .ok_or_else(|| EngineError::Transcribe(format!("no embedding row for token {id}")))?;
        out.clear();
        out.extend(row.iter().map(|v| v.to_f32()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Little-endian half-float bytes of `values`.
    fn bytes_of(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect()
    }

    #[test]
    fn a_row_is_looked_up_by_token_id_blank_last() {
        let meta = Meta::tiny();
        let table =
            Embedding::from_le_bytes(&bytes_of(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), &meta)
                .expect("a table of exactly (3 + 1) rows of 2");

        let mut row = Vec::new();
        table.row_into(1, &mut row).expect("token 1");
        assert_eq!(row, [2.0, 3.0]);
        table.row_into(meta.blank_idx, &mut row).expect("blank");
        assert_eq!(row, [6.0, 7.0]);
    }

    #[test]
    fn an_id_beyond_the_table_is_an_error() {
        let meta = Meta::tiny();
        let table = Embedding::from_le_bytes(&bytes_of(&[0.0; 8]), &meta).expect("table");
        assert!(matches!(
            table.row_into(4, &mut Vec::new()),
            Err(EngineError::Transcribe(_))
        ));
    }

    /// A short file, and one with a stray trailing byte, are both rejected:
    /// neither may be silently truncated to fit.
    #[test]
    fn a_table_of_the_wrong_length_is_a_load_failure() {
        let meta = Meta::tiny();
        let mut long = bytes_of(&[0.0; 8]);
        long.push(0);
        for bytes in [bytes_of(&[0.0; 6]), long] {
            assert!(matches!(
                Embedding::from_le_bytes(&bytes, &meta),
                Err(EngineError::LoadFailed(_))
            ));
        }
    }
}
