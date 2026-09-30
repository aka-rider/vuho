//! A thin ONNX Runtime session: named tensors in, named tensors out, every
//! output copied into owned memory before the session lock is released.
//!
//! `Session::run` takes `&mut self` and its outputs borrow the session, so
//! each session lives behind its own `Mutex`; that also makes the wrapper
//! `Send + Sync` without `unsafe`.

use std::borrow::Cow;
use std::path::Path;
use std::sync::Mutex;

use ort::session::{Session, SessionInputValue};
use ort::tensor::TensorElementType;
use ort::value::{DynValue, Tensor, ValueType};

use crate::EngineError;

/// One dimension of an expected shape.
#[derive(Clone, Copy)]
pub(crate) enum Dim {
    /// Any size: a batch or time axis the graph declares as dynamic.
    Any,
    /// Exactly this size.
    Is(usize),
}

/// A tensor element type this wrapper can move across the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ElementKind {
    F32,
    I64,
    I32,
}

impl ElementKind {
    fn element_type(self) -> TensorElementType {
        match self {
            Self::F32 => TensorElementType::Float32,
            Self::I64 => TensorElementType::Int64,
            Self::I32 => TensorElementType::Int32,
        }
    }
}

/// One tensor a graph is expected to declare.
pub(crate) struct ExpectedTensor {
    pub(crate) name: &'static str,
    pub(crate) kind: ElementKind,
    pub(crate) shape: &'static [Dim],
}

struct DeclaredTensor {
    name: String,
    element_type: Option<TensorElementType>,
    shape: Vec<i64>,
}

impl DeclaredTensor {
    fn of(name: &str, value_type: &ValueType) -> Self {
        match value_type {
            ValueType::Tensor { ty, shape, .. } => Self {
                name: name.to_owned(),
                element_type: Some(*ty),
                shape: shape.to_vec(),
            },
            ValueType::Sequence(_) | ValueType::Map { .. } | ValueType::Optional(_) => Self {
                name: name.to_owned(),
                element_type: None,
                shape: Vec::new(),
            },
        }
    }

    fn satisfies(&self, expected: &ExpectedTensor) -> bool {
        self.element_type == Some(expected.kind.element_type())
            && shape_matches(&self.shape, expected.shape)
    }
}

fn shape_matches(declared: &[i64], expected: &[Dim]) -> bool {
    declared.len() == expected.len()
        && declared
            .iter()
            .zip(expected)
            .all(|(have, want)| match want {
                Dim::Any => true,
                Dim::Is(size) => usize::try_from(*have) == Ok(*size),
            })
}

fn describe(shape: &[Dim]) -> String {
    let dims: Vec<String> = shape
        .iter()
        .map(|dim| match dim {
            Dim::Any => "any".to_owned(),
            Dim::Is(size) => size.to_string(),
        })
        .collect();
    format!("[{}]", dims.join(", "))
}

/// An input tensor: its shape and its row-major elements.
pub(crate) enum Input {
    F32(Vec<usize>, Vec<f32>),
    I64(Vec<usize>, Vec<i64>),
    I32(Vec<usize>, Vec<i32>),
}

impl Input {
    fn into_value(self) -> Result<SessionInputValue<'static>, EngineError> {
        Ok(match self {
            Self::F32(shape, data) => Tensor::from_array((shape, data))?.into(),
            Self::I64(shape, data) => Tensor::from_array((shape, data))?.into(),
            Self::I32(shape, data) => Tensor::from_array((shape, data))?.into(),
        })
    }
}

enum Elements {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

struct Output {
    name: String,
    shape: Vec<usize>,
    elements: Elements,
}

impl Output {
    fn copy_from(name: &str, value: &DynValue) -> Result<Self, EngineError> {
        let ValueType::Tensor { ty, .. } = value.dtype() else {
            return Err(EngineError::Onnx(format!("output {name} is not a tensor")));
        };
        let (shape, elements) = match ty {
            TensorElementType::Float32 => {
                let (shape, data) = value.try_extract_tensor::<f32>()?;
                (shape.to_vec(), Elements::F32(data.to_vec()))
            }
            TensorElementType::Int64 => {
                let (shape, data) = value.try_extract_tensor::<i64>()?;
                (shape.to_vec(), Elements::I64(data.to_vec()))
            }
            other => {
                return Err(EngineError::Onnx(format!(
                    "output {name} has unsupported element type {other:?}"
                )))
            }
        };
        let shape = shape
            .into_iter()
            .map(|dim| {
                usize::try_from(dim).map_err(|_| {
                    EngineError::Onnx(format!("output {name} has a negative dimension {dim}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            name: name.to_owned(),
            shape,
            elements,
        })
    }
}

/// The owned outputs of one run.
pub(crate) struct Outputs(Vec<Output>);

impl Outputs {
    fn take(&mut self, name: &str) -> Result<Output, EngineError> {
        let index = self
            .0
            .iter()
            .position(|output| output.name == name)
            .ok_or_else(|| EngineError::Onnx(format!("the graph produced no output {name}")))?;
        Ok(self.0.swap_remove(index))
    }

    /// Remove output `name`, which must be `f32`, returning its shape and
    /// row-major elements.
    pub(crate) fn take_f32(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>), EngineError> {
        match self.take(name)? {
            Output {
                shape,
                elements: Elements::F32(data),
                ..
            } => Ok((shape, data)),
            other => Err(wrong_type(name, "f32", &other)),
        }
    }

    /// Remove output `name`, which must be `i64`, returning its elements.
    pub(crate) fn take_i64(&mut self, name: &str) -> Result<Vec<i64>, EngineError> {
        match self.take(name)? {
            Output {
                elements: Elements::I64(data),
                ..
            } => Ok(data),
            other => Err(wrong_type(name, "i64", &other)),
        }
    }
}

fn wrong_type(name: &str, wanted: &str, got: &Output) -> EngineError {
    let have = match got.elements {
        Elements::F32(_) => "f32",
        Elements::I64(_) => "i64",
    };
    EngineError::Onnx(format!("output {name} is {have}, expected {wanted}"))
}

impl From<ort::Error> for EngineError {
    fn from(error: ort::Error) -> Self {
        Self::Onnx(error.to_string())
    }
}

/// A loaded ONNX graph.
pub(crate) struct OnnxSession {
    file: String,
    session: Mutex<Session>,
    inputs: Vec<DeclaredTensor>,
    outputs: Vec<DeclaredTensor>,
}

impl OnnxSession {
    /// Load the graph at `path`, running intra-op work on `threads` threads.
    ///
    /// # Errors
    ///
    /// `EngineError::Onnx` if ONNX Runtime cannot read or compile the file.
    pub(crate) fn load(path: &Path, threads: usize) -> Result<Self, EngineError> {
        let session = Session::builder()
            .and_then(|builder| builder.with_intra_threads(threads))
            .and_then(|builder| builder.commit_from_file(path))
            .map_err(|e| EngineError::Onnx(format!("{}: {e}", path.display())))?;
        let inputs = session
            .inputs
            .iter()
            .map(|input| DeclaredTensor::of(&input.name, &input.input_type))
            .collect();
        let outputs = session
            .outputs
            .iter()
            .map(|output| DeclaredTensor::of(&output.name, &output.output_type))
            .collect();
        Ok(Self {
            file: path.file_name().map_or_else(
                || path.display().to_string(),
                |n| n.to_string_lossy().into(),
            ),
            session: Mutex::new(session),
            inputs,
            outputs,
        })
    }

    /// Check the graph declares exactly `inputs`, and at least `outputs`.
    ///
    /// # Errors
    ///
    /// `EngineError::LoadFailed` naming the first tensor that is missing or
    /// declared with another element type or shape.
    pub(crate) fn check_signature(
        &self,
        inputs: &[ExpectedTensor],
        outputs: &[ExpectedTensor],
    ) -> Result<(), EngineError> {
        if self.inputs.len() != inputs.len() {
            return Err(EngineError::LoadFailed(format!(
                "{} declares {} inputs, expected {}",
                self.file,
                self.inputs.len(),
                inputs.len()
            )));
        }
        Self::check_tensors(&self.file, "input", &self.inputs, inputs)?;
        Self::check_tensors(&self.file, "output", &self.outputs, outputs)
    }

    fn check_tensors(
        file: &str,
        direction: &str,
        declared: &[DeclaredTensor],
        expected: &[ExpectedTensor],
    ) -> Result<(), EngineError> {
        for want in expected {
            let have = declared
                .iter()
                .find(|tensor| tensor.name == want.name)
                .ok_or_else(|| {
                    EngineError::LoadFailed(format!("{file} declares no {direction} {}", want.name))
                })?;
            if !have.satisfies(want) {
                return Err(EngineError::LoadFailed(format!(
                    "{file} {direction} {} is {:?} {:?}, expected {:?} {}",
                    want.name,
                    have.element_type,
                    have.shape,
                    want.kind,
                    describe(want.shape)
                )));
            }
        }
        Ok(())
    }

    /// Run the graph once and copy out the outputs named `wanted`; the
    /// graph's other outputs are dropped unread.
    ///
    /// # Errors
    ///
    /// `EngineError::Onnx` if a tensor cannot be built, the run fails, or a
    /// wanted output is missing or cannot be copied out.
    pub(crate) fn run(
        &self,
        inputs: Vec<(&'static str, Input)>,
        wanted: &[&str],
    ) -> Result<Outputs, EngineError> {
        let values = inputs
            .into_iter()
            .map(|(name, input)| Ok((Cow::Borrowed(name), input.into_value()?)))
            .collect::<Result<Vec<_>, EngineError>>()?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| EngineError::Onnx(format!("{}: session lock was poisoned", self.file)))?;
        let raw = session.run(values)?;
        let mut copied = Outputs(Vec::with_capacity(wanted.len()));
        for name in wanted {
            let value = raw.get(name).ok_or_else(|| {
                EngineError::Onnx(format!(
                    "{}: the graph produced no output {name}",
                    self.file
                ))
            })?;
            copied.0.push(Output::copy_from(name, value)?);
        }
        Ok(copied)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dynamic_expected_dimension_accepts_any_declared_size() {
        assert!(shape_matches(
            &[-1, 128, 1501],
            &[Dim::Any, Dim::Is(128), Dim::Any]
        ));
        assert!(shape_matches(
            &[1, 128, -1],
            &[Dim::Any, Dim::Is(128), Dim::Any]
        ));
    }

    #[test]
    fn a_fixed_expected_dimension_rejects_another_size() {
        assert!(!shape_matches(
            &[-1, 80, -1],
            &[Dim::Any, Dim::Is(128), Dim::Any]
        ));
        assert!(!shape_matches(
            &[-1, -1, -1],
            &[Dim::Any, Dim::Is(128), Dim::Any]
        ));
    }

    #[test]
    fn a_rank_mismatch_is_not_a_match() {
        assert!(!shape_matches(
            &[-1, 128],
            &[Dim::Any, Dim::Is(128), Dim::Any]
        ));
    }
}
