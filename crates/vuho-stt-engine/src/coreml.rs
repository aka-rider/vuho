//! Thin `objc2-core-ml` wrapper for Parakeet-TDT inference.
//!
//! Loads `.mlmodelc` bundles, builds feature dictionaries, runs predictions,
//! and extracts `f32` results from `MLMultiArray` outputs — including
//! automatic Float16 → Float32 conversion.
//!
//! # Safety
//!
//! `MLModel` predictions are documented by Apple as thread-safe. All
//! Objective-C calls here are `unsafe` but wrapped in a safe API; the
//! [`SendModel`] newtype carries `Send + Sync` because predictions built on
//! it are serialized on one thread by the callers in `parakeet/models.rs`.

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::c_void;
    use std::path::Path;
    use std::ptr::NonNull;

    use objc2::rc::Retained;
    use objc2::runtime::{AnyObject, ProtocolObject};
    use objc2::AnyThread;
    use objc2_core_ml::{
        MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
        MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
    };
    use objc2_foundation::{NSArray, NSInteger, NSMutableDictionary, NSNumber, NSString, NSURL};

    use crate::EngineError;

    /// Compute-unit selection for model loading.
    #[derive(Debug, Clone, Copy)]
    pub(crate) enum ComputeUnits {
        /// CPU only — used for `Preprocessor`, which is all CPU-side ops
        /// (framing, FFT, mel filterbank); `FluidAudio` does the same.
        CpuOnly,
        /// CPU + ANE — used for the encoder/decoder/joint, where dispatch to
        /// the Neural Engine is the point of this whole engine.
        CpuAndNeuralEngine,
    }

    impl ComputeUnits {
        fn into_ml(self) -> MLComputeUnits {
            match self {
                Self::CpuOnly => MLComputeUnits::CPUOnly,
                Self::CpuAndNeuralEngine => MLComputeUnits::CPUAndNeuralEngine,
            }
        }
    }

    /// A loaded `CoreML` model.
    pub(crate) struct CoreMlModel {
        model: Retained<MLModel>,
    }

    impl CoreMlModel {
        /// Load a `.mlmodelc` bundle from `mlmodelc_dir`.
        ///
        /// `.mlmodelc` directories load directly — no compile step — and
        /// this works even when `metadata.json` is absent (as with
        /// `ParakeetDecoder`): the interface is read from `model.mil`.
        ///
        /// # Errors
        ///
        /// Returns `EngineError::LoadFailed` if `CoreML` cannot load the bundle.
        pub(crate) fn load(mlmodelc_dir: &Path, units: ComputeUnits) -> Result<Self, EngineError> {
            let url = Self::file_url(mlmodelc_dir);
            let config = Self::make_config(units);
            let model =
                unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
                    .map_err(|e| {
                        EngineError::LoadFailed(format!(
                            "CoreML load failed ({}): {}",
                            mlmodelc_dir.display(),
                            e.localizedDescription()
                        ))
                    })?;
            Ok(Self { model })
        }

        fn file_url(path: &Path) -> Retained<NSURL> {
            let ns_path = NSString::from_str(&path.to_string_lossy());
            NSURL::fileURLWithPath_isDirectory(&ns_path, true)
        }

        fn make_config(units: ComputeUnits) -> Retained<MLModelConfiguration> {
            let config = unsafe { MLModelConfiguration::new() };
            unsafe { config.setComputeUnits(units.into_ml()) };
            config
        }

        /// Run a prediction with the given feature map.
        ///
        /// # Errors
        ///
        /// Returns `EngineError::CoreMl` if `CoreML` prediction fails.
        pub(crate) fn predict(
            &self,
            inputs: &[(&str, MlArray)],
        ) -> Result<Prediction, EngineError> {
            let dict = Self::build_feature_dict(inputs);
            let provider = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    Self::as_ns_dictionary(&dict),
                )
            }
            .map_err(|e| {
                EngineError::CoreMl(format!(
                    "MLDictionaryFeatureProvider failed: {}",
                    e.localizedDescription()
                ))
            })?;
            let provider_ref: &ProtocolObject<dyn MLFeatureProvider> =
                ProtocolObject::from_ref(&*provider);
            let result =
                unsafe { self.model.predictionFromFeatures_error(provider_ref) }.map_err(|e| {
                    EngineError::CoreMl(format!(
                        "CoreML prediction failed: {}",
                        e.localizedDescription()
                    ))
                })?;
            Ok(Prediction { provider: result })
        }

        /// Build an `NSMutableDictionary<NSString, MLFeatureValue>` from inputs.
        fn build_feature_dict(
            inputs: &[(&str, MlArray)],
        ) -> Retained<NSMutableDictionary<NSString, MLFeatureValue>> {
            let dict: Retained<NSMutableDictionary<NSString, MLFeatureValue>> =
                NSMutableDictionary::dictionary();
            for (key, arr) in inputs {
                let ns_key = NSString::from_str(key);
                let feature_value = arr.to_feature_value();
                unsafe {
                    dict.setObject_forKey(&feature_value, ProtocolObject::from_ref(&*ns_key));
                };
            }
            dict
        }

        /// View an `NSMutableDictionary<NSString, MLFeatureValue>` as the
        /// `NSDictionary<NSString, AnyObject>` the feature-provider
        /// initializer expects.
        ///
        /// SAFETY: `NSMutableDictionary` is an Objective-C subclass of
        /// `NSDictionary`, and `MLFeatureValue` objects are also
        /// `AnyObject`s, so the pointer is valid to reinterpret at either
        /// generic parameter — no layout or ownership change occurs.
        fn as_ns_dictionary(
            dict: &NSMutableDictionary<NSString, MLFeatureValue>,
        ) -> &objc2_foundation::NSDictionary<NSString, AnyObject> {
            unsafe {
                &*std::ptr::from_ref(dict)
                    .cast::<objc2_foundation::NSDictionary<NSString, AnyObject>>()
            }
        }
    }

    /// A prediction result wrapping an `MLFeatureProvider`.
    pub(crate) struct Prediction {
        provider: Retained<ProtocolObject<dyn MLFeatureProvider>>,
    }

    impl Prediction {
        /// Extract an `MLMultiArray` output by feature name.
        ///
        /// # Errors
        ///
        /// Returns `EngineError::CoreMl` if the named feature is absent
        /// or is not an array.
        pub(crate) fn array(&self, name: &str) -> Result<MlArray, EngineError> {
            let ns_name = NSString::from_str(name);
            let fv = unsafe { self.provider.featureValueForName(&ns_name) }
                .ok_or_else(|| EngineError::CoreMl(format!("missing output feature: {name}")))?;
            let inner = unsafe { fv.multiArrayValue() }
                .ok_or_else(|| EngineError::CoreMl(format!("feature '{name}' is not an array")))?;
            Ok(MlArray { inner })
        }
    }

    /// Opaque handle to an `MLMultiArray`.
    pub(crate) struct MlArray {
        inner: Retained<MLMultiArray>,
    }

    impl MlArray {
        /// Create a new `MLMultiArray` filled with `data` (row-major /
        /// "first-major contiguous" layout, `CoreML`'s default for a
        /// shape+dataType-only initializer).
        ///
        /// # Errors
        ///
        /// Returns `EngineError::CoreMl` if `data.len()` does not match
        /// the product of `shape`, or if `CoreML` allocation fails.
        pub(crate) fn f32(shape: &[usize], data: &[f32]) -> Result<Self, EngineError> {
            let inner = Self::alloc(shape, MLMultiArrayDataType::Float32, data.len())?;
            let (src, len) = (data.as_ptr(), data.len());
            let block = block2::StackBlock::new(
                move |ptr: NonNull<c_void>,
                      _size: NSInteger,
                      _strides: NonNull<NSArray<NSNumber>>| unsafe {
                    std::ptr::copy_nonoverlapping(src, ptr.as_ptr().cast::<f32>(), len);
                },
            );
            unsafe { inner.getMutableBytesWithHandler(&block) };
            Ok(Self { inner })
        }

        /// Create a new `MLMultiArray` from i32 data (row-major layout).
        ///
        /// # Errors
        ///
        /// Returns `EngineError::CoreMl` if `data.len()` does not match
        /// the product of `shape`, or if `CoreML` allocation fails.
        pub(crate) fn i32(shape: &[usize], data: &[i32]) -> Result<Self, EngineError> {
            let inner = Self::alloc(shape, MLMultiArrayDataType::Int32, data.len())?;
            let (src, len) = (data.as_ptr(), data.len());
            let block = block2::StackBlock::new(
                move |ptr: NonNull<c_void>,
                      _size: NSInteger,
                      _strides: NonNull<NSArray<NSNumber>>| unsafe {
                    std::ptr::copy_nonoverlapping(src, ptr.as_ptr().cast::<i32>(), len);
                },
            );
            unsafe { inner.getMutableBytesWithHandler(&block) };
            Ok(Self { inner })
        }

        fn alloc(
            shape: &[usize],
            data_type: MLMultiArrayDataType,
            data_len: usize,
        ) -> Result<Retained<MLMultiArray>, EngineError> {
            let expected: usize = shape.iter().product();
            if data_len != expected {
                return Err(EngineError::CoreMl(format!(
                    "MlArray shape {shape:?} expects {expected} elements, got {data_len}"
                )));
            }
            let shape_array = Self::ns_number_array(shape);
            unsafe {
                MLMultiArray::initWithShape_dataType_error(
                    MLMultiArray::alloc(),
                    &shape_array,
                    data_type,
                )
            }
            .map_err(|e| {
                EngineError::CoreMl(format!(
                    "MLMultiArray alloc failed: {}",
                    e.localizedDescription()
                ))
            })
        }

        /// Convert this `MLMultiArray` to an `f32` vector.
        ///
        /// Convenience wrapper around [`Self::to_f32_vec_into`] for callers
        /// that don't have (or don't need) a reusable buffer — see that
        /// method for the one place the actual conversion logic lives
        /// (CONSTITUTION rule 26).
        ///
        /// # Errors
        ///
        /// Returns `EngineError::CoreMl` for an unsupported data type.
        pub(crate) fn to_f32_vec(&self) -> Result<Vec<f32>, EngineError> {
            let mut out = Vec::new();
            self.to_f32_vec_into(&mut out)?;
            Ok(out)
        }

        /// Convert this `MLMultiArray` to an `f32` vector, writing into a
        /// caller-owned `out` buffer instead of allocating a fresh `Vec`
        /// every call (WP9: hot-path allocation discipline — the per-frame
        /// `RNNTJoint` logits extraction is the hottest caller of this
        /// method, once per encoder frame visited).
        ///
        /// `out` is cleared then resized to this array's element count, so
        /// its capacity is reused across calls once it has grown to the
        /// largest size ever requested (`Vec::resize` only reallocates when
        /// growing past current capacity).
        ///
        /// Branches on `dataType()`: `Float32` copies directly, `Float16`
        /// converts each element, `Int32` widens to `f32`. Any other data
        /// type is an error — `CoreML` never returns anything else for the
        /// Parakeet model set.
        ///
        /// # Errors
        ///
        /// Returns `EngineError::CoreMl` for an unsupported data type.
        pub(crate) fn to_f32_vec_into(&self, out: &mut Vec<f32>) -> Result<(), EngineError> {
            let dt = unsafe { self.inner.dataType() };
            let total: usize = self.shape().iter().product();
            out.clear();
            out.resize(total, 0.0);
            let dst = out.as_mut_ptr();
            match dt {
                MLMultiArrayDataType::Float32 => {
                    let block = block2::StackBlock::new(
                        move |ptr: NonNull<c_void>, _size: NSInteger| unsafe {
                            std::ptr::copy_nonoverlapping(ptr.as_ptr().cast::<f32>(), dst, total);
                        },
                    );
                    unsafe { self.inner.getBytesWithHandler(&block) };
                }
                MLMultiArrayDataType::Float16 => {
                    let block = block2::StackBlock::new(
                        move |ptr: NonNull<c_void>, _size: NSInteger| unsafe {
                            let src = ptr.as_ptr().cast::<half::f16>();
                            for i in 0..total {
                                *dst.add(i) = (*src.add(i)).to_f32();
                            }
                        },
                    );
                    unsafe { self.inner.getBytesWithHandler(&block) };
                }
                MLMultiArrayDataType::Int32 => {
                    let block = block2::StackBlock::new(
                        move |ptr: NonNull<c_void>, _size: NSInteger| unsafe {
                            let src = ptr.as_ptr().cast::<i32>();
                            for i in 0..total {
                                // Int32 outputs in this model set are always small counters
                                // (lengths, indices), never near f32's 2^24 exact-integer
                                // boundary, so the precision loss this lint warns about
                                // cannot occur in practice.
                                #[allow(clippy::cast_precision_loss)]
                                let widened = *src.add(i) as f32;
                                *dst.add(i) = widened;
                            }
                        },
                    );
                    unsafe { self.inner.getBytesWithHandler(&block) };
                }
                other => {
                    return Err(EngineError::CoreMl(format!(
                        "unsupported MLMultiArray data type: {other:?}"
                    )));
                }
            }
            Ok(())
        }

        /// Return the dimension sizes as a `Vec<usize>`.
        pub(crate) fn shape(&self) -> Vec<usize> {
            let ns_arr = unsafe { self.inner.shape() };
            // Tensor dimensions are always non-negative; `MLMultiArray` shapes
            // are small (≤ a few thousand), well within usize/NSInteger range.
            #[allow(clippy::cast_sign_loss)]
            let dims = ns_arr.iter().map(|n| n.integerValue() as usize).collect();
            dims
        }

        /// Build an `NSArray<NSNumber>` from a shape slice.
        fn ns_number_array(dims: &[usize]) -> Retained<NSArray<NSNumber>> {
            let objs: Vec<Retained<NSNumber>> = dims
                .iter()
                .map(|&d| {
                    // Tensor dimensions never approach i64::MAX.
                    #[allow(clippy::cast_possible_wrap)]
                    let d = d as i64;
                    NSNumber::new_i64(d)
                })
                .collect();
            let refs: Vec<&NSNumber> = objs.iter().map(AsRef::as_ref).collect();
            NSArray::from_slice(&refs)
        }

        /// Convert `&MlArray` → `MLFeatureValue` for feature provider construction.
        fn to_feature_value(&self) -> Retained<MLFeatureValue> {
            unsafe { MLFeatureValue::featureValueWithMultiArray(&self.inner) }
        }
    }

    /// Thread-safe wrapper for the parts of `ParakeetModels` that aren't
    /// inherently `Send`/`Sync` on their own: this is the crate's single
    /// `unsafe impl Send + Sync`.
    ///
    /// Narrowed to the one concrete type this crate ever wraps
    /// (`SendModel<ParakeetModels>`, from `engine.rs`) rather than a
    /// blanket `impl<T>` — a blanket impl would let ANY future `T`
    /// (including one holding non-thread-safe non-`CoreML` state) become
    /// `Send`/`Sync` for free just by being wrapped here, silently
    /// widening this crate's one unsafe-impl invariant to types it was
    /// never audited against.
    pub(crate) struct SendModel<T>(pub T);
    // SAFETY: Apple documents `MLModel` prediction as thread-safe, so the
    // four `Retained<MLModel>` handles (`objc2`'s `Retained` is `!Send` /
    // `!Sync` by default regardless of the underlying object's actual
    // thread-safety) are sound to share and call concurrently. `Vocab` is
    // plain owned data, trivially `Sync`. `ParakeetModels::padded`, the one
    // piece of interior-mutable state, is a `Mutex<Vec<f32>>` (not a
    // `RefCell`, see `parakeet/models.rs`'s field doc comment) — so this
    // impl does NOT rely on every caller happening to serialize CoreML
    // calls on one thread. That happens to be true of every caller in this
    // crate today, but this type's soundness must not — and, with the
    // `Mutex` in place, no longer does — depend on that convention holding
    // forever: a `Sync` type has to be safe under *any* concurrent access
    // pattern the type system permits, not just the ones current callers
    // happen to use.
    unsafe impl Send for SendModel<crate::parakeet::models::ParakeetModels> {}
    unsafe impl Sync for SendModel<crate::parakeet::models::ParakeetModels> {}
}

#[cfg(target_os = "macos")]
pub(crate) use imp::*;

#[cfg(not(target_os = "macos"))]
mod not_macos {
    use std::path::Path;

    use crate::EngineError;

    #[derive(Debug, Clone, Copy)]
    pub(crate) enum ComputeUnits {
        CpuOnly,
        CpuAndNeuralEngine,
    }

    pub(crate) struct CoreMlModel;

    impl CoreMlModel {
        pub(crate) fn load(
            _mlmodelc_dir: &Path,
            _units: ComputeUnits,
        ) -> Result<Self, EngineError> {
            Err(EngineError::LoadFailed(
                "CoreML not available on this platform".into(),
            ))
        }

        pub(crate) fn predict(
            &self,
            _inputs: &[(&str, MlArray)],
        ) -> Result<Prediction, EngineError> {
            Err(EngineError::CoreMl(
                "CoreML not available on this platform".into(),
            ))
        }
    }

    pub(crate) struct Prediction;

    impl Prediction {
        pub(crate) fn array(&self, _name: &str) -> Result<MlArray, EngineError> {
            Err(EngineError::CoreMl(
                "CoreML not available on this platform".into(),
            ))
        }
    }

    pub(crate) struct MlArray;

    impl MlArray {
        pub(crate) fn f32(_shape: &[usize], _data: &[f32]) -> Result<Self, EngineError> {
            Err(EngineError::CoreMl(
                "CoreML not available on this platform".into(),
            ))
        }

        pub(crate) fn i32(_shape: &[usize], _data: &[i32]) -> Result<Self, EngineError> {
            Err(EngineError::CoreMl(
                "CoreML not available on this platform".into(),
            ))
        }

        pub(crate) fn to_f32_vec(&self) -> Result<Vec<f32>, EngineError> {
            Err(EngineError::CoreMl(
                "CoreML not available on this platform".into(),
            ))
        }

        pub(crate) fn to_f32_vec_into(&self, _out: &mut Vec<f32>) -> Result<(), EngineError> {
            Err(EngineError::CoreMl(
                "CoreML not available on this platform".into(),
            ))
        }

        pub(crate) fn shape(&self) -> Vec<usize> {
            vec![]
        }
    }

    /// Non-macOS stub — narrowed the same way as the real `imp::SendModel`
    /// (see its doc comment) even though every method here just returns an
    /// error: consistency, not a live safety requirement on this platform.
    pub(crate) struct SendModel<T>(pub T);
    // SAFETY: this platform's `ParakeetModels` holds only stub types that
    // never touch any non-Send/Sync FFI state (every method is a stub
    // returning `EngineError::CoreMl`).
    unsafe impl Send for SendModel<crate::parakeet::models::ParakeetModels> {}
    unsafe impl Sync for SendModel<crate::parakeet::models::ParakeetModels> {}
}

#[cfg(not(target_os = "macos"))]
pub(crate) use not_macos::*;

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    /// Model-gated smoke test: loads `Preprocessor.mlmodelc` and runs one
    /// prediction on a zeroed 15 s window, verifying the wrapper's
    /// load/predict/array round trip end to end. Skips (not fails) when no
    /// model is provisioned, so CI without `models/` stays green.
    #[test]
    fn preprocessor_predicts_on_zeros() {
        let Ok(folder) = crate::resolve_model_folder() else {
            eprintln!("skipping: no model folder resolved in this environment");
            return;
        };
        let model =
            match CoreMlModel::load(&folder.join("Preprocessor.mlmodelc"), ComputeUnits::CpuOnly) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("skipping: Preprocessor failed to load: {e}");
                    return;
                }
            };
        let audio_signal =
            MlArray::f32(&[1, 240_000], &vec![0.0f32; 240_000]).expect("build audio_signal");
        let audio_length = MlArray::i32(&[1], &[240_000]).expect("build audio_length");
        let prediction = model
            .predict(&[
                ("audio_signal", audio_signal),
                ("audio_length", audio_length),
            ])
            .expect("predict");
        let mel = prediction.array("mel").expect("mel output");
        assert_eq!(mel.shape(), vec![1, 128, 1501]);
        let mel_data = mel.to_f32_vec().expect("mel to_f32_vec");
        assert_eq!(mel_data.len(), 128 * 1501);
    }
}
