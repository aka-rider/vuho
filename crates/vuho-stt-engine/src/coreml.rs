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

        /// The declared shape of the input feature `name`, or `None` if the
        /// model has no such input or it is not a multi-array.
        ///
        /// Exists so a backend can read a dimension the export chose (e.g.
        /// the decoder's maximum sequence length, which differs between
        /// builds of the same model) out of the model itself instead of
        /// hardcoding today's value.
        pub(crate) fn input_shape(&self, name: &str) -> Option<Vec<usize>> {
            let ns_name = NSString::from_str(name);
            let description = unsafe { self.model.modelDescription() };
            let inputs = unsafe { description.inputDescriptionsByName() };
            let constraint = unsafe { inputs.objectForKey(&ns_name)?.multiArrayConstraint() }?;
            let shape = unsafe { constraint.shape() };
            // Tensor dimensions are always non-negative.
            #[allow(clippy::cast_sign_loss)]
            let dims = shape.iter().map(|n| n.integerValue() as usize).collect();
            Some(dims)
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

    /// Bytes per element of an `MLMultiArray` data type.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::CoreMl` for a type this crate cannot read.
    fn element_size(dt: MLMultiArrayDataType) -> Result<usize, EngineError> {
        match dt {
            MLMultiArrayDataType::Float32 | MLMultiArrayDataType::Int32 => Ok(4),
            MLMultiArrayDataType::Float16 => Ok(2),
            other => Err(EngineError::CoreMl(format!(
                "unsupported MLMultiArray data type: {other:?}"
            ))),
        }
    }

    /// Read one element of a raw `MLMultiArray` buffer as `f32`.
    ///
    /// SAFETY: `ptr` must point at a buffer of `dt`-typed elements and
    /// `offset` must be within it — [`MlArray::gather_f32_into`], the only
    /// caller, checks that against the size `CoreML` reports.
    unsafe fn read_f32_at(ptr: NonNull<c_void>, dt: MLMultiArrayDataType, offset: usize) -> f32 {
        unsafe {
            match dt {
                MLMultiArrayDataType::Float16 => {
                    (*ptr.as_ptr().cast::<half::f16>().add(offset)).to_f32()
                }
                MLMultiArrayDataType::Int32 => {
                    // Int32 arrays in this model set hold small counters
                    // (lengths, indices), never near f32's 2^24
                    // exact-integer boundary, so the precision loss this
                    // lint warns about cannot occur in practice.
                    #[allow(clippy::cast_precision_loss)]
                    let widened = *ptr.as_ptr().cast::<i32>().add(offset) as f32;
                    widened
                }
                _ => *ptr.as_ptr().cast::<f32>().add(offset),
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
    ///
    /// `Clone` is a retain, not a copy of the buffer: `predict` consumes
    /// the `MlArray`s it is given, so a caller that submits the same input
    /// on every step of a decode loop (Canary's encoder embeddings) shares
    /// one allocation instead of rebuilding a megabyte per step.
    #[derive(Clone)]
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

        /// Element strides per dimension, as `CoreML` actually laid the
        /// buffer out.
        ///
        /// Vendor quirk, and the reason this accessor exists at all:
        /// `CoreML` pads an array's **last** dimension up to a 64-element
        /// boundary, so a declared `[1, 1024, 188]` output is physically
        /// `[1, 1024, 192]`. [`Self::to_f32_vec_into`]'s flat copy assumes
        /// dense packing and would silently return misaligned garbage — not
        /// an error — for such an array. Any reader of a possibly-padded
        /// output must go through [`Self::gather_f32_into`] instead.
        pub(crate) fn strides(&self) -> Vec<usize> {
            let ns_arr = unsafe { self.inner.strides() };
            // Strides are element counts, always positive and small.
            #[allow(clippy::cast_sign_loss)]
            let dims = ns_arr.iter().map(|n| n.integerValue() as usize).collect();
            dims
        }

        /// Gather the elements at `offsets` (element offsets into this
        /// array's physical buffer, as computed from [`Self::strides`])
        /// into `out`, in `offsets` order.
        ///
        /// The one place this crate reads a strided/padded `MLMultiArray`
        /// (CONSTITUTION rule 26). Callers express both a stride-aware
        /// transpose and a single-row read as an offset list, so neither
        /// re-derives buffer arithmetic of its own.
        ///
        /// `out` is cleared and resized, reusing its capacity across calls.
        ///
        /// # Errors
        ///
        /// Returns `EngineError::CoreMl` for an unsupported data type, or
        /// if any offset lies outside the buffer `CoreML` reports — an
        /// out-of-range offset means the layout assumption behind the
        /// offsets is wrong, which must surface rather than read garbage.
        pub(crate) fn gather_f32_into(
            &self,
            offsets: &[usize],
            out: &mut Vec<f32>,
        ) -> Result<(), EngineError> {
            let dt = unsafe { self.inner.dataType() };
            let element_size = element_size(dt)?;
            let Some(&max_offset) = offsets.iter().max() else {
                out.clear();
                return Ok(());
            };

            out.clear();
            out.resize(offsets.len(), 0.0);
            let dst = out.as_mut_ptr();
            let (src_offsets, count) = (offsets.as_ptr(), offsets.len());
            let out_of_range = std::cell::Cell::new(None);

            let block = block2::StackBlock::new(|ptr: NonNull<c_void>, size: NSInteger| {
                // Byte counts from CoreML are always non-negative.
                #[allow(clippy::cast_sign_loss)]
                let capacity = size as usize / element_size;
                if max_offset >= capacity {
                    out_of_range.set(Some(capacity));
                    return;
                }
                unsafe {
                    let offsets = std::slice::from_raw_parts(src_offsets, count);
                    for (i, &off) in offsets.iter().enumerate() {
                        *dst.add(i) = read_f32_at(ptr, dt, off);
                    }
                }
            });
            unsafe { self.inner.getBytesWithHandler(&block) };

            if let Some(capacity) = out_of_range.get() {
                out.clear();
                return Err(EngineError::CoreMl(format!(
                    "strided read offset {max_offset} exceeds the {capacity}-element buffer \
                     CoreML reports for shape {:?} / strides {:?}",
                    self.shape(),
                    self.strides()
                )));
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

    /// Thread-safe wrapper for a loaded backend's models, which aren't
    /// inherently `Send`/`Sync` on their own: this is the crate's only
    /// `unsafe impl Send + Sync`.
    ///
    /// Written out per concrete wrapped type — today `ParakeetModels` and
    /// `CanaryModels` — rather than as a blanket `impl<T>`. A blanket impl
    /// would let ANY future `T` (including one holding non-thread-safe
    /// non-`CoreML` state) become `Send`/`Sync` for free just by being
    /// wrapped here, silently widening this crate's one unsafe-impl
    /// invariant to types it was never audited against. The cost of that
    /// refusal is that a generic wrapper over a backend cannot derive
    /// thread-safety and must carry `where SendModel<M>: Send + Sync`
    /// explicitly (see `streaming_engine`); that is the intended cost.
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
    // SAFETY: identical reasoning for the Canary backend — four
    // `Retained<MLModel>` handles (thread-safe to predict on per Apple's
    // documentation), a plain-data `Vocab`, and one `Mutex<Vec<f32>>`
    // scratch buffer, so concurrent `&self` access blocks rather than
    // races. Written out again rather than folded into a blanket impl, for
    // the reason this module's doc comment gives.
    unsafe impl Send for SendModel<crate::canary::models::CanaryModels> {}
    unsafe impl Sync for SendModel<crate::canary::models::CanaryModels> {}
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

        pub(crate) fn input_shape(&self, _name: &str) -> Option<Vec<usize>> {
            None
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

        pub(crate) fn strides(&self) -> Vec<usize> {
            vec![]
        }

        pub(crate) fn gather_f32_into(
            &self,
            _offsets: &[usize],
            _out: &mut Vec<f32>,
        ) -> Result<(), EngineError> {
            Err(EngineError::CoreMl(
                "CoreML not available on this platform".into(),
            ))
        }
    }

    /// Non-macOS stub — narrowed the same way as the real `imp::SendModel`
    /// (see its doc comment) even though every method here just returns an
    /// error: consistency, not a live safety requirement on this platform.
    pub(crate) struct SendModel<T>(pub T);
    // SAFETY: this platform's backends hold only stub types that never
    // touch any non-Send/Sync FFI state (every method is a stub returning
    // `EngineError::CoreMl`).
    unsafe impl Send for SendModel<crate::parakeet::models::ParakeetModels> {}
    unsafe impl Sync for SendModel<crate::parakeet::models::ParakeetModels> {}
    unsafe impl Send for SendModel<crate::canary::models::CanaryModels> {}
    unsafe impl Sync for SendModel<crate::canary::models::CanaryModels> {}
}

#[cfg(not(target_os = "macos"))]
pub(crate) use not_macos::*;

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::EngineError;

    /// Load one asset of one model, or `None` when this environment has no
    /// model provisioned (every test here skips rather than fails then, so
    /// CI without `models/` stays green).
    fn load_asset(model_id: &str, role: &str, units: ComputeUnits) -> Option<CoreMlModel> {
        let folder = crate::resolve_model_folder(model_id).ok().or_else(|| {
            eprintln!("skipping: no model folder resolved for {model_id}");
            None
        })?;
        let path = crate::asset_path(model_id, role, &folder).ok()?;
        match CoreMlModel::load(&path, units) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("skipping: {model_id}/{role} failed to load: {e}");
                None
            }
        }
    }

    /// Model-gated smoke test: loads the default model's preprocessor and
    /// runs one prediction on a zeroed 15 s window, verifying the wrapper's
    /// load/predict/array round trip end to end.
    #[test]
    fn preprocessor_predicts_on_zeros() {
        let default_model = vuho_model_paths::manifest().stt.default_model.as_str();
        let Some(model) = load_asset(
            default_model,
            crate::asset_role::PREPROCESSOR,
            ComputeUnits::CpuOnly,
        ) else {
            return;
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

    /// An array this crate builds itself is densely packed, so `strides()`
    /// equals the row-major dense strides — the baseline the padded case
    /// below is a departure from. Model-free.
    #[test]
    fn a_dense_array_reports_dense_strides() {
        let arr = MlArray::f32(&[1, 3, 4], &[0.0f32; 12]).expect("build array");
        assert_eq!(arr.strides(), vec![12, 4, 1]);
    }

    /// A gather over an array's dense offsets reproduces its contents, and
    /// an offset past the buffer is a typed error rather than a garbage
    /// read. Model-free.
    #[test]
    fn gather_reads_by_offset_and_rejects_an_out_of_range_one() {
        let data = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let arr = MlArray::f32(&[1, 2, 3], &data).expect("build array");

        let mut out = Vec::new();
        arr.gather_f32_into(&[5, 3, 0], &mut out).expect("gather");
        assert_eq!(out, vec![5.0, 3.0, 0.0]);

        let err = arr
            .gather_f32_into(&[6], &mut out)
            .expect_err("an offset past the buffer must be an error, not garbage");
        assert!(
            matches!(err, EngineError::CoreMl(_)),
            "expected a CoreMl error, got {err:?}"
        );
    }

    /// **WP6.S1 — the stride probe.** `CoreML` pads an array's last
    /// dimension to a 64-element boundary, which makes a dense read of a
    /// padded output return silent garbage. This proves empirically what
    /// the Canary encoder's real layout is, against the actual shipped
    /// model, before any decode logic relies on it.
    ///
    /// Loads with the same compute units the backend ships with, so what it
    /// reports is the layout production actually sees.
    ///
    /// `#[ignore]`d: it loads two large models and is a diagnostic, not a
    /// regression gate. Run with
    /// `cargo test -p vuho-stt-engine -- --ignored canary_stride_probe --nocapture`.
    #[test]
    #[ignore = "diagnostic probe: loads the Canary preprocessor + encoder"]
    fn canary_stride_probe() {
        let Some(model_id) = crate::canary::manifest_model_id() else {
            eprintln!("skipping: the manifest declares no Canary model");
            return;
        };
        let Some(preprocessor) = load_asset(
            model_id,
            crate::asset_role::PREPROCESSOR,
            ComputeUnits::CpuOnly,
        ) else {
            return;
        };
        let Some(encoder) = load_asset(model_id, crate::asset_role::ENCODER, ComputeUnits::CpuOnly)
        else {
            return;
        };

        let window = crate::stream::windower::WINDOW_SAMPLES;
        let audio_signal = MlArray::f32(&[1, window], &vec![0.0f32; window]).expect("audio_signal");
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        let audio_length = MlArray::i32(&[1], &[window as i32]).expect("audio_length");
        let prep = preprocessor
            .predict(&[
                ("audio_signal", audio_signal),
                ("audio_length", audio_length),
            ])
            .expect("preprocessor predict");
        let processed = prep.array("processed").expect("processed output");
        println!(
            "probe: processed shape={:?} strides={:?}",
            processed.shape(),
            processed.strides()
        );

        let enc = encoder
            .predict(&[
                ("features", processed),
                (
                    "features_length",
                    prep.array("processed_length").expect("processed_length"),
                ),
            ])
            .expect("encoder predict");
        let encoder_out = enc.array("encoder").expect("encoder output");
        let (shape, strides) = (encoder_out.shape(), encoder_out.strides());
        println!("probe: encoder shape={shape:?} strides={strides:?}");

        if let Some(decoder) =
            load_asset(model_id, crate::asset_role::DECODER, ComputeUnits::CpuOnly)
        {
            println!(
                "probe: decoder input_ids declared shape={:?}",
                decoder.input_shape("input_ids")
            );
        }

        let dense: Vec<usize> = shape
            .iter()
            .rev()
            .scan(1usize, |acc, &d| {
                let s = *acc;
                *acc *= d;
                Some(s)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        println!(
            "probe: dense strides would be {dense:?} — equal? {}",
            dense == strides
        );
    }
}
