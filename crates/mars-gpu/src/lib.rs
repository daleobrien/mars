//! GPU exhaustive fractal search — Step 7.
//!
//! `implementation-plan.md`'s Step 7 brief: on Apple Silicon's unified memory there is no
//! host<->device transfer to amortise, so the GPU is useful *early*, and the workload that
//! most needs it is not the product encoder but the **oracle** (Step 8) — a full
//! `N_ranges x M_domains x 8 isometries` sweep that is embarrassingly parallel and
//! otherwise too expensive to run across a corpus. This crate parallelises exactly that
//! sweep: for one image and one fixed range-block size, it finds the best
//! `(domain, isometry, qalfa, qbeta, rms)` for every range block at once.
//!
//! **wgpu/WGSL, not raw Metal.** The brief allows dropping to direct Metal "only if
//! profiling demands it, and record the decision either way." Profiling *did* turn up a
//! real throughput problem here (docs/decisions.md D24: measured speedup is ~7-14x, not
//! the brief's >= 50x floor), but the instrumentation in `GpuSearcher::search`
//! (`MARS_GPU_TIMING=1`) attributes essentially all of the GPU-side wall time to the
//! kernel's own execution, not to wgpu's buffer/bind-group/submit overhead (all under
//! ~1ms every measurement) — so the bottleneck is this crate's kernel design (D24's
//! working hypothesis: uncoalesced per-thread memory access into the `contracted`
//! buffer), not wgpu-vs-Metal call overhead. Dropping to direct Metal would not by itself
//! fix a kernel that under-uses the GPU's memory system; the decision recorded here is
//! that no Metal fallback was written because nothing so far points at wgpu as the cause.
//!
//! **Bit-identical to the CPU: mostly, not entirely — see D23.** This is the hard part of
//! the step, and it rests on two things the search.wgsl module doc explains in full: (1)
//! every moment is a sum of non-negative integer terms and fits in `u32` at this
//! project's `max_size` (docs/predictions.md P7.1, confirmed: largest observed magnitude
//! 944,326,860 against `u32::MAX` 4,294,967,295), so the GPU's accumulation is exactly as
//! exact as the CPU's `i64` accumulation, only narrower; (2) the parallel top-1 reduction
//! uses a canonical-enumeration-order tie-break that is provably independent of
//! reduction order (see `better()` in the shader), so it reproduces the CPU's
//! strict-less-than, first-enumerated-wins rule regardless of how work is scheduled
//! across threads. Neither of those was the problem. What is **not** bit-identical: the
//! fit arithmetic itself runs in `f32` on the GPU (Metal has no fp64 — §2.2) where the
//! CPU's `search()` uses `fit_f64`, and `docs/decisions.md` D23 found 3 of 101,099
//! compared blocks (0.003%) where `f32`'s cancellation error in the fit's residual
//! expression flips which of two near-tied candidates a search keeps. `marsbench
//! gpu-search-check` (`just gate-7`) reports this honestly as a failing check — see D23
//! for the full mechanism and diagnostic detail, and the plan's own kill-criterion
//! language ("GPU search cannot be made bit-identical to CPU -> stop the project") for
//! why this is not something to route around with a tolerance.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use mars_core::Plane;

/// The subset of `mars_codec::encode::EncodeParams` this crate's search needs. Kept as an
/// independent, smaller type (rather than depending on `mars-codec`) so this crate's
/// dependency graph stays what the plan's crate-boundary section expects: `mars-gpu`
/// knows about images and GPUs, not about the codec's partition/bitstream types.
#[derive(Debug, Clone, Copy)]
pub struct GpuSearchParams {
    pub size: u32,
    pub shift: u32,
    pub bits_alfa: u32,
    pub bits_beta: u32,
    pub max_alfa: f32,
}

/// One range block's winning candidate, in the same shape as
/// `mars_codec::encode::Candidate` (minus the raw moments, which the CPU keeps only for
/// its own f32-vs-f64 divergence measurement and which this crate has no analogous use
/// for).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuCandidate {
    pub dom_row: u32,
    pub dom_col: u32,
    pub isometry: u8,
    pub qalfa: u32,
    pub qbeta: u32,
    pub rms: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no suitable GPU adapter found")]
    NoAdapter,
    #[error("failed to acquire a wgpu device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("GPU buffer readback failed: {0}")]
    Map(String),
}

/// Owns the GPU device/queue/pipeline. Construction is the expensive, one-time part
/// (adapter negotiation, shader compilation); `search` is cheap to call repeatedly.
pub struct GpuSearcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    /// Step 8's top-32 entry point (`main_top32` in `search.wgsl`), a separate pipeline
    /// from the top-1 `pipeline` above rather than `search()` delegating to
    /// `search_top32()` and taking entry 0 -- the top-32 kernel's workgroup size (32, vs.
    /// `pipeline`'s 256) and per-thread bookkeeping is real extra work (see
    /// `docs/predictions.md` P8.2), and `search()` backs `gate-7`'s own speed floor
    /// (D25's >= 8x). Keeping the two kernels independent means Step 8's additions
    /// cannot regress a gate that already passed. See `search.wgsl`'s module doc for the
    /// kernel design.
    pipeline_top32: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Mirrors `search.wgsl`'s `Params` struct field-for-field. `#[repr(C)]` plus
/// `bytemuck::Pod` is this crate's one legitimate use of a bit-level contract with GPU
/// code; there is no `unsafe` here or anywhere else in this crate; wgpu's safe Rust API
/// does not need it for the buffer/pipeline operations this kernel uses.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsGpu {
    width: u32,
    height: u32,
    size: u32,
    shift: u32,
    contracted_stride: u32,
    num_blocks_x: u32,
    num_dom_rows: u32,
    num_dom_cols: u32,
    bits_alfa: u32,
    bits_beta: u32,
    max_alfa: f32,
    _pad: u32,
}

/// Mirrors `search.wgsl`'s `Candidate` struct field-for-field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CandidateGpu {
    dom_row: u32,
    dom_col: u32,
    isometry: u32,
    qalfa: u32,
    qbeta: u32,
    rms_bits: u32,
    valid: u32,
    _pad: u32,
}

impl GpuSearcher {
    /// Blocks on adapter/device negotiation and shader compilation — do this once and
    /// reuse the searcher across images/sizes.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("mars-gpu device"),
                ..Default::default()
            })
            .await?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mars-gpu search kernel"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("search.wgsl"))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mars-gpu search bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mars-gpu search pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mars-gpu search pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_top32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mars-gpu search-top32 pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main_top32"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            pipeline_top32,
            bind_group_layout,
        })
    }

    /// The 2:1 box-sum contraction of the whole image — deliberately duplicated from
    /// `mars_codec::encode::Contracted::build` rather than shared, since this crate does
    /// not otherwise depend on `mars-codec` (see the module doc). `D(row, col)` is the
    /// *sum*, not the mean, of the four pixels at
    /// `(2row,2col),(2row+1,2col),(2row,2col+1),(2row+1,2col+1)`.
    fn contract(image: &Plane) -> (Vec<u32>, usize) {
        let (w, h) = (image.width(), image.height());
        let (stride, rows) = (w / 2, h / 2);
        let px = image.as_slice();
        let mut data = vec![0u32; stride * rows];
        for i in 0..rows {
            for j in 0..stride {
                let (r0, c0) = (2 * i, 2 * j);
                let sum = u32::from(px[r0 * w + c0])
                    + u32::from(px[r0 * w + c0 + 1])
                    + u32::from(px[(r0 + 1) * w + c0])
                    + u32::from(px[(r0 + 1) * w + c0 + 1]);
                data[i * stride + j] = sum;
            }
        }
        (data, stride)
    }

    /// Run the exhaustive search for every full `size x size` range block on a fixed
    /// grid — `(row, col)` at every multiple of `size` that fits inside the image, in
    /// row-major order (row outer, col inner). This is the granularity the brief asks
    /// for: fixed-size block search, not the quadtree partition, which stays CPU-side.
    ///
    /// Returns one entry per grid block, in that same row-major order, so a caller can
    /// zip it directly against `mars_codec::encode::search_block` called at the same
    /// positions. `None` means no domain position is legal at this size (the image is
    /// smaller than `2*size` in some dimension) — matching the CPU's `checked_sub`
    /// underflow case exactly.
    ///
    /// # Panics
    /// If `size` is 0 or exceeds 32 (this project's `max_size` in every config; the
    /// shader's workgroup-shared `range_block` array is sized for exactly that bound).
    pub fn search(&self, image: &Plane, params: &GpuSearchParams) -> Vec<Option<GpuCandidate>> {
        let raw = self.dispatch(image, params, &self.pipeline, 1);
        raw.into_iter()
            .map(|slot| {
                slot.into_iter().next().and_then(|c| {
                    if c.valid == 0 {
                        None
                    } else {
                        Some(GpuCandidate {
                            dom_row: c.dom_row,
                            dom_col: c.dom_col,
                            isometry: c.isometry as u8,
                            qalfa: c.qalfa,
                            qbeta: c.qbeta,
                            rms: f32::from_bits(c.rms_bits),
                        })
                    }
                })
            })
            .collect()
    }

    /// Step 8's oracle primitive: the same exhaustive sweep as [`Self::search`], but
    /// every range block keeps its top-32 candidates by `rms` (ascending), not just the
    /// winner — computed in one GPU pass via `search.wgsl`'s `main_top32` entry point
    /// (see that module's doc for the in-kernel per-thread-bounded-list +
    /// merge-reduce design). `None` means the block has no legal domain position at all
    /// (same underflow case as `search`'s `None`); a `Some` block's `Vec` is ascending by
    /// `rms` and has length `min(32, total candidate count)` — shorter than 32 only when
    /// the domain-position pool itself has fewer than 32 `(domain, isometry)` pairs,
    /// which does not happen for this project's configs (`min_size >= 4`, `shift <= 8` on
    /// images >= 64x64 all give domain-position counts in the thousands).
    ///
    /// # Panics
    /// Same as [`Self::search`]: `size` must be in `1..=32`.
    pub fn search_top32(
        &self,
        image: &Plane,
        params: &GpuSearchParams,
    ) -> Vec<Option<Vec<GpuCandidate>>> {
        let raw = self.dispatch(image, params, &self.pipeline_top32, 32);
        raw.into_iter()
            .map(|slots| {
                let valid: Vec<GpuCandidate> = slots
                    .into_iter()
                    .filter(|c| c.valid != 0)
                    .map(|c| GpuCandidate {
                        dom_row: c.dom_row,
                        dom_col: c.dom_col,
                        isometry: c.isometry as u8,
                        qalfa: c.qalfa,
                        qbeta: c.qbeta,
                        rms: f32::from_bits(c.rms_bits),
                    })
                    .collect();
                if valid.is_empty() {
                    None
                } else {
                    Some(valid)
                }
            })
            .collect()
    }

    /// Shared buffer setup/dispatch/readback for [`Self::search`] and
    /// [`Self::search_top32`] — identical apart from which pipeline runs and how many
    /// `CandidateGpu` slots each range block occupies in the results buffer
    /// (`slots_per_block`: 1 for the top-1 kernel, 32 for the top-32 kernel — both
    /// kernels share the same output buffer *type*, just at different strides, per
    /// `search.wgsl`'s module doc). Returns one `Vec<CandidateGpu>` of length
    /// `slots_per_block` per grid block, in row-major grid order.
    fn dispatch(
        &self,
        image: &Plane,
        params: &GpuSearchParams,
        pipeline: &wgpu::ComputePipeline,
        slots_per_block: u32,
    ) -> Vec<Vec<CandidateGpu>> {
        assert!(
            params.size > 0 && params.size <= 32,
            "size must be in 1..=32"
        );
        let (width, height) = (image.width() as u32, image.height() as u32);
        let num_blocks_x = width / params.size;
        let num_blocks_y = height / params.size;
        if num_blocks_x == 0 || num_blocks_y == 0 {
            return Vec::new();
        }

        let debug_timing = std::env::var_os("MARS_GPU_TIMING").is_some();
        let t_start = std::time::Instant::now();
        let (contracted, contracted_stride) = Self::contract(image);
        let image_px: Vec<u32> = image.as_slice().iter().map(|&p| u32::from(p)).collect();
        if debug_timing {
            eprintln!(
                "  [gpu timing] contract+upload-prep: {:?}",
                t_start.elapsed()
            );
        }

        let (num_dom_rows, num_dom_cols) = match (
            height.checked_sub(2 * params.size),
            width.checked_sub(2 * params.size),
        ) {
            (Some(max_dr), Some(max_dc)) => (max_dr / params.shift + 1, max_dc / params.shift + 1),
            _ => (0, 0),
        };

        let params_gpu = ParamsGpu {
            width,
            height,
            size: params.size,
            shift: params.shift,
            contracted_stride: contracted_stride as u32,
            num_blocks_x,
            num_dom_rows,
            num_dom_cols,
            bits_alfa: params.bits_alfa,
            bits_beta: params.bits_beta,
            max_alfa: params.max_alfa,
            _pad: 0,
        };

        let num_blocks = (num_blocks_x * num_blocks_y) as usize;

        use wgpu::util::DeviceExt;
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mars-gpu params"),
                contents: bytemuck::bytes_of(&params_gpu),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let image_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mars-gpu image"),
                contents: bytemuck::cast_slice(&image_px),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let contracted_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mars-gpu contracted"),
                contents: bytemuck::cast_slice(&contracted),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let results_size =
            (num_blocks * slots_per_block as usize * std::mem::size_of::<CandidateGpu>()) as u64;
        let results_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mars-gpu results"),
            size: results_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mars-gpu readback"),
            size: results_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mars-gpu bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: image_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: contracted_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        if debug_timing {
            eprintln!("  [gpu timing] buffers+bind group: {:?}", t_start.elapsed());
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mars-gpu search encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mars-gpu search pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_blocks_x, num_blocks_y, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &readback_buf, 0, results_size);
        let t_submit = std::time::Instant::now();
        self.queue.submit(Some(encoder.finish()));
        if debug_timing {
            eprintln!("  [gpu timing] submit() returned: {:?}", t_submit.elapsed());
        }

        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let t_poll = std::time::Instant::now();
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll failed");
        if debug_timing {
            eprintln!("  [gpu timing] poll(wait): {:?}", t_poll.elapsed());
        }
        let t_recv = std::time::Instant::now();
        rx.recv()
            .expect("map_async callback dropped without a result")
            .expect("failed to map GPU results buffer for readback");
        if debug_timing {
            eprintln!("  [gpu timing] rx.recv(): {:?}", t_recv.elapsed());
        }

        let data = slice
            .get_mapped_range()
            .expect("failed to get mapped range of GPU results buffer");
        if debug_timing {
            eprintln!("  [gpu timing] TOTAL search(): {:?}", t_start.elapsed());
        }
        let raw: &[CandidateGpu] = bytemuck::cast_slice(&data);
        let out: Vec<Vec<CandidateGpu>> = raw
            .chunks_exact(slots_per_block as usize)
            .map(<[CandidateGpu]>::to_vec)
            .collect();
        drop(data);
        readback_buf.unmap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GPU may not be available in every CI/sandbox environment, so tests that need one
    /// skip (rather than fail) when `GpuSearcher::new` cannot find an adapter -- the real
    /// bit-identical check lives in `marsbench gpu-search-check`, which is meant to be run
    /// on the target M3 hardware per the plan, not under an arbitrary sandbox.
    fn searcher_or_skip() -> Option<GpuSearcher> {
        match GpuSearcher::new() {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    #[test]
    fn flat_image_matches_the_flat128_worked_example() {
        // Same constant-128 block `mars_codec::encode`'s `fit_f64_matches_the_flat128_
        // worked_example` test exercises: `det == 0` forces `alfa = 0`, and the residual
        // rms is not exactly zero because `qbeta`'s quantisation (128/255*127 -> round 64)
        // does not land back on exactly 128 when dequantised — it is ~0.504, not ~0.
        let Some(gpu) = searcher_or_skip() else {
            return;
        };
        let image = Plane::filled(64, 64, 128);
        let params = GpuSearchParams {
            size: 8,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
        };
        let results = gpu.search(&image, &params);
        assert_eq!(results.len(), 64);
        for c in results.into_iter().flatten() {
            assert_eq!(c.qalfa, 0, "det==0 on a flat block forces alfa to 0");
            assert!(
                (c.rms - 0.503_937).abs() < 1e-4,
                "expected the flat128 worked example's rms, got {}",
                c.rms
            );
        }
    }

    #[test]
    fn too_small_image_yields_no_candidates() {
        let Some(gpu) = searcher_or_skip() else {
            return;
        };
        let image = Plane::filled(8, 8, 10);
        let params = GpuSearchParams {
            size: 8,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
        };
        let results = gpu.search(&image, &params);
        assert_eq!(results, vec![None]);
    }

    /// Step 8's own correctness bar (P8.3 / gate-8's self-test, exercised here at unit
    /// scale rather than across a whole oracle build): the top-32 kernel's rank-0 entry
    /// must exactly equal the top-1 kernel's independently-computed winner for every
    /// block, for both the flat (degenerate, `det==0`) and a non-degenerate synthetic
    /// image. A mismatch here would mean the merge-reduce tree can lose the true minimum
    /// -- see `search.wgsl`'s module doc.
    #[test]
    fn top32_rank0_matches_top1_on_flat_image() {
        let Some(gpu) = searcher_or_skip() else {
            return;
        };
        let image = Plane::filled(64, 64, 128);
        let params = GpuSearchParams {
            size: 8,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
        };
        let top1 = gpu.search(&image, &params);
        let top32 = gpu.search_top32(&image, &params);
        assert_eq!(top1.len(), top32.len());
        for (t1, t32) in top1.iter().zip(top32.iter()) {
            match (t1, t32) {
                (None, None) => {}
                (Some(a), Some(list)) => {
                    let best = list.first().expect("Some(list) is never empty");
                    assert_eq!(*a, *best, "top-32 rank-0 must equal the top-1 winner");
                    // Ascending by rms, and self-consistent length bound.
                    assert!(list.len() <= 32);
                    for w in list.windows(2) {
                        assert!(w[0].rms <= w[1].rms, "top-32 list must be rms-ascending");
                    }
                }
                _ => panic!("top-1 and top-32 disagree on whether this block has candidates"),
            }
        }
    }

    #[test]
    fn top32_rank0_matches_top1_on_synthetic_image() {
        let Some(gpu) = searcher_or_skip() else {
            return;
        };
        // Same small xorshift PRNG pattern `mars-bench`'s gpu_search differential test
        // uses, so this is a non-degenerate image (a flat/constant image can hide a
        // broken fit behind `alfa == 0` short-circuits).
        struct Rng(u64);
        impl Rng {
            fn next_u64(&mut self) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0
            }
            fn byte(&mut self) -> u8 {
                (self.next_u64() & 0xff) as u8
            }
        }
        let (w, h) = (64usize, 64usize);
        let mut rng = Rng(0xC0FFEE);
        let data: Vec<u8> = (0..w * h).map(|_| rng.byte()).collect();
        let image = Plane::from_vec(w, h, data);

        for &size in &[4u32, 8, 16] {
            let params = GpuSearchParams {
                size,
                shift: 4,
                bits_alfa: 4,
                bits_beta: 7,
                max_alfa: 1.0,
            };
            let top1 = gpu.search(&image, &params);
            let top32 = gpu.search_top32(&image, &params);
            assert_eq!(top1.len(), top32.len());
            for (t1, t32) in top1.iter().zip(top32.iter()) {
                match (t1, t32) {
                    (None, None) => {}
                    (Some(a), Some(list)) => {
                        let best = list.first().unwrap();
                        assert_eq!(
                            *a, *best,
                            "size={size}: top-32 rank-0 must equal the top-1 winner"
                        );
                        for w in list.windows(2) {
                            assert!(w[0].rms <= w[1].rms, "size={size}: not rms-ascending");
                        }
                    }
                    _ => panic!("size={size}: top-1/top-32 disagree on candidate presence"),
                }
            }
        }
    }
}
