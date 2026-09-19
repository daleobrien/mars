//! `decmars`' output side: the output filename's extension selects the writer, and
//! `--quality` is a JPEG setting rather than a silent no-op on other formats.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use mars_core::io::read_image;

fn encmars_bin() -> &'static str {
    env!("CARGO_BIN_EXE_encmars")
}
fn decmars_bin() -> &'static str {
    env!("CARGO_BIN_EXE_decmars")
}

struct Scratch(PathBuf);
impl Scratch {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mars-decmars-output-{label}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_test_pgm(path: &Path) {
    let (w, h) = (48usize, 48usize);
    let mut data = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let v = ((x * 7 + y * 11) % 256) + (((x * x + y) % 37) * 5);
            data[y * w + x] = (v % 256) as u8;
        }
    }
    let mut pgm = format!("P5\n{w} {h}\n255\n").into_bytes();
    pgm.extend_from_slice(&data);
    std::fs::write(path, pgm).expect("write test pgm");
}

/// Encode the shared test image into a `.mars` stream once; every case below decodes it.
fn encoded(scratch: &Scratch) -> PathBuf {
    let pgm = scratch.path("in.pgm");
    write_test_pgm(&pgm);
    let mars = scratch.path("out.mars");
    let status = Command::new(encmars_bin())
        .arg(&pgm)
        .arg(&mars)
        .args(["--t-rms", "8"])
        .status()
        .expect("encmars runs");
    assert!(status.success(), "encmars failed on the test pgm");
    mars
}

fn decode(scratch: &Scratch, mars: &Path, name: &str, extra: &[&str]) -> Output {
    Command::new(decmars_bin())
        .arg(mars)
        .arg(scratch.path(name))
        .args(["-i", "3"])
        .args(extra)
        .output()
        .expect("decmars runs")
}

#[test]
fn jpeg_output_is_selected_by_extension_and_quality_changes_the_size() {
    let scratch = Scratch::new("jpeg");
    let mars = encoded(&scratch);
    let low = decode(&scratch, &mars, "low.jpg", &["--quality", "10"]);
    let high = decode(&scratch, &mars, "high.jpg", &["--quality", "95"]);
    assert!(
        low.status.success(),
        "{}",
        String::from_utf8_lossy(&low.stderr)
    );
    assert!(
        high.status.success(),
        "{}",
        String::from_utf8_lossy(&high.stderr)
    );

    let low_bytes = std::fs::read(scratch.path("low.jpg")).unwrap();
    let high_bytes = std::fs::read(scratch.path("high.jpg")).unwrap();
    assert!(
        low_bytes.starts_with(&[0xFF, 0xD8, 0xFF]),
        "a `.jpg` output is not a JPEG"
    );
    assert!(
        high_bytes.starts_with(&[0xFF, 0xD8, 0xFF]),
        "a `.jpg` output is not a JPEG"
    );
    assert!(
        high_bytes.len() > low_bytes.len(),
        "quality 95 ({} bytes) should exceed quality 10 ({} bytes)",
        high_bytes.len(),
        low_bytes.len()
    );

    let image = read_image(&scratch.path("high.jpg"), None).unwrap();
    assert_eq!((image.width(), image.height()), (48, 48));
    assert_eq!(image.planes().len(), 1);
}

#[test]
fn quality_with_a_non_jpeg_output_is_refused() {
    let scratch = Scratch::new("refused");
    let mars = encoded(&scratch);
    let output = decode(&scratch, &mars, "out.png", &["--quality", "50"]);
    assert!(
        !output.status.success(),
        "--quality must not be silently ignored for a non-JPEG output"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--quality"), "{stderr}");
    assert!(
        !scratch.path("out.png").exists(),
        "nothing should be written when the options are refused"
    );
}

#[test]
fn quality_out_of_range_is_rejected() {
    let scratch = Scratch::new("range");
    let mars = encoded(&scratch);
    let output = decode(&scratch, &mars, "out.jpg", &["--quality", "0"]);
    assert!(!output.status.success(), "--quality 0 must be rejected");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("quality"), "{stderr}");
}

#[test]
fn tga_tiff_and_webp_outputs_are_selected_by_extension() {
    let scratch = Scratch::new("formats");
    let mars = encoded(&scratch);
    for (name, expected_planes) in [
        ("out.tga", 1),
        ("out.tif", 1),
        ("out.tiff", 1),
        // WebP has no grayscale mode, so a gray decode reads back as three equal planes.
        ("out.webp", 3),
    ] {
        let output = decode(&scratch, &mars, name, &[]);
        assert!(
            output.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let image = read_image(&scratch.path(name), None).unwrap();
        assert_eq!((image.width(), image.height()), (48, 48), "{name}");
        assert_eq!(image.planes().len(), expected_planes, "{name}");
    }
}

#[test]
fn png_output_still_writes_without_quality() {
    let scratch = Scratch::new("png");
    let mars = encoded(&scratch);
    let output = decode(&scratch, &mars, "out.png", &[]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let image = read_image(&scratch.path("out.png"), None).unwrap();
    assert_eq!((image.width(), image.height()), (48, 48));
}
