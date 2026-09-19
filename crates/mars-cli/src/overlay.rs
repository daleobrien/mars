//! Debug overlay for `encmars --debug-regions`: draw the active human-adaptive regions onto
//! a copy of the image so the effect of `--human-adaptive` can be checked visually.
//!
//! The rectangles drawn are exactly the [`HumanRegion`]s the encoder is handed -- the same
//! list, not a re-derived approximation -- so the overlay cannot disagree with the encode.
//!
//! [`draw_regions`] is pure (it edits a copy of the codec's own [`Image`]) and is unit
//! tested without a model or the `face-detect` feature; only writing the file needs a PNG
//! encoder.

use mars_core::image::{ColorSpace, Image, Plane};

use crate::human::{HumanRegion, RegionKind};

/// Outline thickness in pixels. Thick enough to stay visible over busy detail.
const THICKNESS: usize = 2;

/// Outline value per channel for a region kind. Grayscale has a single channel, so the two
/// kinds are told apart by brightness there instead of hue.
fn outline_colour(kind: RegionKind, colour_space: ColorSpace) -> [u8; 3] {
    match colour_space {
        ColorSpace::Gray => match kind {
            RegionKind::Face => [255, 255, 255],
            RegionKind::Feature => [128, 128, 128],
        },
        ColorSpace::Rgb => match kind {
            RegionKind::Face => [0, 255, 0],
            RegionKind::Feature => [255, 0, 0],
        },
    }
}

/// Draw every region's outline onto a copy of `image`, leaving the original untouched.
///
/// Outlines are clipped to the image, so a region that reaches the edge (or lies entirely
/// off it) cannot panic or wrap around. An empty region list returns an unchanged copy.
pub fn draw_regions(image: &Image, regions: &[HumanRegion]) -> Image {
    let width = image.width();
    let height = image.height();
    let mut planes: Vec<Vec<u8>> = image
        .planes()
        .iter()
        .map(|plane| plane.as_slice().to_vec())
        .collect();

    for planned in regions {
        let colour = outline_colour(planned.kind, image.color());
        let region = planned.region;
        let top = (region.row as usize).min(height);
        let left = (region.col as usize).min(width);
        let bottom = ((region.row + region.height) as usize).min(height);
        let right = ((region.col + region.width) as usize).min(width);
        if bottom <= top || right <= left {
            continue;
        }
        for y in top..bottom {
            for x in left..right {
                let on_border = y - top < THICKNESS
                    || bottom - 1 - y < THICKNESS
                    || x - left < THICKNESS
                    || right - 1 - x < THICKNESS;
                if !on_border {
                    continue;
                }
                for (plane_index, data) in planes.iter_mut().enumerate() {
                    data[y * width + x] = colour[plane_index];
                }
            }
        }
    }

    let rebuilt: Vec<Plane> = planes
        .into_iter()
        .map(|data| Plane::from_vec(width, height, data))
        .collect();
    match image.color() {
        ColorSpace::Gray => Image::gray(rebuilt.into_iter().next().expect("gray has one plane")),
        ColorSpace::Rgb => {
            let mut planes = rebuilt.into_iter();
            Image::rgb(
                planes.next().expect("rgb has three planes"),
                planes.next().expect("rgb has three planes"),
                planes.next().expect("rgb has three planes"),
            )
        }
    }
}

/// Write `image` to `path`. The extension selects the format, so pass a `.png` path.
#[cfg(feature = "face-detect")]
pub fn write_image(image: &Image, path: &std::path::Path) -> anyhow::Result<()> {
    use anyhow::Context;

    let (width, height) = (image.width() as u32, image.height() as u32);
    let dynamic = match image.color() {
        ColorSpace::Gray => image::DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(width, height, image.planes()[0].as_slice().to_vec())
                .context("the gray buffer does not match the image dimensions")?,
        ),
        ColorSpace::Rgb => {
            let planes = image.planes();
            let pixels = (width as usize) * (height as usize);
            let mut data = Vec::with_capacity(pixels * 3);
            for index in 0..pixels {
                data.push(planes[0].as_slice()[index]);
                data.push(planes[1].as_slice()[index]);
                data.push(planes[2].as_slice()[index]);
            }
            image::DynamicImage::ImageRgb8(
                image::RgbImage::from_raw(width, height, data)
                    .context("the rgb buffer does not match the image dimensions")?,
            )
        }
    };
    dynamic.save(path).with_context(|| {
        format!(
            "saving {} (the extension selects the format)",
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mars_codec::encode::LambdaRegion;

    fn region(row: u32, col: u32, height: u32, width: u32, kind: RegionKind) -> HumanRegion {
        HumanRegion {
            kind,
            region: LambdaRegion {
                row,
                col,
                height,
                width,
                // Irrelevant to drawing; the plan's real scales come from the CLI flags.
                scale: 1.0,
            },
        }
    }

    #[test]
    fn an_outline_is_drawn_on_the_border_and_the_interior_is_left_alone() {
        let image = Image::gray(Plane::filled(16, 16, 0));
        let drawn = draw_regions(&image, &[region(4, 4, 8, 8, RegionKind::Face)]);
        let plane = &drawn.planes()[0];
        assert_eq!(plane.get(4, 4), 255, "top-left corner of the outline");
        assert_eq!(plane.get(11, 11), 255, "bottom-right corner of the outline");
        assert_eq!(plane.get(0, 0), 0, "outside the region is untouched");
        assert_eq!(
            plane.get(7, 7),
            0,
            "a 2px border on an 8x8 region leaves a 4x4 interior untouched"
        );
    }

    #[test]
    fn the_two_kinds_get_different_gray_levels() {
        let image = Image::gray(Plane::filled(8, 8, 0));
        let drawn = draw_regions(
            &image,
            &[
                region(0, 0, 4, 4, RegionKind::Face),
                region(4, 4, 4, 4, RegionKind::Feature),
            ],
        );
        let plane = &drawn.planes()[0];
        assert_eq!(plane.get(0, 0), 255, "face outline is white");
        assert_eq!(plane.get(4, 4), 128, "feature outline is mid-grey");
    }

    #[test]
    fn the_two_kinds_get_different_colours_on_rgb() {
        let image = Image::rgb(
            Plane::filled(8, 8, 0),
            Plane::filled(8, 8, 0),
            Plane::filled(8, 8, 0),
        );
        let drawn = draw_regions(
            &image,
            &[
                region(0, 0, 4, 4, RegionKind::Face),
                region(4, 4, 4, 4, RegionKind::Feature),
            ],
        );
        let planes = drawn.planes();
        let pixel = |x: usize, y: usize| {
            (
                planes[0].get(x, y),
                planes[1].get(x, y),
                planes[2].get(x, y),
            )
        };
        assert_eq!(pixel(0, 0), (0, 255, 0), "face outline is green");
        assert_eq!(pixel(4, 4), (255, 0, 0), "feature outline is red");
    }

    #[test]
    fn empty_and_off_image_regions_are_ignored_without_panicking() {
        let image = Image::gray(Plane::filled(8, 8, 0));
        let drawn = draw_regions(
            &image,
            &[
                region(100, 100, 4, 4, RegionKind::Face), // entirely off-image
                region(2, 2, 0, 0, RegionKind::Feature),  // zero-sized
            ],
        );
        assert_eq!(drawn, image, "nothing to draw leaves the image unchanged");
    }

    /// The PNG writer is the one part of `--debug-regions` a unit test can exercise without
    /// a model: draw, write, read back.
    #[cfg(feature = "face-detect")]
    #[test]
    fn the_written_overlay_round_trips_through_the_png_encoder() {
        let image = Image::rgb(
            Plane::filled(8, 8, 10),
            Plane::filled(8, 8, 20),
            Plane::filled(8, 8, 30),
        );
        let drawn = draw_regions(&image, &[region(0, 0, 4, 4, RegionKind::Feature)]);
        let path = std::env::temp_dir().join(format!("mars-overlay-{}.png", std::process::id()));
        write_image(&drawn, &path).expect("writing the overlay must succeed");

        let decoded = image::open(&path).expect("reading the overlay back");
        assert_eq!((decoded.width(), decoded.height()), (8, 8));
        let rgb = decoded.to_rgb8();
        assert_eq!(
            rgb.get_pixel(0, 0).0,
            [255, 0, 0],
            "the feature outline survives the round trip as red"
        );
        assert_eq!(
            rgb.get_pixel(5, 5).0,
            [10, 20, 30],
            "the untouched interior survives the round trip"
        );
        let _ = std::fs::remove_file(&path);
    }
}
