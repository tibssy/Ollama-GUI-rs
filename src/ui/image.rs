use anyhow::{Context, Result};
use gtk::prelude::*;
use gtk::{gdk, gio, glib};

use crate::store::StoredImage;

/// Maximum number of images attached to one message.
pub const MAX_ATTACHMENTS: usize = 4;
/// Reject files larger than this before decoding them.
pub const MAX_FILE_BYTES: usize = 20 * 1024 * 1024;

/// Longest edge kept when an image is stored and sent to the model.
const MAX_SOURCE_DIM: i32 = 1024;
/// Upper bound on decoded pixels after scaling.
const MAX_SOURCE_PIXELS: i64 = 1_000_000;
/// Keep the encoded payload below this size so a model cannot be overwhelmed
/// by a single image request.
const MAX_ENCODED_BYTES: usize = 1_500_000;
/// Smallest edge kept while shrinking an oversized encoded image.
const MIN_ENCODED_DIM: i32 = 256;
/// Longest edge of the composer thumbnail.
const PREVIEW_DIM: i32 = 72;
/// Reference texture for message images. The chat bubble itself is laid out at
/// half of the message column, and the picture keeps this 4:3 aspect ratio.
const MESSAGE_IMAGE_WIDTH: i32 = 400;
const MESSAGE_IMAGE_HEIGHT: i32 = 300;

pub struct ProcessedImage {
    pub mime: String,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub preview_data: Vec<u8>,
}

/// Decode an image, downscale it to a conservative vision-model size, and
/// re-encode it. Re-encoding also drops most embedded metadata such as EXIF
/// location tags.
pub fn process(data: &[u8]) -> Result<ProcessedImage> {
    // Scale while decoding so a decompression bomb cannot allocate a huge
    // intermediate buffer before we get a chance to resize it. The size-prepared
    // callback only scales down; it never enlarges a small image.
    let loader = gdk_pixbuf::PixbufLoader::new();
    let sized = loader.clone();
    loader.connect_size_prepared(move |_, width, height| {
        let (width, height) = fit_dimensions(
            width as u32,
            height as u32,
            MAX_SOURCE_DIM,
            MAX_SOURCE_PIXELS,
        );
        sized.set_size(width, height);
    });
    loader
        .write(data)
        .map_err(|err| anyhow::anyhow!("could not decode image: {err}"))?;
    loader
        .close()
        .map_err(|err| anyhow::anyhow!("could not decode image: {err}"))?;
    let pixbuf = loader
        .pixbuf()
        .context("image decoder returned no pixbuf")?;
    let pixbuf = pixbuf.apply_embedded_orientation().unwrap_or(pixbuf);
    let (mime, data, pixbuf) = encode_within_budget(pixbuf, MAX_ENCODED_BYTES)?;

    let width = pixbuf.width() as u32;
    let height = pixbuf.height() as u32;

    let preview = downscale_to_fit(pixbuf, PREVIEW_DIM, (PREVIEW_DIM as i64).pow(2))
        .context("could not scale preview")?;
    let preview_data = preview
        .save_to_bufferv("png", &[])
        .map_err(|err| anyhow::anyhow!("could not encode preview: {err}"))?;

    Ok(ProcessedImage {
        mime: mime.to_string(),
        width,
        height,
        data,
        preview_data,
    })
}

fn downscale_to_fit(
    pixbuf: gdk_pixbuf::Pixbuf,
    max_dim: i32,
    max_pixels: i64,
) -> Option<gdk_pixbuf::Pixbuf> {
    let (width, height) = fit_dimensions(
        pixbuf.width() as u32,
        pixbuf.height() as u32,
        max_dim,
        max_pixels,
    );
    if width == pixbuf.width() && height == pixbuf.height() {
        return Some(pixbuf);
    }
    pixbuf.scale_simple(width, height, gdk_pixbuf::InterpType::Bilinear)
}

fn fit_dimensions(width: u32, height: u32, max_dim: i32, max_pixels: i64) -> (i32, i32) {
    let width = width.max(1) as f64;
    let height = height.max(1) as f64;
    let mut scale = (max_dim as f64 / width.max(height)).min(1.0);

    let pixels = width * height * scale * scale;
    if pixels > max_pixels as f64 {
        scale *= (max_pixels as f64 / pixels).sqrt();
    }

    (
        (width * scale).round().max(1.0) as i32,
        (height * scale).round().max(1.0) as i32,
    )
}

fn encode(pixbuf: &gdk_pixbuf::Pixbuf) -> Result<(&'static str, Vec<u8>)> {
    if pixbuf.has_alpha() {
        return Ok((
            "image/png",
            pixbuf
                .save_to_bufferv("png", &[])
                .map_err(|err| anyhow::anyhow!("could not encode image: {err}"))?,
        ));
    }

    match pixbuf.save_to_bufferv("jpeg", &[("quality", "88")]) {
        Ok(data) => Ok(("image/jpeg", data)),
        Err(_) => Ok((
            "image/png",
            pixbuf
                .save_to_bufferv("png", &[])
                .map_err(|err| anyhow::anyhow!("could not encode image: {err}"))?,
        )),
    }
}

fn encode_within_budget(
    mut pixbuf: gdk_pixbuf::Pixbuf,
    max_bytes: usize,
) -> Result<(&'static str, Vec<u8>, gdk_pixbuf::Pixbuf)> {
    loop {
        let (mime, data) = encode(&pixbuf)?;
        if data.len() <= max_bytes || pixbuf.width().min(pixbuf.height()) <= MIN_ENCODED_DIM {
            return Ok((mime, data, pixbuf));
        }

        let max_dim = ((pixbuf.width().max(pixbuf.height()) as f64 * 0.8).round() as i32).max(1);
        let (width, height) = fit_dimensions(
            pixbuf.width() as u32,
            pixbuf.height() as u32,
            max_dim,
            i64::MAX,
        );
        let Some(scaled) = pixbuf.scale_simple(width, height, gdk_pixbuf::InterpType::Bilinear)
        else {
            return Ok((mime, data, pixbuf));
        };
        pixbuf = scaled;
    }
}

pub fn texture(data: &[u8]) -> Option<gdk::Texture> {
    let bytes = glib::Bytes::from(data);
    gdk::Texture::from_bytes(&bytes).ok()
}

fn frame_dimensions(width: u32, height: u32, frame_width: i32, frame_height: i32) -> (i32, i32) {
    if width == 0 || height == 0 {
        return (1, 1);
    }
    let width = width as f64;
    let height = height as f64;
    let scale = (frame_width as f64 / width).min(frame_height as f64 / height);
    (
        (width * scale).round().max(1.0) as i32,
        (height * scale).round().max(1.0) as i32,
    )
}

/// Compose an image into a fixed-size transparent frame. This forces the widget
/// to keep the same dimensions instead of expanding to the texture's natural
/// size, which made some chat images much wider than others.
fn framed_texture(data: &[u8], frame_width: i32, frame_height: i32) -> Option<gdk::Texture> {
    let bytes = glib::Bytes::from(data);
    let stream = gio::MemoryInputStream::from_bytes(&bytes);
    let pixbuf = gdk_pixbuf::Pixbuf::from_stream(&stream, None::<&gio::Cancellable>).ok()?;

    let (width, height) = frame_dimensions(
        pixbuf.width() as u32,
        pixbuf.height() as u32,
        frame_width,
        frame_height,
    );
    let scaled = if width == pixbuf.width() && height == pixbuf.height() {
        pixbuf
    } else {
        pixbuf.scale_simple(width, height, gdk_pixbuf::InterpType::Bilinear)?
    };
    let scaled = if scaled.has_alpha() {
        scaled
    } else {
        scaled.add_alpha(false, 0, 0, 0).ok()?
    };

    let frame = gdk_pixbuf::Pixbuf::new(
        gdk_pixbuf::Colorspace::Rgb,
        true,
        8,
        frame_width,
        frame_height,
    )?;
    frame.fill(0x00000000);
    scaled.copy_area(
        0,
        0,
        width,
        height,
        &frame,
        (frame_width - width) / 2,
        (frame_height - height) / 2,
    );

    let png = frame.save_to_bufferv("png", &[]).ok()?;
    gdk::Texture::from_bytes(&glib::Bytes::from(&png)).ok()
}

/// Fixed-size composer thumbnail, independent of the source aspect ratio.
pub fn preview_widget(data: &[u8]) -> gtk::Widget {
    let Some(texture) = framed_texture(data, PREVIEW_DIM, PREVIEW_DIM) else {
        return gtk::Label::new(Some("Image")).upcast();
    };

    let picture = gtk::Picture::for_paintable(&texture);
    picture.set_size_request(PREVIEW_DIM, PREVIEW_DIM);
    picture.set_content_fit(gtk::ContentFit::Contain);
    picture.set_can_shrink(true);
    picture.add_css_class("attachment-image");
    picture.upcast()
}

/// Build the image shown inside a user bubble. The picture expands to the size
/// of its aspect frame, so every message image fills the same right-half card.
/// Clicking opens a larger view.
pub fn message_picture(image: &StoredImage, parent: &gtk::Window) -> gtk::Widget {
    let Some(texture) = framed_texture(&image.data, MESSAGE_IMAGE_WIDTH, MESSAGE_IMAGE_HEIGHT)
    else {
        let label = gtk::Label::new(Some("Image unavailable"));
        label.add_css_class("dim-label");
        return label.upcast();
    };

    let picture = gtk::Picture::for_paintable(&texture);
    picture.set_content_fit(gtk::ContentFit::Contain);
    picture.set_can_shrink(true);
    picture.set_hexpand(true);
    picture.add_css_class("message-image");
    picture.set_tooltip_text(Some(&format!(
        "Open image ({}×{})",
        image.width, image.height
    )));
    picture.set_cursor_from_name(Some("pointer"));

    let gesture = gtk::GestureClick::new();
    let data = image.data.clone();
    let parent = parent.clone();
    gesture.connect_released(move |_, _, _, _| open_dialog(&parent, &data));
    picture.add_controller(gesture);

    picture.upcast()
}

fn open_dialog(parent: &gtk::Window, data: &[u8]) {
    let Some(texture) = texture(data) else {
        return;
    };

    let picture = gtk::Picture::for_paintable(&texture);
    picture.set_can_shrink(true);
    picture.set_content_fit(gtk::ContentFit::Contain);
    picture.set_hexpand(true);
    picture.set_vexpand(true);

    let window = gtk::Window::builder()
        .title("Image")
        .default_width(900)
        .default_height(700)
        .transient_for(parent)
        .build();
    window.set_child(Some(&picture));
    window.present();
}

#[cfg(test)]
mod tests {
    use super::{fit_dimensions, frame_dimensions};

    #[test]
    fn fits_images_inside_the_model_budget() {
        assert_eq!(fit_dimensions(4000, 3000, 1024, 1_000_000), (1024, 768));
        assert_eq!(fit_dimensions(100, 50, 1024, 1_000_000), (100, 50));
        assert_eq!(fit_dimensions(2000, 2000, 1024, 1_000_000), (1000, 1000));
        assert_eq!(fit_dimensions(4000, 1000, 1024, 1_000_000), (1024, 256));
        assert_eq!(fit_dimensions(0, 0, 1024, 1_000_000), (1, 1));
    }

    #[test]
    fn fits_previews_into_the_thumbnail_box() {
        assert_eq!(fit_dimensions(1000, 500, 72, 72 * 72), (72, 36));
        assert_eq!(fit_dimensions(100, 100, 72, 72 * 72), (72, 72));
    }

    #[test]
    fn frames_images_into_a_consistent_box() {
        assert_eq!(frame_dimensions(100, 50, 400, 300), (400, 200));
        assert_eq!(frame_dimensions(1024, 768, 400, 300), (400, 300));
        assert_eq!(frame_dimensions(1024, 1024, 400, 300), (300, 300));
        assert_eq!(frame_dimensions(0, 0, 400, 300), (1, 1));
    }

    #[test]
    fn does_not_upscale_small_images() {
        let source = gdk_pixbuf::Pixbuf::new(gdk_pixbuf::Colorspace::Rgb, false, 8, 100, 50)
            .expect("source pixbuf");
        source.fill(0x336699ff);
        let encoded = source.save_to_bufferv("png", &[]).expect("encode source");

        let processed = super::process(&encoded).expect("process image");
        assert!(processed.width <= 100);
        assert!(processed.height <= 50);
    }

    #[test]
    fn scales_large_images_to_the_model_budget() {
        let source = gdk_pixbuf::Pixbuf::new(gdk_pixbuf::Colorspace::Rgb, false, 8, 4000, 3000)
            .expect("source pixbuf");
        source.fill(0x336699ff);
        let encoded = source.save_to_bufferv("png", &[]).expect("encode source");

        let processed = super::process(&encoded).expect("process image");
        assert!(processed.width <= 1024);
        assert!(processed.height <= 1024);
        assert!(processed.width as i64 * processed.height as i64 <= 1_000_000);
        assert!(processed.data.len() <= 1_500_000);
        assert_eq!(processed.mime, "image/jpeg");
        assert!(!processed.preview_data.is_empty());
    }
}
