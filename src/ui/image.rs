use anyhow::{Context, Result};
use gtk::prelude::*;
use gtk::{gdk, gio, glib};

use crate::store::StoredImage;

/// Maximum number of images attached to one message.
pub const MAX_ATTACHMENTS: usize = 4;
/// Reject files larger than this before decoding them.
pub const MAX_FILE_BYTES: usize = 20 * 1024 * 1024;

/// Longest edge kept when an image is stored and sent to the model.
const MAX_SOURCE_DIM: i32 = 1568;
/// Longest edge of the composer thumbnail.
const PREVIEW_DIM: i32 = 72;
/// Display bounds inside a chat bubble.
const MESSAGE_MAX_WIDTH: i32 = 320;
const MESSAGE_MAX_HEIGHT: i32 = 240;

pub struct ProcessedImage {
    pub mime: String,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub preview_data: Vec<u8>,
}

/// Decode an image, downscale it to a sane vision-model size, and re-encode it.
/// Re-encoding also drops most embedded metadata such as EXIF location tags.
pub fn process(data: &[u8]) -> Result<ProcessedImage> {
    let bytes = glib::Bytes::from(data);
    let stream = gio::MemoryInputStream::from_bytes(&bytes);
    let pixbuf = gdk_pixbuf::Pixbuf::from_stream(&stream, None::<&gio::Cancellable>)
        .map_err(|err| anyhow::anyhow!("could not decode image: {err}"))?;
    let pixbuf = pixbuf.apply_embedded_orientation().unwrap_or(pixbuf);
    let pixbuf = downscale(pixbuf, MAX_SOURCE_DIM).context("could not scale image")?;
    let width = pixbuf.width() as u32;
    let height = pixbuf.height() as u32;

    let (mime, data) = if pixbuf.has_alpha() {
        (
            "image/png",
            pixbuf
                .save_to_bufferv("png", &[])
                .map_err(|err| anyhow::anyhow!("could not encode image: {err}"))?,
        )
    } else {
        match pixbuf.save_to_bufferv("jpeg", &[("quality", "90")]) {
            Ok(data) => ("image/jpeg", data),
            Err(_) => (
                "image/png",
                pixbuf
                    .save_to_bufferv("png", &[])
                    .map_err(|err| anyhow::anyhow!("could not encode image: {err}"))?,
            ),
        }
    };

    let preview = downscale(pixbuf, PREVIEW_DIM).context("could not scale preview")?;
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

fn downscale(pixbuf: gdk_pixbuf::Pixbuf, max: i32) -> Option<gdk_pixbuf::Pixbuf> {
    let width = pixbuf.width();
    let height = pixbuf.height();
    if width <= max && height <= max {
        return Some(pixbuf);
    }

    let scale = max as f64 / width.max(height) as f64;
    let width = (width as f64 * scale).round().max(1.0) as i32;
    let height = (height as f64 * scale).round().max(1.0) as i32;
    pixbuf.scale_simple(width, height, gdk_pixbuf::InterpType::Bilinear)
}

pub fn texture(data: &[u8]) -> Option<gdk::Texture> {
    let bytes = glib::Bytes::from(data);
    gdk::Texture::from_bytes(&bytes).ok()
}

/// Fit an image into a box while preserving its aspect ratio.
pub fn display_size(width: u32, height: u32) -> (i32, i32) {
    if width == 0 || height == 0 {
        return (1, 1);
    }
    let scale = (MESSAGE_MAX_WIDTH as f64 / width as f64)
        .min(MESSAGE_MAX_HEIGHT as f64 / height as f64)
        .min(1.0);
    let width = (width as f64 * scale).round().max(1.0) as i32;
    let height = (height as f64 * scale).round().max(1.0) as i32;
    (width, height)
}

/// Build the image shown inside a user bubble. Clicking opens a larger view.
pub fn message_widget(image: &StoredImage, parent: &gtk::Window) -> gtk::Widget {
    let Some(texture) = texture(&image.data) else {
        let label = gtk::Label::new(Some("Image unavailable"));
        label.add_css_class("dim-label");
        return label.upcast();
    };

    let (width, height) = display_size(image.width, image.height);
    let picture = gtk::Picture::for_paintable(&texture);
    picture.set_size_request(width, height);
    picture.set_content_fit(gtk::ContentFit::Contain);
    picture.set_can_shrink(true);
    picture.set_halign(gtk::Align::Start);
    picture.add_css_class("message-image");
    picture.set_tooltip_text(Some("Open image"));
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
    use super::display_size;

    #[test]
    fn fits_images_inside_the_bubble() {
        assert_eq!(display_size(800, 600), (320, 240));
        assert_eq!(display_size(100, 50), (100, 50));
        assert_eq!(display_size(1000, 4000), (60, 240));
        assert_eq!(display_size(0, 0), (1, 1));
    }
}
