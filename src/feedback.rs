//! Short synthesized cues that confirm wake-word state changes.
//!
//! The cues are generated instead of shipped as files so the app stays a
//! single binary and does not depend on a sound theme being installed.

use std::sync::OnceLock;

const SAMPLE_RATE: u32 = 48_000;
const AMPLITUDE: f32 = 0.16;
/// Fade in/out so the tones do not click.
const FADE_SECS: f32 = 0.006;
/// Gap between the notes of a cue.
const GAP_SECS: f32 = 0.012;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cue {
    /// Wake word detected; live listening is about to start.
    Wake,
    /// The wake-triggered listening window timed out.
    Timeout,
    /// The user ended the conversation with a sleep phrase.
    Sleep,
}

impl Cue {
    fn samples(self) -> Vec<f32> {
        match self {
            // Rising pair: "I heard you".
            Cue::Wake => sequence(&[(700.0, 0.09), (1_050.0, 0.12)]),
            // Falling pair: "I stopped listening".
            Cue::Timeout => sequence(&[(700.0, 0.09), (520.0, 0.14)]),
            // Three descending notes: "conversation closed".
            Cue::Sleep => sequence(&[(620.0, 0.08), (466.0, 0.10), (349.0, 0.14)]),
        }
    }
}

fn sequence(notes: &[(f32, f32)]) -> Vec<f32> {
    let mut out = Vec::new();
    for (index, (frequency, seconds)) in notes.iter().enumerate() {
        if index > 0 {
            out.extend(std::iter::repeat_n(
                0.0,
                (SAMPLE_RATE as f32 * GAP_SECS) as usize,
            ));
        }
        let count = (SAMPLE_RATE as f32 * seconds) as usize;
        let fade = ((SAMPLE_RATE as f32 * FADE_SECS) as usize)
            .min(count / 2)
            .max(1);
        for index in 0..count {
            let envelope = if index < fade {
                index as f32 / fade as f32
            } else if index + fade >= count {
                count.saturating_sub(index + 1) as f32 / fade as f32
            } else {
                1.0
            };
            let time = index as f32 / SAMPLE_RATE as f32;
            let sample = (time * frequency * std::f32::consts::TAU).sin();
            out.push(sample * envelope * AMPLITUDE);
        }
    }
    out
}

static CUES: OnceLock<async_channel::Sender<Cue>> = OnceLock::new();

fn sender() -> &'static async_channel::Sender<Cue> {
    CUES.get_or_init(|| {
        let (tx, rx) = async_channel::unbounded::<Cue>();
        let spawned = std::thread::Builder::new()
            .name("ollama-gui-cue".to_string())
            .spawn(move || {
                let stream = match rodio::OutputStreamBuilder::open_default_stream() {
                    Ok(stream) => stream,
                    Err(error) => {
                        tracing::warn!("feedback: audio output unavailable: {error}");
                        return;
                    }
                };
                while let Ok(cue) = rx.recv_blocking() {
                    tracing::debug!("feedback: playing {cue:?}");
                    let sink = rodio::Sink::connect_new(stream.mixer());
                    sink.append(rodio::buffer::SamplesBuffer::new(
                        1,
                        SAMPLE_RATE,
                        cue.samples(),
                    ));
                    sink.sleep_until_end();
                }
            });
        if let Err(error) = spawned {
            tracing::warn!("feedback: could not start the cue thread: {error}");
        }
        tx
    })
}

/// Queue a cue for playback. Never blocks and never touches the GTK loop.
pub fn play(cue: Cue) {
    if sender().try_send(cue).is_err() {
        tracing::debug!("feedback: cue queue is full");
    }
}

#[cfg(test)]
mod tests {
    use super::{sequence, Cue, AMPLITUDE, SAMPLE_RATE};

    #[test]
    fn cues_are_short_and_bounded() {
        for cue in [Cue::Wake, Cue::Timeout, Cue::Sleep] {
            let samples = cue.samples();
            let seconds = samples.len() as f32 / SAMPLE_RATE as f32;
            assert!((0.2..0.4).contains(&seconds), "{cue:?} lasted {seconds}s");
            let peak = samples
                .iter()
                .fold(0.0f32, |peak, sample| peak.max(sample.abs()));
            assert!(peak <= AMPLITUDE + f32::EPSILON);
            assert!(peak > AMPLITUDE * 0.5, "{cue:?} was too quiet");
        }
    }

    #[test]
    fn wake_and_timeout_are_different() {
        assert_ne!(Cue::Wake.samples(), Cue::Timeout.samples());
        assert_ne!(Cue::Wake.samples(), Cue::Sleep.samples());
        assert_ne!(Cue::Timeout.samples(), Cue::Sleep.samples());
    }

    #[test]
    fn notes_start_and_end_at_zero() {
        let samples = sequence(&[(440.0, 0.05)]);
        assert!(samples.first().copied().unwrap().abs() < 1e-6);
        assert!(samples.last().copied().unwrap().abs() < 1e-6);
    }
}
