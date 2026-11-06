/// Incremental newline-delimited JSON splitter.
///
/// Ollama streams one JSON object per line, and network chunks may contain
/// partial lines or several lines at once.
pub struct NdjsonBuffer {
    buf: Vec<u8>,
}

impl Default for NdjsonBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl NdjsonBuffer {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Feed a chunk and return every complete line it completed.
    pub fn push(&mut self, data: &[u8]) -> Vec<Vec<u8>> {
        self.buf.extend_from_slice(data);
        let mut lines = Vec::new();
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let mut line: Vec<u8> = self.buf.drain(..=pos).collect();
            line.pop(); // '\n'
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if !line.is_empty() {
                lines.push(line);
            }
        }
        lines
    }

    /// Return a trailing line that was not newline-terminated.
    pub fn finish(&mut self) -> Option<Vec<u8>> {
        if self.buf.is_empty() {
            return None;
        }
        let line = std::mem::take(&mut self.buf);
        let line = line
            .into_iter()
            .filter(|b| *b != b'\r')
            .collect::<Vec<u8>>();
        (!line.is_empty()).then_some(line)
    }
}

#[cfg(test)]
mod tests {
    use super::NdjsonBuffer;

    #[test]
    fn splits_complete_lines() {
        let mut buf = NdjsonBuffer::new();
        let lines = buf.push(b"{\"a\":1}\n{\"b\":2}\n");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], b"{\"a\":1}");
    }

    #[test]
    fn reassembles_partial_lines() {
        let mut buf = NdjsonBuffer::new();
        assert!(buf.push(b"{\"mess").is_empty());
        let lines = buf.push(b"age\":\"hi\"}\n");
        assert_eq!(lines, vec![b"{\"message\":\"hi\"}".to_vec()]);
    }

    #[test]
    fn handles_crlf_and_blank_lines() {
        let mut buf = NdjsonBuffer::new();
        let lines = buf.push(b"\r\n{\"a\":1}\r\n\n");
        assert_eq!(lines, vec![b"{\"a\":1}".to_vec()]);
    }

    #[test]
    fn finish_returns_unterminated_line() {
        let mut buf = NdjsonBuffer::new();
        assert!(buf.push(b"{\"done\":true}").is_empty());
        assert_eq!(buf.finish(), Some(b"{\"done\":true}".to_vec()));
        assert_eq!(buf.finish(), None);
    }
}
