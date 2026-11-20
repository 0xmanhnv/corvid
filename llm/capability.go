package llm

// SupportsVision reports whether a wire format can carry image and document
// content blocks (BlockImage, BlockDocument). Only the Anthropic Messages API
// format encodes them today; the OpenAI Chat Completions and Responses adapters
// have no content-block shape for them and reject a request that includes one
// with CodeUnsupportedContent. A caller that needs to read an image or a
// document picks a format for which this returns true, or handles the error.
func SupportsVision(f Format) bool {
	return f == FormatAnthropic
}

// hasVisionBlock reports whether any message carries an image or document block,
// so an adapter that cannot encode one rejects the request instead of dropping
// the block silently.
func hasVisionBlock(msgs []Message) bool {
	for _, m := range msgs {
		for _, b := range m.Content {
			if b.Type == BlockImage || b.Type == BlockDocument {
				return true
			}
		}
	}
	return false
}

// errUnsupportedVision is the rejection an OpenAI-format adapter returns when a
// request carries an image or document block it has no shape for.
func errUnsupportedVision() error {
	return NewError(CodeUnsupportedContent,
		"llm: this model/endpoint cannot read image or document content; route the request to a vision-capable (Anthropic) model")
}
