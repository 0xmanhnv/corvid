package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// decodeFirstUserContent returns the first message's first content block of an
// Anthropic request body, as a generic map, for shape assertions.
func decodeFirstUserContent(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var req struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if len(req.Messages) == 0 {
		t.Fatalf("no messages in body: %s", body)
	}
	var msg struct {
		Content []map[string]any `json:"content"`
	}
	if err := json.Unmarshal(req.Messages[0], &msg); err != nil {
		t.Fatalf("decode message: %v", err)
	}
	if len(msg.Content) == 0 {
		t.Fatalf("no content blocks: %s", req.Messages[0])
	}
	return msg.Content[0]
}

// An ImageBlock encodes to the Anthropic image content-block shape:
// {"type":"image","source":{"type":"base64","media_type":...,"data":...}}.
func TestAnthropicImageBlockWireShape(t *testing.T) {
	p := &anthropicProvider{cfg: Config{Model: "claude"}}
	body, err := p.buildBody(CompletionRequest{
		DisableCache: true, // no cache_control marker, so the block is the bare shape
		Messages:     []Message{{Role: RoleUser, Content: []ContentBlock{ImageBlock("image/png", "aW1nMTIz")}}},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	block := decodeFirstUserContent(t, body)
	if block["type"] != "image" {
		t.Fatalf("type = %v, want image", block["type"])
	}
	src, ok := block["source"].(map[string]any)
	if !ok {
		t.Fatalf("source missing or wrong type: %v", block["source"])
	}
	if src["type"] != "base64" || src["media_type"] != "image/png" || src["data"] != "aW1nMTIz" {
		t.Fatalf("source = %v", src)
	}
}

// A DocumentBlock encodes to the Anthropic document content-block shape with a
// PDF media type, so the model reads the PDF directly (no rasterizing).
func TestAnthropicDocumentBlockWireShape(t *testing.T) {
	p := &anthropicProvider{cfg: Config{Model: "claude"}}
	body, err := p.buildBody(CompletionRequest{
		DisableCache: true,
		Messages:     []Message{{Role: RoleUser, Content: []ContentBlock{DocumentBlock("application/pdf", "JVBERi0x")}}},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	block := decodeFirstUserContent(t, body)
	if block["type"] != "document" {
		t.Fatalf("type = %v, want document", block["type"])
	}
	src, ok := block["source"].(map[string]any)
	if !ok {
		t.Fatalf("source missing: %v", block["source"])
	}
	if src["type"] != "base64" || src["media_type"] != "application/pdf" || src["data"] != "JVBERi0x" {
		t.Fatalf("source = %v", src)
	}
}

// Regression guard: adding the Source field must not change the text path. A
// text block marshals to exactly its old bytes (no "source" key), and a
// text/tool-only request body carries no "source" anywhere.
func TestTextPathUnchangedBySourceField(t *testing.T) {
	if b, _ := json.Marshal(TextBlock("hi")); string(b) != `{"type":"text","text":"hi"}` {
		t.Fatalf("text block marshaled to %s", b)
	}
	// A tool_use block likewise carries no source key.
	tu := ContentBlock{Type: BlockToolUse, ID: "t1", Name: "do", Input: json.RawMessage(`{"x":1}`)}
	if b, _ := json.Marshal(tu); strings.Contains(string(b), "source") {
		t.Fatalf("tool_use block gained a source key: %s", b)
	}
	p := &anthropicProvider{cfg: Config{Model: "claude"}}
	body, err := p.buildBody(CompletionRequest{
		System:   []string{"sys"},
		Messages: []Message{usr(TextBlock("hello")), asst(TextBlock("hi"), tu)},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `"source"`) {
		t.Fatalf("text/tool request body contains a source key:\n%s", body)
	}
}

// Capability gate: an OpenAI-format provider rejects an image or document block
// with CodeUnsupportedContent (clear, surfaced) rather than panicking or
// dropping it. SupportsVision agrees.
func TestOpenAIRejectsVisionBlocks(t *testing.T) {
	if SupportsVision(FormatOpenAI) || SupportsVision(FormatOpenAIResponses) {
		t.Fatal("SupportsVision should be false for the OpenAI formats")
	}
	if !SupportsVision(FormatAnthropic) {
		t.Fatal("SupportsVision should be true for Anthropic")
	}
	cases := []CompletionRequest{
		{Messages: []Message{{Role: RoleUser, Content: []ContentBlock{ImageBlock("image/png", "aa")}}}},
		{Messages: []Message{{Role: RoleUser, Content: []ContentBlock{DocumentBlock("application/pdf", "bb")}}}},
	}
	for _, req := range cases {
		op := &openaiProvider{cfg: Config{Model: "gpt"}}
		if _, err := op.buildBody(req, false); !IsCode(err, CodeUnsupportedContent) {
			t.Fatalf("openai buildBody err = %v, want CodeUnsupportedContent", err)
		}
		orp := &openaiResponsesProvider{cfg: Config{Model: "gpt"}}
		if _, err := orp.buildBody(req, false); !IsCode(err, CodeUnsupportedContent) {
			t.Fatalf("responses buildBody err = %v, want CodeUnsupportedContent", err)
		}
	}
}
