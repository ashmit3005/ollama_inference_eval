// Scoring proxy: sits between Python (model.py) and Ollama.
//
// Accepts POST /score with a batch of (context, continuation) pairs,
// fans them out to Ollama using goroutines (one per continuation),
// and returns aggregated logprob scores.
//
// Architecture:
//   Python model.py  →  this proxy (:9090)  →  Ollama (:11434)
//
// The proxy's advantage over Python async:
//   - goroutines are ~2KB vs Python coroutine overhead
//   - native HTTP/2 connection multiplexing
//   - zero-GIL concurrent JSON parsing
//   - in-memory prompt cache with sync.Map (lock-free reads)

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// --- Ollama API types ---

type OllamaRequest struct {
	Model      string            `json:"model"`
	Prompt     string            `json:"prompt"`
	Stream     bool              `json:"stream"`
	Raw        bool              `json:"raw"`
	Logprobs   bool              `json:"logprobs"`
	TopLogprobs int              `json:"top_logprobs"`
	Options    map[string]any    `json:"options"`
}

type TopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
}

type LogprobEntry struct {
	TopLogprobs []TopLogprob `json:"top_logprobs"`
}

type OllamaResponse struct {
	Response string         `json:"response"`
	Logprobs []LogprobEntry `json:"logprobs"`
}

// --- Proxy API types ---

type ScoreRequest struct {
	Items []ScoreItem `json:"items"`
	Config ScoreConfig `json:"config"`
}

type ScoreItem struct {
	Index        int    `json:"index"`
	Context      string `json:"context"`
	Continuation string `json:"continuation"`
}

type ScoreConfig struct {
	Model         string `json:"model"`
	Seed          int    `json:"seed"`
	TopLogprobs   int    `json:"top_logprobs"`
	MaxScoreTokens int   `json:"max_score_tokens"`
	ScoringMode   string `json:"scoring_mode"`
}

type ScoreResult struct {
	Index    int     `json:"index"`
	Logprob  float64 `json:"logprob"`
	IsGreedy bool    `json:"is_greedy"`
	Tokens   int     `json:"tokens_scored"`
}

type ScoreResponse struct {
	Results  []ScoreResult `json:"results"`
	Elapsed  float64       `json:"elapsed_sec"`
}

// --- Global state ---

var (
	ollamaURL  string
	httpClient *http.Client
	cache      sync.Map // key: string → value: *OllamaResponse
)

const logprobFloor = -100.0

func cacheKey(model string, seed int, prompt string) string {
	return fmt.Sprintf("%s|%d|%s", model, seed, prompt)
}

func generateOne(model string, seed int, topK int, prompt string) (*OllamaResponse, error) {
	key := cacheKey(model, seed, prompt)
	if v, ok := cache.Load(key); ok {
		return v.(*OllamaResponse), nil
	}

	reqBody := OllamaRequest{
		Model:       model,
		Prompt:      prompt,
		Stream:      false,
		Raw:         true,
		Logprobs:    true,
		TopLogprobs: topK,
		Options: map[string]any{
			"temperature": 0,
			"top_p":       1.0,
			"num_predict": 1,
			"seed":        seed,
		},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	resp, err := httpClient.Post(ollamaURL+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var result OllamaResponse
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}

	cache.Store(key, &result)
	return &result, nil
}

func scoreContinuation(cfg ScoreConfig, context, continuation string) ScoreResult {
	if strings.TrimSpace(continuation) == "" {
		return ScoreResult{Logprob: 0, IsGreedy: true}
	}

	totalLogprob := 0.0
	isGreedy := true
	remaining := continuation
	currentPrompt := context
	tokensScored := 0
	useSoftFloor := cfg.ScoringMode == "soft_floor"

	for len(remaining) > 0 && tokensScored < cfg.MaxScoreTokens {
		data, err := generateOne(cfg.Model, cfg.Seed, cfg.TopLogprobs, currentPrompt)
		if err != nil {
			totalLogprob += logprobFloor
			isGreedy = false
			break
		}

		if len(data.Logprobs) == 0 {
			totalLogprob += logprobFloor
			isGreedy = false
			break
		}

		topProbs := data.Logprobs[0].TopLogprobs

		minLP := math.MaxFloat64
		for _, tp := range topProbs {
			if tp.Logprob < minLP {
				minLP = tp.Logprob
			}
		}
		if minLP == math.MaxFloat64 {
			minLP = logprobFloor
		}
		softFloor := minLP - 1.0

		bestToken := ""
		bestLogprob := softFloor
		for _, tp := range topProbs {
			if len(tp.Token) > 0 && strings.HasPrefix(remaining, tp.Token) {
				if bestToken == "" || len(tp.Token) > len(bestToken) {
					bestToken = tp.Token
					bestLogprob = tp.Logprob
				}
			}
		}

		if bestToken == "" {
			if useSoftFloor {
				advance := string([]rune(remaining)[0])
				totalLogprob += softFloor
				isGreedy = false
				currentPrompt += advance
				remaining = remaining[len(advance):]
				tokensScored++
				continue
			} else {
				totalLogprob += logprobFloor
				isGreedy = false
				break
			}
		}

		if len(topProbs) > 0 && topProbs[0].Token != bestToken {
			isGreedy = false
		}

		totalLogprob += bestLogprob
		currentPrompt += bestToken
		remaining = remaining[len(bestToken):]
		tokensScored++
	}

	return ScoreResult{
		Logprob:  totalLogprob,
		IsGreedy: isGreedy,
		Tokens:   tokensScored,
	}
}

func handleScore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req ScoreRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Config.MaxScoreTokens == 0 {
		req.Config.MaxScoreTokens = 50
	}
	if req.Config.TopLogprobs == 0 {
		req.Config.TopLogprobs = 20
	}
	if req.Config.ScoringMode == "" {
		req.Config.ScoringMode = "soft_floor"
	}

	start := time.Now()

	// Group items by context for parallel fan-out
	type contextGroup struct {
		context string
		items   []ScoreItem
	}
	groups := make(map[string]*contextGroup)
	var groupOrder []string
	for _, item := range req.Items {
		if _, exists := groups[item.Context]; !exists {
			groups[item.Context] = &contextGroup{context: item.Context}
			groupOrder = append(groupOrder, item.Context)
		}
		groups[item.Context].items = append(groups[item.Context].items, item)
	}

	resultsCh := make(chan ScoreResult, len(req.Items))
	var wg sync.WaitGroup

	// Process each context group: all continuations within a group
	// run concurrently via goroutines.
	for _, ctx := range groupOrder {
		group := groups[ctx]
		for _, item := range group.items {
			wg.Add(1)
			go func(it ScoreItem) {
				defer wg.Done()
				sr := scoreContinuation(req.Config, it.Context, it.Continuation)
				sr.Index = it.Index
				resultsCh <- sr
			}(item)
		}
		// Wait for this group to finish before starting the next,
		// so Ollama's KV cache stays warm on the shared context prefix.
		wg.Wait()
	}
	close(resultsCh)

	resultsMap := make(map[int]ScoreResult)
	for sr := range resultsCh {
		resultsMap[sr.Index] = sr
	}

	results := make([]ScoreResult, len(req.Items))
	for i, item := range req.Items {
		results[i] = resultsMap[item.Index]
	}

	elapsed := time.Since(start).Seconds()

	resp := ScoreResponse{Results: results, Elapsed: elapsed}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func main() {
	ollamaURL = os.Getenv("OLLAMA_URL")
	if ollamaURL == "" {
		ollamaURL = "http://localhost:11434"
	}

	port := os.Getenv("SCORER_PORT")
	if port == "" {
		port = "9090"
	}

	httpClient = &http.Client{
		Timeout: 120 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        64,
			MaxIdleConnsPerHost: 64,
			MaxConnsPerHost:     64,
			IdleConnTimeout:     90 * time.Second,
		},
	}

	http.HandleFunc("/score", handleScore)
	http.HandleFunc("/health", handleHealth)

	log.Printf("Go scoring proxy listening on :%s → %s", port, ollamaURL)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
