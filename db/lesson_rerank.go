package db

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// Semantic re-ranking for RSI lessons.
//
// The base selection (TaskLessons) orders by scope + evidence support + recency.
// That is correct but not relevance-aware: in a category with many lessons the
// top-N by evidence are not necessarily the N most relevant to THIS task's goal.
// This layer re-ranks a widened candidate set by cosine similarity between the
// task goal and each lesson, then the caller keeps the best K.
//
// Properties, by design:
//   - OPT-IN: with ARTEX_EMBED_URL unset the functions are no-ops and the
//     existing evidence ordering is returned unchanged. Nothing new runs.
//   - FAIL-SAFE: any embedding/transport/DB error degrades to the original
//     (evidence-ordered) head of the slice. Task start is never blocked.
//   - LOCAL ONLY: lesson text (target-controlled) and the goal are sent solely
//     to the operator's local embedding endpoint for vectorization, never to a
//     remote LLM. Vectors are cached in Postgres keyed by content hash, so warm
//     runs embed only the goal (one call), not the candidates.

// LessonRerankPool is how many scope/evidence-ranked candidates the injection
// path should pull before semantic re-ranking narrows them to the inject count.
// Large enough to give the re-rank something to choose from, small enough that a
// cold cache is a bounded one-time cost.
const LessonRerankPool = 40

const (
	embedHTTPTimeout = 8 * time.Second
	// nomic-embed-text conventions; harmless prefixes for other models.
	embedQueryPrefix = "search_query: "
	embedDocPrefix   = "search_document: "
)

// goalVecCache memoizes the goal embedding across the planner's frequent
// wake-ups so a steady task re-embeds its goal at most once per (model, goal).
var (
	goalVecMu    sync.Mutex
	goalVecCache = map[string][]float64{}
)

// rerankConfig reads the opt-in embedding endpoint. ok=false disables re-ranking.
func rerankConfig() (base, model string, ok bool) {
	base = strings.TrimRight(strings.TrimSpace(os.Getenv("ARTEX_EMBED_URL")), "/")
	if base == "" {
		return "", "", false
	}
	model = strings.TrimSpace(os.Getenv("ARTEX_EMBED_MODEL"))
	if model == "" {
		model = "nomic-embed-text"
	}
	return base, model, true
}

// RerankLessonsByEmbedding reorders lessons by semantic similarity to the task
// goal and returns at most topK. It never errors: on any failure (feature off,
// no goal, endpoint down, bad vector) it returns the original head of the slice,
// so the caller always gets a usable, evidence-ordered result.
func (d *DB) RerankLessonsByEmbedding(ctx context.Context, taskID int64, lessons []Lesson, topK int) []Lesson {
	if topK <= 0 {
		topK = 8
	}
	head := func() []Lesson {
		if len(lessons) > topK {
			return lessons[:topK]
		}
		return lessons
	}
	if len(lessons) <= 1 {
		return head()
	}
	base, model, ok := rerankConfig()
	if !ok {
		return head()
	}
	goal, err := taskGoal(ctx, d, taskID)
	if err != nil || strings.TrimSpace(goal) == "" {
		return head()
	}

	gvec, err := d.goalVector(ctx, base, model, goal)
	if err != nil {
		log.Printf("[rsi] rerank disabled for task %d (goal embed failed: %v)", taskID, err)
		return head()
	}

	type scored struct {
		l   Lesson
		sim float64
		idx int // original position, for a stable tie-break toward evidence order
	}
	ranked := make([]scored, 0, len(lessons))
	var embedErrs int
	for i, l := range lessons {
		text, srcHash := lessonEmbedInput(l)
		vec, err := d.cachedLessonVector(ctx, base, model, l.ID, srcHash, text)
		if err != nil {
			embedErrs++
			// Missing vector sinks to the bottom but keeps evidence order among peers.
			ranked = append(ranked, scored{l: l, sim: math.Inf(-1), idx: i})
			continue
		}
		ranked = append(ranked, scored{l: l, sim: cosineSim(gvec, vec), idx: i})
	}
	// If every candidate failed to embed, trust the original ordering entirely.
	if embedErrs == len(lessons) {
		log.Printf("[rsi] rerank fell back for task %d (all %d candidates failed to embed)", taskID, embedErrs)
		return head()
	}
	sort.SliceStable(ranked, func(a, b int) bool {
		if ranked[a].sim != ranked[b].sim {
			return ranked[a].sim > ranked[b].sim
		}
		return ranked[a].idx < ranked[b].idx // tie -> original (evidence) order
	})
	out := make([]Lesson, 0, topK)
	for i := 0; i < len(ranked) && i < topK; i++ {
		out = append(out, ranked[i].l)
	}
	return out
}

// lessonEmbedInput builds the text embedded for a lesson and a hash of that text,
// so a cached vector is invalidated when the lesson's cue/action change.
func lessonEmbedInput(l Lesson) (text, hash string) {
	text = embedDocPrefix + l.Kind + ": " +
		lessonText(l.Cue, lessonCueMax) + " — " + lessonText(l.Action, lessonActionMax)
	sum := sha256.Sum256([]byte(text))
	return text, hex.EncodeToString(sum[:])
}

// goalVector returns the (process-cached) embedding of the task goal.
func (d *DB) goalVector(ctx context.Context, base, model, goal string) ([]float64, error) {
	key := model + "\x00" + goal
	goalVecMu.Lock()
	if v, ok := goalVecCache[key]; ok {
		goalVecMu.Unlock()
		return v, nil
	}
	goalVecMu.Unlock()

	vec, err := embedText(ctx, base, model, embedQueryPrefix+lessonText(goal, 512))
	if err != nil {
		return nil, err
	}
	goalVecMu.Lock()
	if len(goalVecCache) > 512 { // crude bound; goals are few and small
		goalVecCache = map[string][]float64{}
	}
	goalVecCache[key] = vec
	goalVecMu.Unlock()
	return vec, nil
}

// cachedLessonVector returns a lesson's embedding from rsi_lesson_embeddings when
// the cached source hash still matches, otherwise it embeds and upserts the cache.
func (d *DB) cachedLessonVector(ctx context.Context, base, model string, lessonID int64, srcHash, text string) ([]float64, error) {
	var raw []byte
	var storedHash string
	err := d.QueryRowContext(ctx,
		`SELECT vec, source_hash FROM rsi_lesson_embeddings WHERE lesson_id=$1 AND model=$2`,
		lessonID, model).Scan(&raw, &storedHash)
	if err == nil && storedHash == srcHash {
		var vec []float64
		if json.Unmarshal(raw, &vec) == nil && len(vec) > 0 {
			return vec, nil
		}
	}
	// Miss, stale, or unreadable -> (re)embed and upsert.
	vec, err := embedText(ctx, base, model, text)
	if err != nil {
		return nil, err
	}
	payload, err := json.Marshal(vec)
	if err != nil {
		return nil, err
	}
	if _, err := d.ExecContext(ctx, `
		INSERT INTO rsi_lesson_embeddings(lesson_id, model, dim, vec, source_hash, updated_at)
		VALUES ($1,$2,$3,$4,$5, now())
		ON CONFLICT (lesson_id, model)
		DO UPDATE SET dim=EXCLUDED.dim, vec=EXCLUDED.vec,
		              source_hash=EXCLUDED.source_hash, updated_at=now()`,
		lessonID, model, len(vec), payload, srcHash); err != nil {
		// Cache write failure is non-fatal: we still have a usable vector.
		log.Printf("[rsi] embedding cache write failed for lesson %d: %v", lessonID, err)
	}
	return vec, nil
}

// taskGoal fetches a task's goal text for use as the re-rank query.
func taskGoal(ctx context.Context, d *DB, taskID int64) (string, error) {
	var goal string
	err := d.QueryRowContext(ctx,
		`SELECT COALESCE(goal,'') FROM tasks WHERE id=$1 AND deleted_at IS NULL`, taskID).
		Scan(&goal)
	return goal, err
}

// cosineSim is the cosine similarity of two equal-length vectors. A dimension
// mismatch (e.g. a model change left a stale cached vector) yields -Inf so the
// candidate sinks rather than corrupting the ranking.
func cosineSim(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return math.Inf(-1)
	}
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return math.Inf(-1)
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// --- local embedding client ------------------------------------------------

var embedClient = &http.Client{Timeout: embedHTTPTimeout}

// embedText calls the local embedding endpoint. It tries the OpenAI-compatible
// shape (/v1/embeddings) first, then Ollama native (/api/embeddings), so it
// works against llama.cpp-server, Ollama, or any OpenAI-shaped local server.
func embedText(ctx context.Context, base, model, text string) ([]float64, error) {
	if vec, err := embedOpenAI(ctx, base, model, text); err == nil {
		return vec, nil
	} else if vec, err2 := embedOllama(ctx, base, model, text); err2 == nil {
		return vec, nil
	} else {
		return nil, fmt.Errorf("openai-shape: %v; ollama-shape: %v", err, err2)
	}
}

func embedOpenAI(ctx context.Context, base, model, text string) ([]float64, error) {
	body, _ := json.Marshal(map[string]any{"model": model, "input": text})
	var out struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := postJSON(ctx, base+"/v1/embeddings", body, &out); err != nil {
		return nil, err
	}
	if len(out.Data) == 0 || len(out.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("empty embedding")
	}
	return out.Data[0].Embedding, nil
}

func embedOllama(ctx context.Context, base, model, text string) ([]float64, error) {
	root := strings.TrimSuffix(base, "/v1")
	body, _ := json.Marshal(map[string]any{"model": model, "prompt": text})
	var out struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := postJSON(ctx, root+"/api/embeddings", body, &out); err != nil {
		return nil, err
	}
	if len(out.Embedding) == 0 {
		return nil, fmt.Errorf("empty embedding")
	}
	return out.Embedding, nil
}

func postJSON(ctx context.Context, url string, body []byte, dst any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := embedClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(dst)
}
