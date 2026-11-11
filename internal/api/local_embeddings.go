package api

import (
	"context"
	"errors"
	"net/http"
	"net/url"

	"github.com/danielgtaylor/huma/v2"
)

const localEmbeddingsEndpointPath = "/api/v1/embeddings/local"

// LocalEmbeddingsController uses the daemon's existing vector runtime and CLI builder.
type LocalEmbeddingsController interface {
	Inspect(ctx context.Context) (LocalEmbeddingsStatus, error)
	Enable(ctx context.Context, report func(LocalEmbeddingsStatus)) error
}

type LocalEmbeddingsStatus struct {
	Host          string `json:"host"`
	Enabled       bool   `json:"enabled"`
	Provider      string `json:"provider"`
	DownloadBytes int64  `json:"download_bytes"`
	Phase         string `json:"phase" enum:"idle,downloading,indexing"`
	DoneBytes     int64  `json:"done_bytes"`
	TotalBytes    int64  `json:"total_bytes"`
	Error         string `json:"error,omitempty"`
	Running       bool   `json:"running,omitempty"`
}

func (s *Server) SetLocalEmbeddings(controller LocalEmbeddingsController) {
	s.importMu.Lock()
	s.localEmbeddings = controller
	s.importMu.Unlock()
}

func (s *Server) registerLocalEmbeddingsRoutes(api huma.API) {
	get := rawAPIV1Operation("getLocalEmbeddings", http.MethodGet, "/embeddings/local", "Get local search setup")
	get.Responses = jsonResponsesFor[LocalEmbeddingsStatus](api)
	addErrorResponses(api, get.Responses, http.StatusUnauthorized, http.StatusServiceUnavailable)
	registerRawHumaRoute(api, get, s.handleGetLocalEmbeddings)
	post := rawAPIV1Operation("enableLocalEmbeddings", http.MethodPost, "/embeddings/local", "Turn on local search")
	post.Responses = jsonResponsesFor[LocalEmbeddingsStatus](api, http.StatusAccepted)
	addErrorResponses(api, post.Responses, http.StatusUnauthorized, http.StatusServiceUnavailable)
	registerRawHumaRoute(api, post, s.handleEnableLocalEmbeddings)
}

func (s *Server) handleGetLocalEmbeddings(w http.ResponseWriter, r *http.Request) {
	s.importMu.Lock()
	controller, progress := s.localEmbeddings, s.localSetup
	running := progress.Running
	s.importMu.Unlock()
	if controller == nil {
		writeError(w, http.StatusServiceUnavailable, "unavailable", "Local search setup is unavailable")
		return
	}
	status, err := controller.Inspect(r.Context())
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, "unavailable", err.Error())
		return
	}

	if running || progress.Error != "" {
		status.Error = progress.Error
	}
	status.Running = running
	if running {
		status.Phase, status.DoneBytes, status.TotalBytes = progress.Phase, progress.DoneBytes, progress.TotalBytes
	}
	if isLoopbackAuthority(r.Host, 0) {
		status.Host = ""
	} else {
		status.Host = (&url.URL{Host: r.Host}).Hostname()
	}
	writeJSON(w, http.StatusOK, status)
}

func (s *Server) handleEnableLocalEmbeddings(w http.ResponseWriter, r *http.Request) {
	status, err := s.StartLocalEmbeddings(nil)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, "unavailable", err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, status)
}

// StartLocalEmbeddings starts a setup job shared by startup and the API.
// A nil enable function uses the registered controller.
func (s *Server) StartLocalEmbeddings(enable func(context.Context, func(LocalEmbeddingsStatus)) error) (LocalEmbeddingsStatus, error) {
	s.importMu.Lock()
	if s.localEmbeddings == nil || s.importsClosed {
		s.importMu.Unlock()
		return LocalEmbeddingsStatus{}, errors.New("Local search setup is unavailable")
	}
	if s.localSetup.Running {
		status := s.localSetup
		s.importMu.Unlock()
		return status, nil
	}
	finishIdle, ok := s.idleTracker.BeginWork()
	if !ok {
		s.importMu.Unlock()
		return LocalEmbeddingsStatus{}, errors.New("The daemon is stopping")
	}
	if enable == nil {
		enable = s.localEmbeddings.Enable
	}
	s.localSetup = LocalEmbeddingsStatus{Phase: "idle", Running: true}
	status := s.localSetup
	s.importWG.Add(1)
	s.importMu.Unlock()
	go func() {
		defer s.importWG.Done()
		defer finishIdle()
		report := func(status LocalEmbeddingsStatus) {
			s.importMu.Lock()
			status.Running = true
			s.localSetup = status
			s.importMu.Unlock()
		}
		err := enable(s.importContext, report)
		s.importMu.Lock()
		s.localSetup.Running = false
		s.localSetup.Phase = "idle"
		if err != nil {
			s.localSetup.Error = err.Error()
		}
		s.importMu.Unlock()
	}()
	return status, nil
}
