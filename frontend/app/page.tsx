'use client'

import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import DetectionCanvas, { colourFor, type Detection } from '@/components/DetectionCanvas'

const BACKEND_URL = 'http://localhost:8000'

type PageView      = 'detection' | 'benchmark'
type DetectionMode = 'video' | 'image'

type HistoryEntry = {
  id:       number
  mode:     DetectionMode
  model:    string
  backend:  string
  fileName: string
  latency:  number
  // image
  imageUrl?:   string
  detections?: Detection[]
  // video
  videoUrl?:     string
  videoResults?: VideoDetectResponse
}

type DetectResponse = {
  model:      string
  backend:    string
  latency_ms: number
  detections: Detection[]
}

type FrameResult = {
  frame_index: number
  latency_ms:  number
  detections:  Detection[]
}

type VideoDetectResponse = {
  model:            string
  backend:          string
  fps:              number
  total_frames:     number
  processed_frames: number
  avg_latency_ms:   number
  frame_results:    FrameResult[]
}

type BenchmarkEntry = {
  model:          string
  backend:        string
  avg_latency_ms: number | null
  map50:          number | null
  map50_95:       number | null
}

const BACKEND_LABELS: Record<string, string> = {
  'pytorch-cpu': 'PyTorch (CPU)',
  pytorch:       'PyTorch (MPS)',
  torchscript:   'TorchScript',
  openvino:      'OpenVINO',
  coreml:        'ONNX + CoreML EP',
}

const BACKEND_ORDER = ['pytorch-cpu', 'pytorch', 'torchscript', 'openvino', 'coreml']

// Derived dynamically from /models API response — see buildCombos() in BenchmarkPanel

export default function Page() {
  const [pageView,       setPageView]       = useState<PageView>('detection')
  const [mode,           setMode]           = useState<DetectionMode>('video')

  // Shared controls
  const [modelName,      setModelName]      = useState('yolov8s')
  const [backend,        setBackend]        = useState('pytorch-cpu')
  const [loading,        setLoading]        = useState(false)
  const [error,          setError]          = useState<string>('')
  const [modelBackends,  setModelBackends]  = useState<Record<string, string[]>>({})
  const [displayLatency, setDisplayLatency] = useState<number | null>(null)
  const [fileName,       setFileName]       = useState<string>('')

  // Image state
  const [imageUrl,     setImageUrl]     = useState<string>('')
  const [detections,   setDetections]   = useState<Detection[]>([])
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)

  // Video state
  const [videoUrl,          setVideoUrl]          = useState<string>('')
  const [videoResults,      setVideoResults]      = useState<VideoDetectResponse | null>(null)
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0)
  const [hoveredVideoClass, setHoveredVideoClass] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)

  // Benchmark state
  const [benchmarkEntries, setBenchmarkEntries] = useState<BenchmarkEntry[]>([])

  // History of past detection runs — session only
  const [history, setHistory] = useState<HistoryEntry[]>([])

  function pushHistory(entry: Omit<HistoryEntry, 'id'>) {
    setHistory(prev => [{ ...entry, id: Date.now() }, ...prev].slice(0, 20))
  }

  function restoreHistory(entry: HistoryEntry) {
    setMode(entry.mode)
    setModelName(entry.model)
    setBackend(entry.backend)
    setFileName(entry.fileName)
    setDisplayLatency(entry.latency)
    if (entry.mode === 'image') {
      setImageUrl(entry.imageUrl!)
      setDetections(entry.detections!)
    } else {
      setVideoUrl(entry.videoUrl!)
      setVideoResults(entry.videoResults!)
    }
  }

  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetch(`${BACKEND_URL}/models`)
      .then(r => r.json())
      .then(data => setModelBackends(data.models ?? {}))
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (pageView !== 'benchmark') return
    fetch(`${BACKEND_URL}/benchmark`)
      .then(r => r.json())
      .then(data => setBenchmarkEntries(data.entries ?? []))
      .catch(() => {})
  }, [pageView])

  function handleModelChange(newModel: string) {
    setModelName(newModel)
    const supported = modelBackends[newModel] ?? []
    if (!supported.includes(backend)) setBackend('pytorch-cpu')
  }

  function handleModeChange(newMode: DetectionMode) {
    setMode(newMode)
    setError('')
    setFileName('')
    setDisplayLatency(null)
    if (fileRef.current) fileRef.current.value = ''
  }

  const sortedBackends = (modelBackends[modelName] ?? Object.keys(BACKEND_LABELS))
    .slice()
    .sort((a, b) => {
      const ai = BACKEND_ORDER.indexOf(a)
      const bi = BACKEND_ORDER.indexOf(b)
      return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi)
    })

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    setFileName(f.name)
    setError('')
    setDisplayLatency(null)
    if (mode === 'image') {
      setImageUrl(URL.createObjectURL(f))
      setDetections([])
    } else {
      setVideoUrl(URL.createObjectURL(f))
      setVideoResults(null)
    }
  }

  async function handleImageSubmit(e: React.SyntheticEvent<HTMLFormElement>) {
    e.preventDefault()
    const f = fileRef.current?.files?.[0]
    if (!f) return
    setLoading(true)
    setError('')
    try {
      const form = new FormData()
      form.append('file', f)
      const res = await fetch(
        `${BACKEND_URL}/detect/image?model_name=${modelName}&backend=${backend}`,
        { method: 'POST', body: form }
      )
      if (!res.ok) throw new Error((await res.json()).detail ?? 'Request failed')
      const data: DetectResponse = await res.json()
      setDetections(data.detections)
      setDisplayLatency(data.latency_ms)
      pushHistory({ mode: 'image', model: modelName, backend, fileName, latency: data.latency_ms, imageUrl, detections: data.detections })
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  async function handleVideoSubmit(e: React.SyntheticEvent<HTMLFormElement>) {
    e.preventDefault()
    const f = fileRef.current?.files?.[0]
    if (!f) return
    setLoading(true)
    setError('')
    try {
      const form = new FormData()
      form.append('file', f)
      const res = await fetch(
        `${BACKEND_URL}/detect/video?model_name=${modelName}&backend=${backend}&frame_interval=5`,
        { method: 'POST', body: form }
      )
      if (!res.ok) throw new Error((await res.json()).detail ?? 'Request failed')
      const data: VideoDetectResponse = await res.json()
      setVideoResults(data)
      setDisplayLatency(data.avg_latency_ms)
      pushHistory({ mode: 'video', model: modelName, backend, fileName, latency: data.avg_latency_ms, videoUrl, videoResults: data })
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  // Aggregate per-class detection counts across all video frames
  const videoClassSummary = useMemo(() => {
    if (!videoResults) return []
    const counts: Record<string, { count: number; totalConf: number }> = {}
    for (const frame of videoResults.frame_results) {
      for (const det of frame.detections) {
        if (!counts[det.class]) counts[det.class] = { count: 0, totalConf: 0 }
        counts[det.class].count++
        counts[det.class].totalConf += det.confidence
      }
    }
    return Object.entries(counts)
      .map(([cls, { count, totalConf }]) => ({ class: cls, count, avgConf: totalConf / count }))
      .sort((a, b) => b.count - a.count)
  }, [videoResults])

  // Classes present in the nearest processed frame to the current playback position
  const currentFrameClasses = useMemo(() => {
    if (!videoResults || videoResults.frame_results.length === 0) return new Set<string>()
    const sorted = [...videoResults.frame_results].sort((a, b) => a.frame_index - b.frame_index)
    let nearest = sorted[0]
    for (const fr of sorted) {
      if (Math.abs(fr.frame_index - currentFrameIndex) < Math.abs(nearest.frame_index - currentFrameIndex))
        nearest = fr
    }
    return new Set(nearest.detections.map(d => d.class))
  }, [videoResults, currentFrameIndex])

  const presentClasses = useMemo(
    () => videoClassSummary.filter(item =>  currentFrameClasses.has(item.class)),
    [videoClassSummary, currentFrameClasses]
  )
  const absentClasses = useMemo(
    () => videoClassSummary.filter(item => !currentFrameClasses.has(item.class)),
    [videoClassSummary, currentFrameClasses]
  )

  function seekToClass(className: string) {
    if (!videoRef.current || !videoResults) return
    const first = videoResults.frame_results.find(fr => fr.detections.some(d => d.class === className))
    if (first) videoRef.current.currentTime = first.frame_index / (videoResults.fps || 30)
  }

  const hasFile = !!fileName

  return (
    <div className="flex flex-col h-screen bg-black overflow-hidden" style={{ fontFamily: 'var(--font-montserrat), sans-serif' }}>

      {/* Header */}
      <header className="flex items-center justify-between px-8 py-4 border-b border-white/10">
        <h1 className="text-xl tracking-[0.2em] uppercase text-white">Object Detection</h1>
        <div className="flex items-center gap-6">
          <nav className="flex gap-1">
            {(['detection', 'benchmark'] as PageView[]).map(v => (
              <button
                key={v}
                onClick={() => setPageView(v)}
                className={`px-4 py-1.5 text-xs tracking-widest uppercase rounded-sm transition-colors ${
                  pageView === v ? 'bg-white text-black' : 'text-white/60 hover:text-white/90'
                }`}
              >
                {v}
              </button>
            ))}
          </nav>
          {displayLatency !== null && pageView === 'detection' && (
            <div className="flex items-center gap-3">
              <span className="text-white/60 tracking-widest uppercase text-xs">
                {mode === 'video' ? 'Avg Latency' : 'Latency'}
              </span>
              <span className="border border-white/30 text-white font-mono px-3 py-1 rounded-sm text-sm">
                {displayLatency.toFixed(1)} ms
              </span>
            </div>
          )}
        </div>
      </header>

      {pageView === 'benchmark' ? (
        <BenchmarkPanel
          entries={benchmarkEntries}
          onEntriesChange={setBenchmarkEntries}
          modelBackends={modelBackends}
        />
      ) : (
        <div className="flex flex-1 overflow-hidden">

          {/* Left panel */}
          <aside className="w-72 flex-shrink-0 border-r border-white/10 p-6 flex flex-col gap-6 overflow-y-auto">
            <div className="flex border border-white/20 rounded-sm overflow-hidden">
              {(['video', 'image'] as DetectionMode[]).map(m => (
                <button
                  key={m}
                  onClick={() => handleModeChange(m)}
                  className={`flex-1 py-2 text-xs tracking-widest uppercase transition-colors ${
                    mode === m ? 'bg-white text-black' : 'text-white/60 hover:text-white/90'
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>

            <form
              onSubmit={mode === 'image' ? handleImageSubmit : handleVideoSubmit}
              className="flex flex-col gap-5"
            >
              <div className="flex flex-col gap-2">
                <label className="text-xs tracking-widest uppercase text-white/60">
                  {mode === 'image' ? 'Image' : 'Video'}
                </label>
                <label className="flex items-center justify-center border border-white/20 hover:border-white/50
                                  rounded-sm py-3 px-4 cursor-pointer transition-colors text-sm text-white/75
                                  hover:text-white/95">
                  <span className="truncate max-w-full">{fileName || `Choose ${mode}`}</span>
                  <input
                    ref={fileRef}
                    type="file"
                    accept={mode === 'image' ? 'image/*' : 'video/*'}
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
              </div>

              <div className="flex flex-col gap-2">
                <label className="text-xs tracking-widest uppercase text-white/60">Model</label>
                <select
                  value={modelName}
                  onChange={e => handleModelChange(e.target.value)}
                  className="bg-black border border-white/20 hover:border-white/50 rounded-sm
                             px-3 py-2 text-sm text-white transition-colors outline-none"
                >
                  <option value="yolov8s">YOLOv8s</option>
                  <option value="rtdetr-l">RT-DETR-l</option>
                </select>
              </div>

              <div className="flex flex-col gap-2">
                <label className="text-xs tracking-widest uppercase text-white/60">Backend</label>
                <select
                  value={backend}
                  onChange={e => setBackend(e.target.value)}
                  className="bg-black border border-white/20 hover:border-white/50 rounded-sm
                             px-3 py-2 text-sm text-white transition-colors outline-none"
                >
                  {sortedBackends.map(b => (
                    <option key={b} value={b}>{BACKEND_LABELS[b] ?? b}</option>
                  ))}
                </select>
              </div>

              <button
                type="submit"
                disabled={!hasFile || loading}
                className="mt-1 bg-white text-black hover:bg-white/80 disabled:bg-white/10
                           disabled:text-white/20 rounded-sm py-2 px-4 text-sm tracking-widest
                           uppercase transition-colors"
              >
                {loading ? (mode === 'video' ? 'Processing...' : 'Running...') : 'Detect'}
              </button>

              {error && <p className="text-red-400 text-xs leading-relaxed">{error}</p>}
            </form>

            {/* History */}
            {history.length > 0 && (
              <div className="pt-4 border-t border-white/10">
                <h3 className="text-xs tracking-widest uppercase text-white/55 mb-3">History</h3>
                <ul className="flex flex-col gap-1">
                  {history.map(entry => (
                    <li
                      key={entry.id}
                      onClick={() => restoreHistory(entry)}
                      className="flex flex-col gap-0.5 px-3 py-2 rounded-sm cursor-pointer
                                 hover:bg-white/5 transition-colors"
                    >
                      <span className="text-xs text-white/70 truncate">
                        {entry.model} · {BACKEND_LABELS[entry.backend] ?? entry.backend}
                      </span>
                      <span className="text-[10px] text-white/50 font-mono">
                        {entry.mode} · {entry.latency.toFixed(1)} ms · {entry.fileName}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </aside>

          {/* Center panel */}
          <main className="flex-1 overflow-hidden flex items-center justify-center p-6">
            {mode === 'image' ? (
              imageUrl ? (
                <div className="flex items-center justify-center h-full w-full">
                  <DetectionCanvas imageUrl={imageUrl} detections={detections} hoveredIndex={hoveredIndex} />
                </div>
              ) : (
                <EmptyPlaceholder icon="image" label="Upload an image to begin" />
              )
            ) : (
              videoUrl ? (
                <VideoDetectionPlayer
                  videoRef={videoRef}
                  videoUrl={videoUrl}
                  frameResults={videoResults?.frame_results ?? []}
                  fps={videoResults?.fps ?? 30}
                  hoveredClass={hoveredVideoClass}
                  onFrameChange={setCurrentFrameIndex}
                />
              ) : (
                <EmptyPlaceholder icon="video" label="Upload a video to begin" />
              )
            )}
          </main>

          {/* Right panel */}
          <aside className="w-64 flex-shrink-0 border-l border-white/10 p-6 overflow-y-auto">
            {mode === 'image' ? (
              <>
                <h2 className="text-xs tracking-widest uppercase text-white/60 mb-4">
                  Detections {detections.length > 0 && `(${detections.length})`}
                </h2>
                {detections.length === 0 ? (
                  <p className="text-white/50 text-xs">None yet</p>
                ) : (
                  <ul className="flex flex-col gap-1">
                    {detections.map((d, i) => (
                      <li
                        key={i}
                        onMouseEnter={() => setHoveredIndex(i)}
                        onMouseLeave={() => setHoveredIndex(null)}
                        className="flex items-center justify-between gap-3 px-3 py-2 rounded-sm
                                   cursor-default transition-colors hover:bg-white/5"
                      >
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: colourFor(d.class) }} />
                        <span className="flex-1 text-sm text-white/80 truncate">{d.class}</span>
                        <span className="text-xs font-mono text-white/65 flex-shrink-0">
                          {(d.confidence * 100).toFixed(0)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            ) : (
              <>
                <h2 className="text-xs tracking-widest uppercase text-white/60 mb-4">
                  Classes {videoClassSummary.length > 0 && `(${videoClassSummary.length})`}
                </h2>
                {videoResults && (
                  <div className="mb-4 pb-4 border-b border-white/10 flex flex-col gap-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-white/55 uppercase tracking-widest">Avg latency</span>
                      <span className="text-white font-mono">{videoResults.avg_latency_ms.toFixed(1)} ms</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-white/55 uppercase tracking-widest">Frames</span>
                      <span className="text-white font-mono">{videoResults.processed_frames}/{videoResults.total_frames}</span>
                    </div>
                  </div>
                )}
                {videoClassSummary.length === 0 ? (
                  <p className="text-white/50 text-xs">None yet</p>
                ) : (
                  <ul className="flex flex-col gap-0.5">
                    {presentClasses.map(item => (
                      <li
                        key={item.class}
                        onMouseEnter={() => setHoveredVideoClass(item.class)}
                        onMouseLeave={() => setHoveredVideoClass(null)}
                        onClick={() => seekToClass(item.class)}
                        className="flex items-center justify-between gap-3 px-3 py-2 rounded-sm
                                   cursor-pointer hover:bg-white/5 transition-colors"
                      >
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: colourFor(item.class) }} />
                        <span className="flex-1 text-sm text-white/90 truncate">{item.class}</span>
                        <span className="text-xs font-mono text-white/65 flex-shrink-0">{item.count}×</span>
                      </li>
                    ))}
                    {presentClasses.length > 0 && absentClasses.length > 0 && (
                      <li className="py-1"><div className="border-t border-white/10" /></li>
                    )}
                    {absentClasses.map(item => (
                      <li
                        key={item.class}
                        onClick={() => seekToClass(item.class)}
                        className="flex items-center justify-between gap-3 px-3 py-2 rounded-sm
                                   cursor-pointer hover:bg-white/5 transition-colors opacity-35"
                      >
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: colourFor(item.class) }} />
                        <span className="flex-1 text-sm text-white/80 truncate">{item.class}</span>
                        <span className="text-xs font-mono text-white/65 flex-shrink-0">{item.count}×</span>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            )}
          </aside>

        </div>
      )}
    </div>
  )
}

// ----- Sub-components -----

function EmptyPlaceholder({ icon, label }: { icon: 'image' | 'video'; label: string }) {
  return (
    <div className="flex flex-col items-center gap-3 text-white/40 select-none">
      {icon === 'image' ? (
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <circle cx="8.5" cy="8.5" r="1.5"/>
          <polyline points="21 15 16 10 5 21"/>
        </svg>
      ) : (
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
          <rect x="2" y="7" width="15" height="10" rx="2"/>
          <polyline points="17 9 22 5 22 19 17 15"/>
        </svg>
      )}
      <span className="text-xs tracking-widest uppercase">{label}</span>
    </div>
  )
}

function VideoDetectionPlayer({
  videoRef,
  videoUrl,
  frameResults,
  fps,
  hoveredClass,
  onFrameChange,
}: {
  videoRef:      React.RefObject<HTMLVideoElement | null>
  videoUrl:      string
  frameResults:  FrameResult[]
  fps:           number
  hoveredClass:  string | null
  onFrameChange: (idx: number) => void
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Sort once so binary search is valid
  const sorted = useMemo(
    () => [...frameResults].sort((a, b) => a.frame_index - b.frame_index),
    [frameResults]
  )

  const draw = useCallback(() => {
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.videoWidth === 0) return

    canvas.width  = video.clientWidth
    canvas.height = video.clientHeight

    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (sorted.length === 0) return

    // Find the nearest processed frame to the current playback position
    const currentFrame = Math.round(video.currentTime * fps)
    onFrameChange(currentFrame)

    let nearest = sorted[0]
    for (const fr of sorted) {
      if (Math.abs(fr.frame_index - currentFrame) < Math.abs(nearest.frame_index - currentFrame)) {
        nearest = fr
      }
    }

    const scaleX = video.clientWidth  / video.videoWidth
    const scaleY = video.clientHeight / video.videoHeight

    nearest.detections.forEach(det => {
      const dimmed = hoveredClass !== null && det.class !== hoveredClass
      const [x1, y1, x2, y2] = det.box
      const sx1 = x1 * scaleX
      const sy1 = y1 * scaleY
      const sw  = (x2 - x1) * scaleX
      const sh  = (y2 - y1) * scaleY

      const colour = colourFor(det.class)
      ctx.globalAlpha = dimmed ? 0.15 : 1.0
      ctx.lineWidth   = 1.5
      ctx.strokeStyle = colour
      ctx.strokeRect(sx1, sy1, sw, sh)

      const label = `${det.class}  ${(det.confidence * 100).toFixed(0)}%`
      ctx.font      = '12px monospace'
      const textW   = ctx.measureText(label).width
      ctx.fillStyle = colour
      ctx.fillRect(sx1, sy1 - 22, textW + 10, 22)
      ctx.globalAlpha = dimmed ? 0.15 : 1.0
      ctx.fillStyle = '#000000'
      ctx.fillText(label, sx1 + 5, sy1 - 6)
    })
  }, [videoRef, sorted, fps, hoveredClass, onFrameChange])

  // Redraw immediately when detection results change (video may be paused — no timeupdate fires)
  useEffect(() => { draw() }, [draw])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    video.addEventListener('timeupdate',     draw)
    video.addEventListener('seeked',         draw)
    video.addEventListener('loadedmetadata', draw)
    return () => {
      video.removeEventListener('timeupdate',     draw)
      video.removeEventListener('seeked',         draw)
      video.removeEventListener('loadedmetadata', draw)
    }
  }, [draw, videoRef])

  return (
    <div
      className="relative"
      style={{ width: '100%', maxWidth: 'min(100vw - 560px, 960px)', lineHeight: 0 }}
    >
      <video
        ref={videoRef}
        src={videoUrl}
        controls
        className="block rounded-sm w-full"
        style={{ maxHeight: 'calc(100vh - 160px)', height: 'auto' }}
      />
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
    </div>
  )
}

function BenchmarkPanel({
  entries,
  onEntriesChange,
  modelBackends,
}: {
  entries:         BenchmarkEntry[]
  onEntriesChange: (e: BenchmarkEntry[]) => void
  modelBackends:   Record<string, string[]>
}) {
  // Build ordered combos dynamically from whatever backends the server reports
  const allCombos = useMemo(() =>
    Object.keys(modelBackends).flatMap(model =>
      BACKEND_ORDER
        .filter(b => modelBackends[model].includes(b))
        .map(b => ({ model, backend: b }))
    ),
    [modelBackends]
  )
  const models = useMemo(() => Object.keys(modelBackends), [modelBackends])
  const [benchFile,    setBenchFile]    = useState<File | null>(null)
  const [running,      setRunning]      = useState(false)
  const [progress,     setProgress]     = useState('')
  const [doneCount,    setDoneCount]    = useState(0)
  const [evalRunning,  setEvalRunning]  = useState(false)
  const [evalStatus,   setEvalStatus]   = useState('')
  const benchFileRef = useRef<HTMLInputElement>(null)

  async function runEval() {
    setEvalRunning(true)
    setEvalStatus('Running mAP evaluation — this may take several minutes...')
    try {
      const res = await fetch(`${BACKEND_URL}/eval/run`, { method: 'POST' })
      const data = await res.json()
      if (!res.ok) {
        setEvalStatus(`Error: ${data.error ?? 'Unknown error'}`)
        return
      }
      onEntriesChange(data.entries ?? [])
      setLive(data.entries ?? [])
      setEvalStatus('Evaluation complete. mAP columns updated.')
    } catch {
      setEvalStatus('Failed to reach backend.')
    } finally {
      setEvalRunning(false)
    }
  }

  // Work on a local copy so the table updates live during the run
  const [live, setLive] = useState<BenchmarkEntry[]>(entries)
  useEffect(() => { setLive(entries) }, [entries])

  function updateRow(model: string, backend: string, latency: number) {
    setLive(prev => {
      const next = prev.map(e =>
        e.model === model && e.backend === backend
          ? { ...e, avg_latency_ms: latency }
          : e
      )
      // Add row if it wasn't in the original list
      if (!next.find(e => e.model === model && e.backend === backend)) {
        next.push({ model, backend, avg_latency_ms: latency, map50: null, map50_95: null })
      }
      return next
    })
  }

  async function runAll() {
    if (!benchFile) return
    setRunning(true)
    setDoneCount(0)

    const updated: BenchmarkEntry[] = live.map(e => ({ ...e }))

    for (let i = 0; i < allCombos.length; i++) {
      const { model, backend } = allCombos[i]
      setProgress(`${model} / ${BACKEND_LABELS[backend] ?? backend}`)

      try {
        const form = new FormData()
        form.append('file', benchFile)
        const res = await fetch(
          `${BACKEND_URL}/detect/image?model_name=${model}&backend=${backend}`,
          { method: 'POST', body: form }
        )
        if (res.ok) {
          const data: DetectResponse = await res.json()
          const idx = updated.findIndex(e => e.model === model && e.backend === backend)
          if (idx >= 0) {
            updated[idx] = { ...updated[idx], avg_latency_ms: data.latency_ms }
          } else {
            updated.push({ model, backend, avg_latency_ms: data.latency_ms, map50: null, map50_95: null })
          }
          updateRow(model, backend, data.latency_ms)
        }
      } catch {
        // Skip combinations that fail (model not loaded)
      }

      setDoneCount(i + 1)
    }

    // Persist to backend so mAP data is preserved
    try {
      const res = await fetch(`${BACKEND_URL}/benchmark`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ entries: updated }),
      })
      if (res.ok) {
        const saved = await res.json()
        onEntriesChange(saved.entries ?? updated)
        setLive(saved.entries ?? updated)
      }
    } catch { /* best-effort save */ }

    setRunning(false)
    setProgress('')
  }

  const cpuBaseline: Record<string, number> = {}
  for (const e of live) {
    if (e.backend === 'pytorch-cpu' && e.avg_latency_ms !== null) {
      cpuBaseline[e.model] = e.avg_latency_ms
    }
  }

  function cell(val: number | null, suffix = '') {
    if (val === null) return <span className="text-white/20">—</span>
    return <span>{val.toFixed(1)}{suffix}</span>
  }

  return (
    <div className="flex-1 overflow-auto p-8">
      <div className="flex flex-col gap-8 max-w-4xl mx-auto">

        {/* Run All controls */}
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-3 border border-white/20 hover:border-white/50
                            rounded-sm py-2 px-4 cursor-pointer transition-colors text-sm text-white/60
                            hover:text-white/90 flex-shrink-0">
            <span>{benchFile?.name ?? 'Choose image for benchmark'}</span>
            <input
              ref={benchFileRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={e => setBenchFile(e.target.files?.[0] ?? null)}
            />
          </label>

          <button
            onClick={runAll}
            disabled={!benchFile || running}
            className="bg-white text-black hover:bg-white/80 disabled:bg-white/10
                       disabled:text-white/20 rounded-sm py-2 px-5 text-sm tracking-widest
                       uppercase transition-colors flex-shrink-0"
          >
            {running ? 'Running...' : 'Run All'}
          </button>

          {running && (
            <div className="flex items-center gap-3 text-sm text-white/70">
              <span className="font-mono">{doneCount}/{allCombos.length}</span>
              <span className="text-white/55">{progress}</span>
            </div>
          )}

          <button
            onClick={runEval}
            disabled={evalRunning || running}
            className="border border-white/20 hover:border-white/50 text-white/60 hover:text-white/90
                       disabled:border-white/10 disabled:text-white/20 rounded-sm py-2 px-5 text-sm
                       tracking-widest uppercase transition-colors flex-shrink-0"
          >
            {evalRunning ? 'Evaluating...' : 'Run Eval'}
          </button>
        </div>

        {evalStatus && (
          <p className={`text-xs ${evalStatus.startsWith('Error') ? 'text-red-400' : 'text-white/65'}`}>
            {evalStatus}
          </p>
        )}

        {/* Comparison tables */}
        {models.map(model => {
          const rows = BACKEND_ORDER
            .map(b => live.find(e => e.model === model && e.backend === b))
            .filter(Boolean) as BenchmarkEntry[]

          if (rows.length === 0) return null

          return (
            <div key={model}>
              <h2 className="text-xs tracking-widest uppercase text-white/65 mb-4">{model}</h2>
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-white/10 text-xs tracking-widest uppercase text-white/55">
                    <th className="text-left py-2 pr-6 font-normal">Backend</th>
                    <th className="text-right py-2 px-4 font-normal">Latency (ms)</th>
                    <th className="text-right py-2 px-4 font-normal">Speedup</th>
                    <th className="text-right py-2 px-4 font-normal">mAP@50</th>
                    <th className="text-right py-2 pl-4 font-normal">mAP@50:95</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(row => {
                    const speedup = (row.avg_latency_ms !== null && cpuBaseline[model])
                      ? cpuBaseline[model] / row.avg_latency_ms
                      : null
                    const isBest = row.avg_latency_ms !== null &&
                      row.avg_latency_ms === Math.min(...rows.map(r => r.avg_latency_ms ?? Infinity))

                    return (
                      <tr key={row.backend} className="border-b border-white/5 text-white/70">
                        <td className="py-3 pr-6 text-white/90">
                          {BACKEND_LABELS[row.backend] ?? row.backend}
                          {row.backend === 'pytorch-cpu' && (
                            <span className="ml-2 text-[10px] text-white/50 tracking-wider uppercase">baseline</span>
                          )}
                        </td>
                        <td className={`text-right py-3 px-4 font-mono ${isBest ? 'text-green-400' : ''}`}>
                          {cell(row.avg_latency_ms)}
                        </td>
                        <td className="text-right py-3 px-4 font-mono text-white/40">
                          {speedup !== null
                            ? <span className={speedup >= 1 ? 'text-white/60' : 'text-red-400/70'}>{speedup.toFixed(2)}×</span>
                            : <span className="text-white/20">—</span>}
                        </td>
                        <td className="text-right py-3 px-4 font-mono">{cell(row.map50)}</td>
                        <td className="text-right py-3 pl-4 font-mono">{cell(row.map50_95)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )
        })}

      </div>
    </div>
  )
}
