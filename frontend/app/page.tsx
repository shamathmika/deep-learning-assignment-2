'use client'

import { useState, useRef, useEffect } from 'react'
import DetectionCanvas, { colourFor, type Detection } from '@/components/DetectionCanvas'

const BACKEND_URL = 'http://localhost:8000'

type PageView      = 'detection' | 'benchmark'
type DetectionMode = 'video' | 'image'

type DetectResponse = {
  model:      string
  backend:    string
  latency_ms: number
  detections: Detection[]
}

type SampleFrame = {
  frame_index: number
  image_b64:   string
  detections:  Detection[]
  latency_ms:  number
}

type VideoDetectResponse = {
  model:            string
  backend:          string
  total_frames:     number
  processed_frames: number
  avg_latency_ms:   number
  frame_results: {
    frame_index: number
    latency_ms:  number
    detections:  Detection[]
  }[]
  sample_frames: SampleFrame[]
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
  const [videoResults, setVideoResults] = useState<VideoDetectResponse | null>(null)

  // Benchmark state
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkEntry[]>([])

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
      .then(data => setBenchmarkData(data.entries ?? []))
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

  const supportedBackends = (modelBackends[modelName] ?? Object.keys(BACKEND_LABELS))
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
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  // Aggregate per-class detection counts across all video frames
  const videoClassSummary = (() => {
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
  })()

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
                  pageView === v ? 'bg-white text-black' : 'text-white/40 hover:text-white/80'
                }`}
              >
                {v}
              </button>
            ))}
          </nav>
          {displayLatency !== null && (
            <div className="flex items-center gap-3">
              <span className="text-white/40 tracking-widest uppercase text-xs">
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
        <BenchmarkPanel entries={benchmarkData} />
      ) : (
        <div className="flex flex-1 overflow-hidden">

          {/* Left panel */}
          <aside className="w-72 flex-shrink-0 border-r border-white/10 p-6 flex flex-col gap-6 overflow-y-auto">
            {/* Mode tabs */}
            <div className="flex border border-white/20 rounded-sm overflow-hidden">
              {(['video', 'image'] as DetectionMode[]).map(m => (
                <button
                  key={m}
                  onClick={() => handleModeChange(m)}
                  className={`flex-1 py-2 text-xs tracking-widest uppercase transition-colors ${
                    mode === m ? 'bg-white text-black' : 'text-white/40 hover:text-white/70'
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
                <label className="text-xs tracking-widest uppercase text-white/40">
                  {mode === 'image' ? 'Image' : 'Video'}
                </label>
                <label className="flex items-center justify-center border border-white/20 hover:border-white/50
                                  rounded-sm py-3 px-4 cursor-pointer transition-colors text-sm text-white/60
                                  hover:text-white/90">
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
                <label className="text-xs tracking-widest uppercase text-white/40">Model</label>
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
                <label className="text-xs tracking-widest uppercase text-white/40">Backend</label>
                <select
                  value={backend}
                  onChange={e => setBackend(e.target.value)}
                  className="bg-black border border-white/20 hover:border-white/50 rounded-sm
                             px-3 py-2 text-sm text-white transition-colors outline-none"
                >
                  {supportedBackends.map(b => (
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
          </aside>

          {/* Center panel */}
          <main className="flex-1 overflow-auto flex items-center justify-center p-6">
            {mode === 'image' ? (
              imageUrl ? (
                <div className="flex items-center justify-center h-full w-full">
                  <DetectionCanvas imageUrl={imageUrl} detections={detections} hoveredIndex={hoveredIndex} />
                </div>
              ) : (
                <EmptyPlaceholder icon="image" label="Upload an image to begin" />
              )
            ) : (
              videoResults ? (
                <VideoResults results={videoResults} />
              ) : (
                <EmptyPlaceholder
                  icon="video"
                  label={loading ? 'Processing video — this may take a moment...' : 'Upload a video to begin'}
                />
              )
            )}
          </main>

          {/* Right panel */}
          <aside className="w-64 flex-shrink-0 border-l border-white/10 p-6 overflow-y-auto">
            {mode === 'image' ? (
              <>
                <h2 className="text-xs tracking-widest uppercase text-white/40 mb-4">
                  Detections {detections.length > 0 && `(${detections.length})`}
                </h2>
                {detections.length === 0 ? (
                  <p className="text-white/20 text-xs">None yet</p>
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
                        <span className="text-xs font-mono text-white/40 flex-shrink-0">
                          {(d.confidence * 100).toFixed(0)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            ) : (
              <>
                <h2 className="text-xs tracking-widest uppercase text-white/40 mb-4">
                  Classes {videoClassSummary.length > 0 && `(${videoClassSummary.length})`}
                </h2>
                {videoClassSummary.length === 0 ? (
                  <p className="text-white/20 text-xs">None yet</p>
                ) : (
                  <ul className="flex flex-col gap-1">
                    {videoClassSummary.map((item, i) => (
                      <li key={i} className="flex items-center justify-between gap-3 px-3 py-2 rounded-sm">
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: colourFor(item.class) }} />
                        <span className="flex-1 text-sm text-white/80 truncate">{item.class}</span>
                        <span className="text-xs font-mono text-white/40 flex-shrink-0">
                          {item.count}×
                        </span>
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
    <div className="flex flex-col items-center gap-3 text-white/20 select-none">
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

function VideoResults({ results }: { results: VideoDetectResponse }) {
  const [hoveredFrame,     setHoveredFrame]     = useState<number | null>(null)
  const [hoveredDetection, setHoveredDetection] = useState<number | null>(null)

  const stats = [
    { label: 'Avg Latency',      value: `${results.avg_latency_ms.toFixed(1)} ms` },
    { label: 'Frames Processed', value: `${results.processed_frames} / ${results.total_frames}` },
    { label: 'Model',            value: results.model },
    { label: 'Backend',          value: BACKEND_LABELS[results.backend] ?? results.backend },
  ]

  return (
    <div className="flex flex-col gap-6 w-full h-full overflow-y-auto py-2">
      {/* Stats row */}
      <div className="grid grid-cols-4 gap-3 flex-shrink-0">
        {stats.map(({ label, value }) => (
          <div key={label} className="border border-white/10 rounded-sm p-3">
            <div className="text-xs tracking-widest uppercase text-white/30 mb-1">{label}</div>
            <div className="text-white font-mono text-sm truncate">{value}</div>
          </div>
        ))}
      </div>

      {/* Annotated frame grid */}
      {results.sample_frames.length > 0 && (
        <div className="grid grid-cols-3 gap-3">
          {results.sample_frames.map((sf, fi) => (
            <div
              key={sf.frame_index}
              className="relative rounded-sm overflow-hidden border border-white/10 cursor-default"
              onMouseEnter={() => { setHoveredFrame(fi); setHoveredDetection(null) }}
              onMouseLeave={() => { setHoveredFrame(null); setHoveredDetection(null) }}
            >
              <DetectionCanvas
                imageUrl={sf.image_b64}
                detections={sf.detections}
                hoveredIndex={hoveredFrame === fi ? hoveredDetection : null}
              />
              <div className="absolute bottom-0 left-0 right-0 px-2 py-1 bg-black/60 text-white/50 text-xs font-mono">
                f{sf.frame_index} · {sf.latency_ms.toFixed(1)}ms
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function BenchmarkPanel({ entries }: { entries: BenchmarkEntry[] }) {
  const models = ['yolov8s', 'rtdetr-l']

  // CPU baseline latency per model, used to compute speedup
  const cpuBaseline: Record<string, number> = {}
  for (const e of entries) {
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
      <div className="flex flex-col gap-10 max-w-4xl mx-auto">
        {models.map(model => {
          const rows = BACKEND_ORDER
            .map(b => entries.find(e => e.model === model && e.backend === b))
            .filter(Boolean) as BenchmarkEntry[]

          return (
            <div key={model}>
              <h2 className="text-xs tracking-widest uppercase text-white/40 mb-4">{model}</h2>
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-white/10 text-xs tracking-widest uppercase text-white/30">
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
                    const isFastestLatency = row.avg_latency_ms !== null &&
                      row.avg_latency_ms === Math.min(...rows.map(r => r.avg_latency_ms ?? Infinity))

                    return (
                      <tr key={row.backend} className="border-b border-white/5 text-white/70 hover:bg-white/3">
                        <td className="py-3 pr-6 text-white/90">
                          {BACKEND_LABELS[row.backend] ?? row.backend}
                          {row.backend === 'pytorch-cpu' && (
                            <span className="ml-2 text-[10px] text-white/30 tracking-wider uppercase">baseline</span>
                          )}
                        </td>
                        <td className={`text-right py-3 px-4 font-mono ${isFastestLatency ? 'text-green-400' : ''}`}>
                          {cell(row.avg_latency_ms)}
                        </td>
                        <td className="text-right py-3 px-4 font-mono text-white/50">
                          {speedup !== null ? `${speedup.toFixed(2)}×` : <span className="text-white/20">—</span>}
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
        <p className="text-white/20 text-xs">
          mAP columns populate after running <span className="font-mono">python scripts/run_map_eval.py</span>
        </p>
      </div>
    </div>
  )
}
