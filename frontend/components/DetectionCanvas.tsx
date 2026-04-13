'use client'

import { useEffect, useRef, useCallback } from 'react'

export type Detection = {
  class: string
  confidence: number
  box: [number, number, number, number]
}

type Props = {
  imageUrl: string
  detections: Detection[]
  hoveredIndex: number | null
}

const PALETTE = [
  '#4ade80', '#f87171', '#fb923c', '#38bdf8', '#c084fc',
  '#fbbf24', '#f472b6', '#34d399', '#e879f9', '#60a5fa',
]

export function colourFor(cls: string): string {
  let hash = 0
  for (let i = 0; i < cls.length; i++) {
    hash = cls.charCodeAt(i) + ((hash << 5) - hash)
  }
  return PALETTE[Math.abs(hash) % PALETTE.length]
}

export default function DetectionCanvas({ imageUrl, detections, hoveredIndex }: Props) {
  const imgRef    = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Extracted into a stable callback so both useEffect and onLoad can call it
  const draw = useCallback(() => {
    const img    = imgRef.current
    const canvas = canvasRef.current
    // Guard: image must be fully loaded before we can read its dimensions
    if (!img || !canvas || img.naturalWidth === 0) return

    const displayW = img.clientWidth
    const displayH = img.clientHeight
    canvas.width   = displayW
    canvas.height  = displayH

    const scaleX = displayW / img.naturalWidth
    const scaleY = displayH / img.naturalHeight

    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, displayW, displayH)

    detections.forEach((det, i) => {
      const [x1, y1, x2, y2] = det.box
      const sx1 = x1 * scaleX
      const sy1 = y1 * scaleY
      const sw  = (x2 - x1) * scaleX
      const sh  = (y2 - y1) * scaleY

      const colour    = colourFor(det.class)
      const isHovered = hoveredIndex === i
      const isDimmed  = hoveredIndex !== null && !isHovered

      ctx.globalAlpha = isDimmed ? 0.2 : 1.0
      ctx.lineWidth   = isHovered ? 3 : 1.5
      ctx.strokeStyle = colour
      ctx.strokeRect(sx1, sy1, sw, sh)

      const label = `${det.class}  ${(det.confidence * 100).toFixed(0)}%`
      ctx.font     = `${isHovered ? 'bold ' : ''}12px monospace`
      const textW  = ctx.measureText(label).width
      ctx.fillStyle = colour
      ctx.fillRect(sx1, sy1 - 22, textW + 10, 22)
      ctx.fillStyle = '#000000'
      ctx.fillText(label, sx1 + 5, sy1 - 6)
    })

    ctx.globalAlpha = 1.0
  }, [detections, hoveredIndex])

  // Redraw whenever detections or hover state changes
  useEffect(() => { draw() }, [draw])

  return (
    <div className="relative" style={{ display: 'inline-block', lineHeight: 0 }}>
      <img
        ref={imgRef}
        src={imageUrl}
        alt="detection input"
        // onLoad fires after the browser has decoded the image and knows naturalWidth/Height.
        // Without this, draw() runs before dimensions are available and boxes are misplaced.
        onLoad={draw}
        className="block rounded-sm"
        style={{
          maxWidth:  'min(100vw - 600px, 960px)',
          maxHeight: 'calc(100vh - 120px)',
          width:     'auto',
          height:    'auto',
        }}
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-none"
      />
    </div>
  )
}
