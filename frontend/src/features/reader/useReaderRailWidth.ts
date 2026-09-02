import {
  type KeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'

export const READING_RAIL_STORAGE_KEY = 'knowledge-reader-rail-width'
export const READING_RAIL_DEFAULT_WIDTH = 360
export const READING_RAIL_MIN_WIDTH = 240
export const READING_RAIL_MAX_WIDTH = 520
const RESIZE_STEP = 16

function clampWidth(width: number): number {
  return Math.min(
    READING_RAIL_MAX_WIDTH,
    Math.max(READING_RAIL_MIN_WIDTH, Math.round(width))
  )
}

function readStored(): number {
  if (typeof window === 'undefined') return READING_RAIL_DEFAULT_WIDTH
  try {
    const raw = window.localStorage.getItem(READING_RAIL_STORAGE_KEY)
    const parsed = Number.parseInt(raw ?? '', 10)
    return Number.isFinite(parsed) ? clampWidth(parsed) : READING_RAIL_DEFAULT_WIDTH
  } catch {
    // 隐私模式下 localStorage 会抛，回落到默认宽度，不让阅读页崩。
    return READING_RAIL_DEFAULT_WIDTH
  }
}

function persistWidth(width: number): void {
  try {
    window.localStorage.setItem(READING_RAIL_STORAGE_KEY, String(width))
  } catch {
    // 存不下就只在本次会话生效。
  }
}

function notifyCharts(): void {
  window.dispatchEvent(new Event('resize'))
}

/**
 * 章节目录宽度：可拖动，记在 localStorage，范围锁在默认版心还能站住的区间。
 */
export function useReaderRailWidth(): {
  width: number
  resizing: boolean
  onResizePointerDown: (event: ReactPointerEvent<HTMLElement>) => void
  onResizeKeyDown: (event: KeyboardEvent<HTMLElement>) => void
  resetWidth: () => void
} {
  const [width, setWidth] = useState(readStored)
  const [resizing, setResizing] = useState(false)
  const startX = useRef(0)
  const startWidth = useRef(READING_RAIL_DEFAULT_WIDTH)
  const liveWidth = useRef(width)
  liveWidth.current = width

  const commitWidth = useCallback((next: number) => {
    const clamped = clampWidth(next)
    setWidth(clamped)
    persistWidth(clamped)
    notifyCharts()
    return clamped
  }, [])

  const resetWidth = useCallback(() => {
    commitWidth(READING_RAIL_DEFAULT_WIDTH)
  }, [commitWidth])

  const onResizePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => {
      if (event.button !== 0) return
      event.preventDefault()
      event.stopPropagation()
      startX.current = event.clientX
      startWidth.current = width
      event.currentTarget.setPointerCapture(event.pointerId)
      setResizing(true)
    },
    [width]
  )

  const onResizeKeyDown = useCallback(
    (event: KeyboardEvent<HTMLElement>) => {
      if (event.key === 'ArrowLeft') {
        event.preventDefault()
        commitWidth(width - RESIZE_STEP)
        return
      }
      if (event.key === 'ArrowRight') {
        event.preventDefault()
        commitWidth(width + RESIZE_STEP)
        return
      }
      if (event.key === 'Home') {
        event.preventDefault()
        commitWidth(READING_RAIL_MIN_WIDTH)
        return
      }
      if (event.key === 'End') {
        event.preventDefault()
        commitWidth(READING_RAIL_MAX_WIDTH)
        return
      }
    },
    [commitWidth, width]
  )

  useEffect(() => {
    if (!resizing) return

    const onMove = (event: PointerEvent) => {
      const next = clampWidth(startWidth.current + (event.clientX - startX.current))
      liveWidth.current = next
      setWidth(next)
    }
    const onUp = () => {
      setResizing(false)
      persistWidth(liveWidth.current)
      notifyCharts()
    }

    document.addEventListener('pointermove', onMove)
    document.addEventListener('pointerup', onUp)
    document.addEventListener('pointercancel', onUp)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    return () => {
      document.removeEventListener('pointermove', onMove)
      document.removeEventListener('pointerup', onUp)
      document.removeEventListener('pointercancel', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [resizing])

  return { width, resizing, onResizePointerDown, onResizeKeyDown, resetWidth }
}
