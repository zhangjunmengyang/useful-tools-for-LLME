import { useCallback, useState } from 'react'

export type ReaderMeasurePreference = 'compact' | 'comfortable' | 'wide'

export const READING_MEASURE_STORAGE_KEY = 'knowledge-reader-measure'
export const READING_MEASURE_DEFAULT: ReaderMeasurePreference = 'comfortable'

/** 17px 全角字下约 38 / 42 / 49 个汉字。适中落在视觉规范的 720-840 区间。 */
export const READING_MEASURE_PX: Record<ReaderMeasurePreference, number> = {
  compact: 640,
  comfortable: 720,
  wide: 840,
}

export const READING_MEASURE_LABEL: Record<ReaderMeasurePreference, string> = {
  compact: '窄栏',
  comfortable: '适中',
  wide: '宽栏',
}

const PREFERENCES: readonly ReaderMeasurePreference[] = ['compact', 'comfortable', 'wide']

function readStored(): ReaderMeasurePreference {
  if (typeof window === 'undefined') return READING_MEASURE_DEFAULT
  try {
    const raw = window.localStorage.getItem(READING_MEASURE_STORAGE_KEY)
    return PREFERENCES.includes(raw as ReaderMeasurePreference)
      ? (raw as ReaderMeasurePreference)
      : READING_MEASURE_DEFAULT
  } catch {
    return READING_MEASURE_DEFAULT
  }
}

/**
 * 课程正文版心：窄 / 适中 / 宽三档，记在 localStorage。
 * 只改阅读器子树的 `--reading-measure`，不是整站字号档位。
 */
export function useReaderMeasure(): {
  preference: ReaderMeasurePreference
  measurePx: number
  setPreference: (next: ReaderMeasurePreference) => void
} {
  const [preference, setPreferenceState] = useState<ReaderMeasurePreference>(readStored)

  const setPreference = useCallback((next: ReaderMeasurePreference) => {
    setPreferenceState(next)
    try {
      window.localStorage.setItem(READING_MEASURE_STORAGE_KEY, next)
    } catch {
      // 存不下就只在本次会话生效。
    }
    window.dispatchEvent(new Event('resize'))
  }, [])

  return {
    preference,
    measurePx: READING_MEASURE_PX[preference],
    setPreference,
  }
}
