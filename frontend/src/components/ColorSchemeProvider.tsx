import {
  type ColorScheme,
  ColorSchemeContext,
  applyColorSchemeToDocument,
  persistColorScheme,
  readStoredColorScheme,
} from '@/lib/colorScheme'
import { type ReactNode, createContext, useCallback, useContext, useEffect, useState } from 'react'

interface ColorSchemeControls {
  colorScheme: ColorScheme
  setColorScheme: (scheme: ColorScheme) => void
  toggleColorScheme: () => void
}

const ColorSchemeControlsContext = createContext<ColorSchemeControls | null>(null)

export function useColorSchemeControls(): ColorSchemeControls {
  const controls = useContext(ColorSchemeControlsContext)
  if (!controls) throw new Error('useColorSchemeControls 必须在 ColorSchemeProvider 内使用')
  return controls
}

/**
 * 全局配色。深色是币安交易端那套面色，浅色是白底加黄。
 * 记在 localStorage，跨标签页同步。
 */
export function ColorSchemeProvider({ children }: { children: ReactNode }) {
  const [colorScheme, setSchemeState] = useState<ColorScheme>(readStoredColorScheme)

  useEffect(() => {
    applyColorSchemeToDocument(colorScheme)
  }, [colorScheme])

  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key !== 'app-color-scheme') return
      setSchemeState(readStoredColorScheme())
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [])

  const setColorScheme = useCallback((next: ColorScheme) => {
    setSchemeState((current) => {
      if (current === next) return current
      persistColorScheme(next)
      // ECharts 在 canvas 里画，切主题要重新取尺寸重绘。
      window.dispatchEvent(new Event('resize'))
      return next
    })
  }, [])

  const toggleColorScheme = useCallback(() => {
    setColorScheme(readStoredColorScheme() === 'dark' ? 'light' : 'dark')
  }, [setColorScheme])

  return (
    <ColorSchemeContext.Provider value={colorScheme}>
      <ColorSchemeControlsContext.Provider
        value={{ colorScheme, setColorScheme, toggleColorScheme }}
      >
        {children}
      </ColorSchemeControlsContext.Provider>
    </ColorSchemeContext.Provider>
  )
}
