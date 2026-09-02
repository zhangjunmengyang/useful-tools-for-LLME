import { createContext, useContext } from 'react'

export type ColorScheme = 'light' | 'dark'

export const COLOR_SCHEME_STORAGE_KEY = 'app-color-scheme'

/**
 * 配色作用域。平台提供浅色和深色两套面孔，共用同一组语义 token，
 * 切换只改颜色，不改布局、字号、间距和组件结构。
 *
 * 组件靠它读当前配色。CSS 侧靠 <html class="dark"> 级联即可，
 * 但 ECharts 画在 canvas 里读不到 CSS 变量，必须在初始化时显式知道自己在哪套配色下。
 */
export const ColorSchemeContext = createContext<ColorScheme>('light')

export function useColorScheme(): ColorScheme {
  return useContext(ColorSchemeContext)
}

export function readStoredColorScheme(): ColorScheme {
  if (typeof window === 'undefined') return 'light'
  try {
    const raw = window.localStorage.getItem(COLOR_SCHEME_STORAGE_KEY)
    if (raw === 'dark' || raw === 'light') return raw
  } catch {
    // 隐私模式下 localStorage 会抛，回落到浅色。
  }
  return 'light'
}

export function persistColorScheme(scheme: ColorScheme): void {
  try {
    window.localStorage.setItem(COLOR_SCHEME_STORAGE_KEY, scheme)
  } catch {
    // 存不下就只在本次会话生效。
  }
}

/** 单一真源：始终由 <html> 的 class 承载，CSS 和 JS 读到的是同一个事实。 */
export function applyColorSchemeToDocument(scheme: ColorScheme): void {
  document.documentElement.classList.toggle('dark', scheme === 'dark')
  document.documentElement.style.colorScheme = scheme
}
