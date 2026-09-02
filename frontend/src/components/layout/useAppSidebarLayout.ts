import { useCallback, useEffect, useRef, useState } from 'react'

export const APP_SIDEBAR_COLLAPSED_STORAGE_KEY = 'app-sidebar-collapsed'
export const APP_SIDEBAR_WIDTH = 248
export const APP_SIDEBAR_COLLAPSED_WIDTH = 72

const DESKTOP_MEDIA = '(min-width: 768px)'

function readStoredCollapsed(): boolean {
  if (typeof window === 'undefined') return false
  try {
    const raw = window.localStorage.getItem(APP_SIDEBAR_COLLAPSED_STORAGE_KEY)
    return raw === '1' || raw === 'true'
  } catch {
    return false
  }
}

function readIsDesktop(): boolean {
  if (typeof window === 'undefined') return true
  return window.matchMedia(DESKTOP_MEDIA).matches
}

function persistCollapsed(collapsed: boolean): void {
  try {
    window.localStorage.setItem(APP_SIDEBAR_COLLAPSED_STORAGE_KEY, collapsed ? '1' : '0')
  } catch {
    // 存不下就只在本次会话生效。
  }
}

function notifyCharts(): void {
  window.dispatchEvent(new Event('resize'))
}

export interface AppSidebarLayout {
  visualWidth: number
  collapsed: boolean
  isDesktop: boolean
  toggleCollapsed: () => void
  setCollapsed: (collapsed: boolean) => void
}

/**
 * 主应用侧栏收起态，记在 localStorage。
 *
 * 宽度是固定值，不可拖动：收起/展开已经覆盖了「要不要占这块地方」的需求，
 * 再加一条拖动边会在侧栏右侧常驻一根 hover 变色的粗条，得不偿失。
 * 窄屏永远是图标栏。
 */
export function useAppSidebarLayout(): AppSidebarLayout {
  const [collapsed, setCollapsedState] = useState(readStoredCollapsed)
  const [isDesktop, setIsDesktop] = useState(readIsDesktop)
  const collapsedRef = useRef(collapsed)
  collapsedRef.current = collapsed

  useEffect(() => {
    const query = window.matchMedia(DESKTOP_MEDIA)
    const onChange = () => setIsDesktop(query.matches)
    query.addEventListener('change', onChange)
    setIsDesktop(query.matches)
    return () => query.removeEventListener('change', onChange)
  }, [])

  const setCollapsed = useCallback((next: boolean) => {
    if (collapsedRef.current === next) return
    collapsedRef.current = next
    setCollapsedState(next)
    persistCollapsed(next)
    notifyCharts()
  }, [])

  const toggleCollapsed = useCallback(() => {
    setCollapsed(!collapsedRef.current)
  }, [setCollapsed])

  const visualWidth = !isDesktop || collapsed ? APP_SIDEBAR_COLLAPSED_WIDTH : APP_SIDEBAR_WIDTH

  return {
    visualWidth,
    collapsed,
    isDesktop,
    toggleCollapsed,
    setCollapsed,
  }
}
