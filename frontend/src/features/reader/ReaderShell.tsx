import { useI18n } from '@/components/I18nProvider'
import { cn } from '@/lib/utils'
import { ChevronRight, PanelLeft, X } from 'lucide-react'
import { type CSSProperties, type ReactNode, useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  READING_MEASURE_LABEL,
  type ReaderMeasurePreference,
  useReaderMeasure,
} from './useReaderMeasure'
import {
  READING_RAIL_MAX_WIDTH,
  READING_RAIL_MIN_WIDTH,
  useReaderRailWidth,
} from './useReaderRailWidth'

const MEASURE_ITEMS: readonly ReaderMeasurePreference[] = ['compact', 'comfortable', 'wide']

function MeasureGlyph({
  preference,
  selected,
}: {
  preference: ReaderMeasurePreference
  selected: boolean
}) {
  const inset = preference === 'compact' ? 5 : preference === 'comfortable' ? 3 : 1
  return (
    <svg viewBox="0 0 16 16" className="h-4 w-4" aria-hidden="true">
      <rect
        x={inset}
        y="2.5"
        width={16 - inset * 2}
        height="11"
        rx="1.5"
        fill={selected ? 'currentColor' : 'none'}
        stroke="currentColor"
        strokeWidth="1.5"
      />
    </svg>
  )
}

function MeasureToggle({
  preference,
  onChange,
}: {
  preference: ReaderMeasurePreference
  onChange: (next: ReaderMeasurePreference) => void
}) {
  const { language, tr } = useI18n()
  const measureLabel =
    language === 'en'
      ? { compact: 'Narrow', comfortable: 'Comfortable', wide: 'Wide' }
      : READING_MEASURE_LABEL
  return (
    <fieldset className="flex items-center">
      <legend className="sr-only">{tr('正文宽度', 'Text width')}</legend>
      {MEASURE_ITEMS.map((item) => {
        const selected = item === preference
        return (
          <button
            key={item}
            type="button"
            aria-pressed={selected}
            aria-label={`${tr('正文宽度', 'Text width')}：${measureLabel[item]}`}
            title={`${tr('正文宽度', 'Text width')}：${measureLabel[item]}`}
            onClick={() => onChange(item)}
            className={cn(
              'rounded-md p-1.5 transition-colors',
              selected
                ? 'bg-muted text-foreground'
                : 'text-muted-foreground hover:bg-muted hover:text-foreground'
            )}
          >
            <MeasureGlyph preference={item} selected={selected} />
          </button>
        )
      })}
    </fieldset>
  )
}

function ProgressMeter({
  position,
  total,
  label,
}: {
  position: number
  total: number
  label?: string
}) {
  return (
    <div className="hidden items-center gap-2 md:flex">
      <span className="font-mono text-2xs tabular-nums text-muted-foreground">
        {label ?? `${String(position).padStart(2, '0')} / ${String(total).padStart(2, '0')}`}
      </span>
    </div>
  )
}

/**
 * 课程阅读器外壳：瘦顶栏 + 左侧章节目录 + 靠左正文。
 *
 * 铺满 Header 下方剩余高度；左栏与正文各自滚动。
 * 顶栏和左栏在整个阅读会话里常驻，正文区自行处理加载态，切章不会整页卸载。
 */
export function ReaderShell({
  courseTitle,
  courseHref,
  stageTitle,
  chapterTitle,
  position,
  total,
  progressLabel,
  rail,
  actions,
  children,
}: {
  courseTitle: string
  courseHref: string
  stageTitle?: string
  chapterTitle?: string
  position: number
  total: number
  progressLabel?: string
  rail: ReactNode
  actions?: ReactNode
  children: ReactNode
}) {
  const location = useLocation()
  const { tr } = useI18n()
  const [railOpen, setRailOpen] = useState(false)
  const { preference: measurePreference, measurePx, setPreference: setMeasure } = useReaderMeasure()
  const {
    width: railWidth,
    resizing,
    onResizePointerDown,
    onResizeKeyDown,
    resetWidth,
  } = useReaderRailWidth()

  // 窄屏抽屉在换页后自动收起，避免挡住刚打开的正文。
  // biome-ignore lint/correctness/useExhaustiveDependencies: 路由变化是刻意的收起触发条件。
  useEffect(() => setRailOpen(false), [location.pathname, location.search])

  useEffect(() => {
    if (!railOpen) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setRailOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [railOpen])

  return (
    <div
      className="flex h-full min-h-0 flex-col bg-background text-foreground"
      style={{ '--reading-measure': `${measurePx}px` } as CSSProperties}
    >
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-border bg-background/90 px-4">
        <button
          type="button"
          onClick={() => setRailOpen(true)}
          aria-label={tr('打开章节目录', 'Open lesson list')}
          className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground lg:hidden"
        >
          <PanelLeft className="h-4 w-4" />
        </button>

        <nav aria-label={tr('当前位置', 'You are here')} className="flex min-w-0 items-center gap-1.5 text-xs">
          <Link
            to={courseHref}
            className="shrink-0 font-medium text-foreground transition-colors hover:text-primary-ink"
          >
            {courseTitle}
          </Link>
          {stageTitle ? (
            <>
              <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground/60" />
              <span className="hidden truncate text-muted-foreground lg:inline">{stageTitle}</span>
            </>
          ) : null}
          {chapterTitle ? (
            <>
              <ChevronRight className="hidden h-3 w-3 shrink-0 text-muted-foreground/60 lg:inline" />
              <span className="truncate text-muted-foreground">{chapterTitle}</span>
            </>
          ) : null}
        </nav>

        <div className="ml-auto flex shrink-0 items-center gap-3">
          <ProgressMeter position={position} total={total} label={progressLabel} />
          <MeasureToggle preference={measurePreference} onChange={setMeasure} />
          {actions}
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <aside
          className="relative hidden h-full shrink-0 overflow-hidden border-r border-border lg:block"
          style={{ width: railWidth }}
        >
          {rail}
          <button
            type="button"
            aria-label={tr('调整章节目录宽度', 'Resize lesson list')}
            aria-orientation="vertical"
            aria-valuemin={READING_RAIL_MIN_WIDTH}
            aria-valuemax={READING_RAIL_MAX_WIDTH}
            aria-valuenow={railWidth}
            title={tr('拖动调整宽度，双击恢复默认', 'Drag to resize, double-click to reset')}
            className={cn(
              'absolute inset-y-0 right-0 z-20 w-2 cursor-col-resize touch-none border-0 bg-transparent p-0 transition-colors hover:bg-primary/40 focus-visible:bg-primary/40 focus-visible:outline-none',
              resizing && 'bg-primary'
            )}
            onPointerDown={onResizePointerDown}
            onKeyDown={onResizeKeyDown}
            onDoubleClick={resetWidth}
          />
        </aside>

        {railOpen ? (
          <div className="fixed inset-0 z-50 lg:hidden">
            <button
              type="button"
              aria-label={tr('关闭章节目录', 'Close lesson list')}
              onClick={() => setRailOpen(false)}
              className="absolute inset-0 bg-foreground/20 backdrop-blur-[1px]"
            />
            <div className="absolute inset-y-0 left-0 flex w-[min(20rem,85vw)] flex-col border-r border-border bg-background shadow-depth-3">
              <div className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4">
                <span className="text-xs font-medium text-foreground">{tr('章节目录', 'Lessons')}</span>
                <button
                  type="button"
                  onClick={() => setRailOpen(false)}
                  aria-label={tr('关闭章节目录', 'Close lesson list')}
                  className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto">{rail}</div>
            </div>
          </div>
        ) : null}

        <div
          data-reader-scroll="article"
          className={cn(
            'min-h-0 min-w-0 flex-1 overflow-y-auto',
            resizing && 'pointer-events-none'
          )}
        >
          {children}
        </div>
      </div>
    </div>
  )
}

/**
 * 正文从章节目录右侧起排，版心限制行长，贴左不居中。
 */
export function ReaderColumn({
  children,
  className,
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <div className={cn('reading-scope w-full px-6 py-8 sm:px-8', className)}>
      <div className="w-full min-w-0 max-w-[var(--reading-measure)]">{children}</div>
    </div>
  )
}
