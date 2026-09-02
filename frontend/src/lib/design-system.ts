/**
 * Frontend design system primitives.
 *
 * 视觉基准是币安。上游原包在 design-system/binance/，
 * 平台映射和每一处偏离的理由在 design-system/platform-tokens.md。
 *
 * Keep visual decisions here instead of scattering color palettes and chip
 * classes across feature pages.
 */

/**
 * 图表色。ECharts 画在 canvas 里读不到 CSS 变量，只能给静态值，
 * 所以这批取中间明度，白底和 #181A20 上都能读出来。
 * 行情涨跌用 up/down，不要拿 warning/info 当方向色。
 */
export const chartColors = {
  primary: '#F0B90B',
  primaryDeep: '#C99400',
  ink: '#1E2026',
  graphite: '#474D57',
  slate: '#707A8A',
  ash: '#929AA5',
  stone: '#B7BDC6',
  up: '#0BA36A',
  down: '#E02D46',
  destructive: '#E02D46',
  warning: '#D48806',
  info: '#2E7CF6',
} as const

export const neutralSeriesPalette = [
  chartColors.primary,
  chartColors.info,
  chartColors.up,
  chartColors.down,
  chartColors.slate,
  chartColors.graphite,
] as const

export const accountSeriesPalette = [
  chartColors.primary,
  chartColors.info,
  chartColors.slate,
  chartColors.primaryDeep,
  chartColors.graphite,
  chartColors.ash,
] as const

export const chartTokens = {
  colors: chartColors,
  series: {
    neutral: neutralSeriesPalette,
    account: accountSeriesPalette,
  },
} as const

export const typographyTokens = {
  pageTitle: 'text-xl font-semibold text-foreground md:text-2xl',
  pageDescription: 'text-xs font-medium text-muted-foreground',
  sectionTitle: 'text-base font-semibold text-foreground',
  sectionDescription: 'text-sm leading-6 text-muted-foreground',
  cardTitle: 'text-base font-semibold leading-snug text-foreground',
  cardTitleCompact: 'text-sm font-semibold text-foreground',
  entityTitle: 'text-lg font-semibold leading-snug text-foreground',
  dialogTitle: 'text-lg font-semibold leading-snug text-foreground',
  body: 'text-sm leading-6 text-foreground',
  /** 量化深耕章节导语。正文性质的大字，不是标题，所以不带字重。 */
  leadParagraph: 'text-lg leading-9 text-foreground',
  bodyMuted: 'text-sm leading-6 text-muted-foreground',
  label: 'text-xs font-medium text-muted-foreground',
  labelStrong: 'text-sm font-medium text-foreground',
  caption: 'text-xs text-muted-foreground',
  table: 'text-sm',
  tableHeader: 'text-xs font-medium text-muted-foreground',
  metricPrimary: 'text-2xl font-semibold tabular-nums',
  metricSecondary: 'text-xl font-semibold tabular-nums',
  metricCompact: 'text-base font-semibold tabular-nums',
  metricTiny: 'text-sm font-semibold tabular-nums',
  monoCaption: 'font-mono text-xs text-muted-foreground',
} as const

export function getSeriesColor(index: number): string {
  return neutralSeriesPalette[index % neutralSeriesPalette.length] ?? chartColors.primary
}

export function getAccountSeriesColor(index: number): string {
  return accountSeriesPalette[index % accountSeriesPalette.length] ?? chartColors.primary
}

export const chipStyles = {
  base: 'inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium',
  neutral: 'border-border bg-muted text-muted-foreground',
  outline: 'border-border bg-card text-foreground',
  // 黄色系一律用 primary-ink 上色：填充黄做文字在白底上读不出来。
  primary: 'border-primary/35 bg-primary-muted text-primary-ink',
  success: 'border-success/20 bg-success-muted text-success',
  warning: 'border-warning/25 bg-warning-muted text-warning',
  destructive: 'border-destructive/20 bg-destructive-muted text-destructive',
  info: 'border-info/20 bg-info-muted text-info',
} as const

export type SemanticIntent = keyof Omit<typeof chipStyles, 'base' | 'outline'>

export const selectedStyles = {
  border: 'border-primary',
  focusRing: 'focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
  item: 'bg-primary-muted text-primary-ink',
  ring: 'ring-2 ring-ring/40',
  segmented: 'border-primary/40 bg-card text-foreground shadow-depth-1',
  subtleSurface: 'bg-primary-muted/35',
  surface: 'bg-primary-muted text-primary-ink',
  tableRow: 'bg-primary-muted/50',
} as const

export const selectedTokens = selectedStyles

export type StatusTone = 'neutral' | 'primary' | 'success' | 'warning' | 'destructive' | 'info'

export const statusTokens = {
  neutral: {
    chipClass: chipStyles.neutral,
    badgeClass: 'border-border bg-muted text-muted-foreground',
    dotClass: 'bg-muted-foreground',
  },
  primary: {
    chipClass: chipStyles.primary,
    badgeClass: chipStyles.primary,
    dotClass: 'bg-primary',
  },
  success: {
    chipClass: chipStyles.success,
    badgeClass: chipStyles.success,
    dotClass: 'bg-success',
  },
  warning: {
    chipClass: chipStyles.warning,
    badgeClass: chipStyles.warning,
    dotClass: 'bg-warning',
  },
  destructive: {
    chipClass: chipStyles.destructive,
    badgeClass: chipStyles.destructive,
    dotClass: 'bg-destructive',
  },
  info: {
    chipClass: chipStyles.info,
    badgeClass: chipStyles.info,
    dotClass: 'bg-info',
  },
} as const satisfies Record<
  StatusTone,
  {
    chipClass: string
    badgeClass: string
    dotClass: string
  }
>

const statusToneMap: Record<string, StatusTone> = {
  active: 'success',
  completed: 'success',
  online: 'success',
  ready: 'success',
  running: 'success',
  success: 'success',
  error: 'destructive',
  failed: 'destructive',
  fatal: 'destructive',
  offline: 'destructive',
  unresponsive: 'destructive',
  cancelled: 'warning',
  paused: 'warning',
  pending: 'warning',
  starting: 'warning',
  stopping: 'warning',
  warning: 'warning',
  chunking: 'info',
  indexing: 'info',
  parsing: 'info',
  processing: 'info',
  syncing: 'info',
  archived: 'neutral',
  disabled: 'neutral',
  draft: 'neutral',
  inactive: 'neutral',
}

const transientStatuses = new Set(['chunking', 'indexing', 'parsing', 'pending', 'processing', 'starting', 'stopping', 'syncing'])

export function getStatusTone(status: string): StatusTone {
  return statusToneMap[status.toLowerCase()] ?? 'neutral'
}

export function getStatusToken(status: string): (typeof statusTokens)[StatusTone] {
  return statusTokens[getStatusTone(status)]
}

export function getStatusBadgeClass(status: string): string {
  return getStatusToken(status).badgeClass
}

export function getStatusDotClass(status: string): string {
  return getStatusToken(status).dotClass
}

export function isTransientStatus(status: string): boolean {
  return transientStatuses.has(status.toLowerCase())
}

export function getLogLevelChipClass(level: string): string {
  switch (level.toLowerCase()) {
    case 'error':
    case 'fatal':
      return getStatusChipClass(level)
    case 'warning':
    case 'warn':
      return statusTokens.warning.chipClass
    case 'info':
      return statusTokens.primary.chipClass
    default:
      return chipStyles.neutral
  }
}

export function getStatusChipClass(status: string): string {
  return getStatusToken(status).chipClass
}

export function getScoreChipClass(score: number): string {
  if (score >= 0.8) return chipStyles.primary
  if (score >= 0.6) return chipStyles.outline
  return chipStyles.neutral
}
