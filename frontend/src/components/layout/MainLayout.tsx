import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'
import { useMatches } from 'react-router-dom'
import { Header } from './Header'
import { Sidebar } from './Sidebar'
import { useAppSidebarLayout } from './useAppSidebarLayout'
import type { TopicOutline, TopicSummary } from '@/types'

interface RouteHandle {
  wide?: boolean
  fill?: boolean
}

export function MainLayout({
  children,
  topics,
  outline,
}: {
  children: ReactNode
  topics: TopicSummary[]
  outline: TopicOutline | null
}) {
  const matches = useMatches()
  const sidebarLayout = useAppSidebarLayout()
  const isWideRoute = matches.some(
    (match) => (match.handle as RouteHandle | undefined)?.wide === true,
  )
  const isFillRoute = matches.some(
    (match) => (match.handle as RouteHandle | undefined)?.fill === true,
  )

  return (
    <div className="relative flex h-dvh overflow-hidden bg-background text-foreground">
      {sidebarLayout.isDesktop && sidebarLayout.collapsed ? null : (
        <Sidebar layout={sidebarLayout} topics={topics} outline={outline} />
      )}
      <div className="relative flex min-h-0 min-w-0 flex-1 flex-col bg-background">
        <Header sidebar={sidebarLayout} outline={outline} />
        <main
          className={cn(
            'min-h-0 flex-1 bg-background',
            isFillRoute ? 'overflow-hidden' : 'overflow-y-auto',
          )}
        >
          {isFillRoute ? (
            <div className="h-full min-h-0">{children}</div>
          ) : (
            <div
              className={cn(
                'mx-auto w-full px-4 py-6 md:px-8 md:py-8 xl:px-10',
                isWideRoute ? 'max-w-[1920px]' : 'max-w-[1480px]',
              )}
            >
              {children}
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
