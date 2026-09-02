import { useI18n } from "@/components/I18nProvider";
import { TopicSwitcher } from "@/components/TopicSwitcher";
import { cn } from "@/lib/utils";
import type { TopicOutline, TopicSummary } from "@/types";
import { BookOpen, FlaskConical, GraduationCap, PanelLeftClose, Wrench } from "lucide-react";
import { type SVGProps, memo } from "react";
import { NavLink, useLocation, useParams } from "react-router-dom";
import type { AppSidebarLayout } from "./useAppSidebarLayout";

function AppLogo({ className, label, ...props }: SVGProps<SVGSVGElement> & { label: string }) {
  return (
    <svg className={className} viewBox="0 0 32 32" fill="none" role="img" aria-label={label} {...props}>
      <rect width="32" height="32" rx="7" fill="#F0B90B" />
      <g fill="#1E2026">
        <rect x="7" y="12" width="4" height="11" rx="1" />
        <rect x="8.5" y="9" width="1" height="17" rx="0.5" />
        <rect x="14" y="7" width="4" height="15" rx="1" />
        <rect x="15.5" y="4" width="1" height="21" rx="0.5" />
        <rect x="21" y="15" width="4" height="8" rx="1" />
        <rect x="22.5" y="12" width="1" height="14" rx="0.5" />
      </g>
    </svg>
  );
}

const NavItem = memo(function NavItem({
  compact,
  to,
  label,
  icon: Icon,
  active,
}: {
  compact: boolean;
  to: string;
  label: string;
  icon: typeof GraduationCap;
  active: boolean;
}) {
  return (
    <li>
      <NavLink
        to={to}
        title={compact ? label : undefined}
        aria-current={active ? "page" : undefined}
        className={cn(
          "group relative flex items-center gap-2 rounded-md px-2 py-2 text-sm font-medium transition-colors duration-200 ease-premium",
          compact ? "justify-center" : "justify-start",
          active
            ? "bg-sidebar-accent text-sidebar-accent-foreground"
            : "text-sidebar-foreground hover:bg-muted hover:text-sidebar-accent-foreground",
        )}
      >
        {compact || !active ? null : <span className="absolute inset-y-1 left-0 w-0.5 rounded-full bg-sidebar-primary" />}
        <Icon
          className={cn(
            "h-4 w-4 shrink-0 transition-colors duration-200 ease-premium",
            active
              ? "text-sidebar-accent-foreground"
              : "text-sidebar-foreground/70 group-hover:text-sidebar-accent-foreground",
          )}
        />
        {compact ? null : <span className="flex-1 truncate">{label}</span>}
      </NavLink>
    </li>
  );
});

export function Sidebar({
  layout,
  topics,
  outline,
}: {
  layout: AppSidebarLayout;
  topics: TopicSummary[];
  outline: TopicOutline | null;
}) {
  const { topicId } = useParams();
  const location = useLocation();
  const { tr } = useI18n();
  const { visualWidth, isDesktop, toggleCollapsed } = layout;
  const compact = !isDesktop;
  const lastTopic = topicId ?? outline?.id ?? topics[0]?.id ?? "omni";
  const courseHref = `/t/${lastTopic}`;
  const brand = tr("学习台", "Learn Bench");
  const navItems = [
    { to: "course" as const, label: tr("课程", "Course"), icon: GraduationCap },
    { to: "/tools" as const, label: tr("工具台", "Tools"), icon: Wrench },
    { to: "/explore" as const, label: tr("实验室", "Labs"), icon: FlaskConical },
    { to: "/notebooks" as const, label: tr("笔记本", "Notebooks"), icon: BookOpen },
  ];

  return (
    <aside
      className="relative z-10 flex h-dvh shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground"
      style={{ width: visualWidth }}
    >
      <div className="shrink-0 border-b border-sidebar-border">
        <div className={cn("flex h-20 items-center", compact ? "justify-center px-2" : "gap-2 px-4")}>
          <AppLogo className="h-7 w-7 shrink-0" label={brand} />
          {compact ? null : (
            <div className="min-w-0 flex-1">
              <span className="block truncate text-sm font-semibold text-sidebar-accent-foreground">{brand}</span>
            </div>
          )}
          {isDesktop ? (
            <button
              type="button"
              onClick={toggleCollapsed}
              aria-label={tr("收起侧栏", "Collapse sidebar")}
              aria-expanded={true}
              title={tr("收起侧栏", "Collapse sidebar")}
              className="shrink-0 rounded-md p-1.5 text-sidebar-foreground/60 transition-colors hover:bg-muted hover:text-sidebar-accent-foreground"
            >
              <PanelLeftClose className="h-4 w-4" />
            </button>
          ) : null}
        </div>
      </div>

      <nav className={cn("flex-1 overflow-auto pb-2 pt-3", compact ? "px-2" : "px-3")}>
        <div className={cn("pb-3", compact ? "flex justify-center" : "")}>
          <TopicSwitcher compact={compact} topics={topics} currentId={topicId ?? outline?.id} />
        </div>
        <div className="mb-4">
          <ul className="space-y-0.5">
            {navItems.map((item) => {
              const to = item.to === "course" ? courseHref : item.to;
              const active =
                item.to === "course"
                  ? location.pathname.startsWith("/t/")
                  : item.to === "/explore"
                    ? location.pathname.startsWith("/explore")
                    : location.pathname === item.to || location.pathname.startsWith(`${item.to}/`);
              return (
                <NavItem
                  key={item.to}
                  compact={compact}
                  to={to}
                  label={item.label}
                  icon={item.icon}
                  active={active}
                />
              );
            })}
          </ul>
        </div>
      </nav>
    </aside>
  );
}
