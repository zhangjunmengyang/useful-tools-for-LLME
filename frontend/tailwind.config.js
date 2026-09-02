/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        border: {
          DEFAULT: 'hsl(var(--border))',
          soft: 'hsl(var(--border-soft))',
        },
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: {
          DEFAULT: 'hsl(var(--background))',
          subtle: 'hsl(var(--background-subtle))',
        },
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
          hover: 'hsl(var(--primary-hover))',
          muted: 'hsl(var(--primary-muted))',
          // 黄色文字/描边专用：填充走 DEFAULT，要被「读」的走 ink。
          // 币安黄对白底只有 1.8:1，text-primary 会读不出来。
          ink: 'hsl(var(--primary-ink))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
          muted: 'hsl(var(--destructive-muted))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        // Status colors with muted variants
        success: {
          DEFAULT: 'hsl(var(--success))',
          foreground: 'hsl(var(--success-foreground))',
          muted: 'hsl(var(--success-muted))',
        },
        warning: {
          DEFAULT: 'hsl(var(--warning))',
          foreground: 'hsl(var(--warning-foreground))',
          muted: 'hsl(var(--warning-muted))',
        },
        info: {
          DEFAULT: 'hsl(var(--info))',
          foreground: 'hsl(var(--info-foreground))',
          muted: 'hsl(var(--info-muted))',
        },
        // 行情方向。与 success/destructive 同值但语义不同：
        // 这两个表示涨跌，success/destructive 表示操作结果，不能混用。
        up: {
          DEFAULT: 'hsl(var(--up))',
          muted: 'hsl(var(--up-muted))',
        },
        down: {
          DEFAULT: 'hsl(var(--down))',
          muted: 'hsl(var(--down-muted))',
        },
        // Sidebar colors
        sidebar: {
          DEFAULT: 'hsl(var(--sidebar-background))',
          foreground: 'hsl(var(--sidebar-foreground))',
          border: 'hsl(var(--sidebar-border))',
          accent: 'hsl(var(--sidebar-accent))',
          'accent-foreground': 'hsl(var(--sidebar-accent-foreground))',
          primary: 'hsl(var(--sidebar-primary))',
          'primary-foreground': 'hsl(var(--sidebar-primary-foreground))',
        },
      },
      borderRadius: {
        lg: 'var(--radius-lg)',
        md: 'var(--radius)',
        sm: 'var(--radius-sm)',
        xl: 'var(--radius-xl)',
      },
      fontFamily: {
        sans: [
          'Archivo',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Noto Sans SC',
          'ui-sans-serif',
          'sans-serif',
        ],
        serif: ['Source Serif Pro', 'Georgia', 'serif'],
        mono: ['JetBrains Mono', 'SF Mono', 'Menlo', 'Consolas', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        // 使用 CSS 变量承载平台固定字号层级
        '2xs': 'var(--text-2xs)',
        'xs': 'var(--text-xs)',
        'sm': 'var(--text-sm)',
        'base': 'var(--text-base)',
        'lg': 'var(--text-lg)',
        'xl': 'var(--text-xl)',
        '2xl': 'var(--text-2xl)',
        '3xl': 'var(--text-3xl)',
        '4xl': 'var(--text-4xl)',
      },
      // 课程阅读页正文排版。中文长文口径：行宽按每行汉字数而非 ch 计，
      // 行高高于西文，段间距代替首行缩进。只在 .prose-lesson 下生效。
      typography: {
        lesson: {
          css: {
            '--tw-prose-body': 'hsl(var(--foreground))',
            '--tw-prose-headings': 'hsl(var(--foreground))',
            '--tw-prose-bold': 'hsl(var(--foreground))',
            '--tw-prose-links': 'hsl(var(--primary))',
            '--tw-prose-counters': 'hsl(var(--muted-foreground))',
            '--tw-prose-bullets': 'hsl(var(--border))',
            '--tw-prose-quotes': 'hsl(var(--foreground))',
            '--tw-prose-captions': 'hsl(var(--muted-foreground))',
            '--tw-prose-code': 'hsl(var(--foreground))',
            '--tw-prose-hr': 'hsl(var(--border))',
            '--tw-prose-th-borders': 'hsl(var(--border))',
            '--tw-prose-td-borders': 'hsl(var(--border))',

            fontSize: 'var(--text-base)',
            lineHeight: '1.9',
            // 跑文里不要等宽数字。globals.css 的 body 开了 tnum，会把「涨 50% 变 150」
            // 这类数字撑成报表字形；这里显式覆盖，表格再单独把 tnum 加回来。
            fontFeatureSettings: '"rlig" 1, "calt" 1, "ss01" 1, "cv02" 1',
            p: {
              // base 层有 `p { @apply leading-relaxed }`，是直接声明，会压过容器继承下来的
              // 行高，所以这里必须在 p 上再写一次，否则正文行高只有 1.625。
              lineHeight: '1.9',
              marginTop: '1.3em',
              marginBottom: '1.3em',
              textWrap: 'pretty',
            },

            // 开篇的 <em> 段是引入问句，不是普通强调。
            '> p:first-child em': {
              display: 'inline-block',
              fontSize: '1.05em',
              fontStyle: 'normal',
              color: 'hsl(var(--muted-foreground))',
            },
            '> p:first-child': { marginBottom: '1.5em' },

            // 正文里唯一在用的标题层级，必须比正文可辨。
            h4: {
              fontSize: '1.06em',
              fontWeight: '600',
              lineHeight: '1.5',
              marginTop: '2em',
              marginBottom: '0.7em',
            },
            h2: { fontSize: '1.28em', fontWeight: '600', marginTop: '2.2em', marginBottom: '0.8em' },
            h3: { fontSize: '1.14em', fontWeight: '600', marginTop: '2em', marginBottom: '0.7em' },

            // 关键句高亮：底纹压到很淡，靠下划线偏移标出，不要方块。
            mark: {
              backgroundColor: 'hsl(var(--primary) / 0.12)',
              color: 'inherit',
              padding: '0.1em 0.2em',
              borderRadius: 'var(--radius-sm)',
              textDecoration: 'underline',
              textDecorationColor: 'hsl(var(--primary) / 0.45)',
              textUnderlineOffset: '0.25em',
            },

            // 每个专题结尾的「记住」收束块，视觉上是卡片不是引文。
            blockquote: {
              marginTop: '2.5em',
              marginBottom: '0.5em',
              padding: '1.1em 1.3em',
              borderLeftWidth: '3px',
              borderLeftColor: 'hsl(var(--primary))',
              borderRadius: '0 var(--radius-sm) var(--radius-sm) 0',
              backgroundColor: 'hsl(var(--primary) / 0.04)',
              fontStyle: 'normal',
              fontWeight: '400',
              quotes: 'none',
            },
            'blockquote p:first-of-type::before': { content: 'none' },
            'blockquote p:last-of-type::after': { content: 'none' },
            'blockquote p': { marginTop: '0', marginBottom: '0' },

            aside: {
              marginTop: '1.8em',
              marginBottom: '1.8em',
              padding: '1em 1.2em',
              borderWidth: '1px',
              borderColor: 'hsl(var(--border))',
              borderRadius: 'var(--radius)',
              backgroundColor: 'hsl(var(--muted) / 0.4)',
            },
            'aside > :first-child': { marginTop: '0' },
            'aside > :last-child': { marginBottom: '0' },
            'aside h4': {
              marginTop: '0',
              marginBottom: '0.5em',
              fontSize: 'var(--text-xs)',
              letterSpacing: '0.04em',
              color: 'hsl(var(--primary))',
            },

            table: {
              fontSize: '0.94em',
              lineHeight: '1.7',
              // 表格里数字要对齐，tnum 加回来。
              fontVariantNumeric: 'tabular-nums',
              fontFeatureSettings: '"rlig" 1, "calt" 1, "ss01" 1, "cv02" 1, "tnum" 1',
            },
            thead: { backgroundColor: 'hsl(var(--muted))' },
            'thead th': { padding: '0.7em 0.9em', fontWeight: '600' },
            // 纯数值列右对齐；表头跟随，避免标题和数字各靠一边。
            'thead th:not(:first-child)': { textAlign: 'right' },
            'tbody td': { padding: '0.7em 0.9em', verticalAlign: 'top' },
            'tbody td:not(:first-child)': { textAlign: 'right' },
            caption: {
              captionSide: 'top',
              padding: '0.6em 0.9em',
              textAlign: 'left',
              fontSize: 'var(--text-xs)',
              color: 'hsl(var(--muted-foreground))',
            },

            // 定义列表的标签列按内容自适应，不再写死 120px 栅格。
            dl: {
              display: 'grid',
              gridTemplateColumns: 'auto minmax(0, 1fr)',
              columnGap: '1.2em',
              rowGap: '0.7em',
              marginTop: '1.5em',
              marginBottom: '1.5em',
            },
            dt: { fontWeight: '500' },
            dd: { margin: '0', color: 'hsl(var(--muted-foreground))' },

            details: {
              borderTopWidth: '1px',
              borderTopColor: 'hsl(var(--border))',
              paddingTop: '0.9em',
              paddingBottom: '0.9em',
            },
            summary: { cursor: 'pointer', fontWeight: '500' },

            code: {
              backgroundColor: 'hsl(var(--muted))',
              padding: '0.15em 0.4em',
              borderRadius: 'var(--radius-sm)',
              fontWeight: '400',
              fontSize: '0.9em',
            },
            'code::before': { content: 'none' },
            'code::after': { content: 'none' },
            pre: {
              backgroundColor: 'hsl(var(--muted))',
              color: 'hsl(var(--foreground))',
              borderRadius: 'var(--radius)',
            },

            'li': { marginTop: '0.4em', marginBottom: '0.4em' },
            'li::marker': { color: 'hsl(var(--muted-foreground))' },
            small: { fontSize: 'var(--text-xs)', color: 'hsl(var(--muted-foreground))' },
            a: { textUnderlineOffset: '0.2em' },
            figcaption: { fontSize: 'var(--text-xs)', lineHeight: '1.6' },
            hr: { marginTop: '2.5em', marginBottom: '2.5em' },

            // KaTeX 默认把公式放大到 1.21em，行内公式会顶破中文行高，压回正文字号。
            '.katex': { fontSize: '1.04em' },
            // 行内分数是撑破行盒的主要来源，单独收一档；块级公式不受影响。
            ':not(.katex-display) > .katex .mfrac': { fontSize: '0.92em' },
            '.katex-display': {
              overflowX: 'auto',
              overflowY: 'hidden',
              marginTop: '1.6em',
              marginBottom: '1.6em',
              paddingTop: '0.2em',
              paddingBottom: '0.2em',
            },
            // 表格脚注（口径说明）紧贴表格，不要读成孤立段落。
            'table + p > small:only-child': { display: 'block', marginTop: '-0.6em' },
          },
        },
      },
      spacing: {
        '0.5': '0.125rem',  // 2px
        '1': '0.25rem',     // 4px
        '1.5': '0.375rem',  // 6px
        '2': '0.5rem',      // 8px
        '2.5': '0.625rem',  // 10px
        '3': '0.75rem',     // 12px
        '3.5': '0.875rem',  // 14px
        '4': '1rem',        // 16px
        '5': '1.25rem',     // 20px
        '6': '1.5rem',      // 24px
        '7': '1.75rem',     // 28px
        '8': '2rem',        // 32px
        '9': '2.25rem',     // 36px
        '10': '2.5rem',     // 40px
        '12': '3rem',       // 48px
        '14': '3.5rem',     // 56px
        '16': '4rem',       // 64px
      },
      boxShadow: {
        'depth-1': 'var(--shadow-1)',
        'depth-2': 'var(--shadow-2)',
        'depth-3': 'var(--shadow-3)',
        'depth-4': 'var(--shadow-4)',
        'glow-sm': '0 0 0 1px hsl(var(--primary) / 0.18), 0 8px 20px -18px hsl(var(--primary) / 0.65)',
        'glow-md': '0 0 0 1px hsl(var(--primary) / 0.2), 0 16px 34px -26px hsl(var(--primary) / 0.7)',
        'glow-lg': '0 0 0 1px hsl(var(--primary) / 0.22), 0 22px 48px -34px hsl(var(--primary) / 0.75)',
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
        'collapsible-down': {
          from: { height: '0', opacity: '0' },
          to: { height: 'var(--radix-collapsible-content-height)', opacity: '1' },
        },
        'collapsible-up': {
          from: { height: 'var(--radix-collapsible-content-height)', opacity: '1' },
          to: { height: '0', opacity: '0' },
        },
        'caret-blink': {
          '0%, 70%, 100%': { opacity: '1' },
          '20%, 50%': { opacity: '0' },
        },
        'fade-in': {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        'fade-out': {
          from: { opacity: '1' },
          to: { opacity: '0' },
        },
        'slide-up': {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-down': {
          from: { opacity: '0', transform: 'translateY(-8px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
        'collapsible-down': 'collapsible-down 0.2s ease-out',
        'collapsible-up': 'collapsible-up 0.2s ease-out',
        'caret-blink': 'caret-blink 1.25s ease-out infinite',
        'fade-in': 'fade-in 0.2s ease-out',
        'fade-out': 'fade-out 0.2s ease-out',
        'slide-up': 'slide-up 0.25s ease-out',
        'slide-down': 'slide-down 0.25s ease-out',
      },
      transitionDuration: {
        150: '150ms',
        250: '250ms',
        350: '350ms',
      },
      transitionTimingFunction: {
        'ease-spring': 'cubic-bezier(0.16, 1, 0.3, 1)',
        'ease-premium': 'cubic-bezier(0.16, 1, 0.3, 1)',
      },
    },
  },
  plugins: [require('tailwindcss-animate'), require('@tailwindcss/typography')],
}
