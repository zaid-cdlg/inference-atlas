# DESIGN.md: inference-atlas

This page is a tool (app UI). It is not a landing page. The layout serves one task: getting a
self-host vs API answer and a command you can run. The style is pencil on warm paper, with one
coral accent.

## Colour tokens (CSS variables on `:root`)

| Token | Value | Use | Contrast |
|---|---|---|---|
| `--bg` | `#F5F5F4` | page background | n/a |
| `--surface` | `#FFFFFF` | command card, inputs | n/a |
| `--ink` | `#1C1B19` | headings, body text | 15.8:1 on bg |
| `--muted` | `#6B6A66` | meta text, axis labels | 5.0:1 on bg |
| `--hairline` | `#E4E2DE` | section dividers, input borders | decorative |
| `--coral` | `#E8735A` | fills only: KV blocks, chart area, crossing dot | not for text or lines |
| `--coral-ink` | `#B8472F` | coral text, chart lines, focus ring, links | 5.3:1 on white, 4.8:1 on bg |
| `--code-bg` | `#1C1B19` | command block | white text 17:1 |

Rule: coral appears once per view as the accent. `--coral` is never used for text or 1px lines.
The theme is light only (`color-scheme: light`). Dark mode is planned for later.

## Type

- **IBM Plex Sans** (400, 600) for UI and text. **IBM Plex Mono** (400) for the command and
  numeric readouts. Both are self-hosted woff2 files in `assets/fonts/` (SIL OFL). No external
  font CDN.
- Scale in px: 14 (meta), 16 (body, minimum), 20 (section headings), 28 (verdict on phone),
  40 (verdict on desktop).
- `font-variant-numeric: tabular-nums` on every number.
- Headings get more space above than below.

## Space, shape, depth

- 4px base grid: 4, 8, 12, 16, 24, 32, 48, 64.
- Radius: 8px for inputs and buttons, 12px for the command card. Nothing else is rounded.
- One shadow, on the command card only: `0 2px 8px rgba(28,27,25,.08)`.
- Sections are separated by whitespace and a `--hairline` rule. There are no cards around
  sections.

## Layout (single column, max width 880px)

1. Header: `inference-atlas` wordmark and a GitHub link.
2. Controls: model (searchable `<input list>`), use case (Chat / Agents / Batch segmented
   buttons), GPU, precision. There are 4 in a row on desktop and a 2x2 grid under 640px.
3. Verdict: "Self-hosting likely wins above ~40M tokens/day" (2 significant figures), with the
   subline "First-order estimate. Check with `vllm bench serve`." and a "How we estimate" link.
   The break-even chart sits below.
4. Command card: `vllm serve …` with "Copy command" and "Copy link" buttons.
5. KV-cache sim: "Why N users fit".
6. Tune assumptions (collapsed `<details>`): KV dtype, input:output ratio, cached prefix %,
   utilization, batch override. Quality and quantization table.
7. Footer: "as of <date>" data stamp, MIT, "By Zaid Ibrahim".

Hero illustration: one pencil drawing of a GPU drawn like a water dispenser, filling with coral
user blocks. It is used in the header area and the README. No other illustrations are needed
for v0.1.

## Components

`Picker` (native select or input list, label always visible) · `Segmented` (3 buttons,
`aria-pressed`) · `Verdict` (`aria-live="polite"`) · `BreakEvenChart` (inline SVG, log x-axis,
`--coral-ink` lines, `--coral` crossing dot, hidden data table) · `CommandCard` · `KvSim` ·
`Details` (native `<details>`) · `Notice` (info, warning or error line with one action).

## Motion

One authored moment only. The KV sim fills once on first view (about 2.5s, ease-out,
no loop). Under `prefers-reduced-motion` it renders the final state with no motion. Nothing else
animates except the 150ms colour transitions on controls.

## Browser surfaces

Themed from the palette: `::selection` uses `--coral` at 25%, `caret-color` and `:focus-visible`
use a 2px `--coral-ink` outline with a 2px offset. Links are underlined with a 3px offset and
visited links use `--muted`.

## Responsive

- Under 640px: 2x2 picker grid with 44px targets. The verdict is 28px. The chart is 200px tall
  with 3 x-axis labels and tap-to-read values. The command block scrolls horizontally inside
  itself, and its copy button stays visible. The KV sim is one horizontal bar. The page never
  scrolls horizontally.
- 640 to 1023px: two-column controls, chart 260px.
- 1024px and up: full layout, chart 320px.
