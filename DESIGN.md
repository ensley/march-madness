---
name: March Madness
description: Local basketball research workspace
colors:
  paper: "#faf9f6"
  surface: "#fff"
  soft: "#f0eee8"
  ink: "#242a2b"
  muted: "#626866"
  line: "#dddeda"
  accent: "#a43d17"
  accent-hover: "#813012"
  warning: "#765115"
  success: "#23634b"
  danger: "#9d2f2f"
typography:
  headline:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "30px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.03em"
  title:
    fontSize: "18px"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "-0.015em"
  body:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontSize: "13px"
    fontWeight: 600
rounded:
  input: "5px"
  control: "6px"
  panel: "8px"
spacing:
  small: "8px"
  medium: "16px"
  panel: "24px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.surface}"
    rounded: "{rounded.control}"
    padding: "10px 16px"
  button-primary-hover:
    backgroundColor: "{colors.accent-hover}"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "10px 16px"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.input}"
    padding: "10px 12px"
---

# Design System: March Madness

## Overview

**Creative North Star: "Scientific results worksheet"**

The implemented interface is a quiet local research workspace. Dense, readable
results take priority over decorative dashboard elements. Controls use ordinary
browser affordances and the same visual language across modeling tasks.

**Key Characteristics:**
- Source-aware tables with explicit units and diagnostic warnings.
- Warm neutral surfaces, dark ink, and restrained orange actions.
- Stable controls while background jobs update.

The implementation source is `src/march_madness/web/static/app.css`; preserve its
actual values when extending the system.

## Colors

Orange identifies actions and navigation selection. Green marks successful checks
and positive ratings; warning and error colors accompany explanatory text.
Paper, white, and soft neutral surfaces separate navigation, forms, and data.

## Typography

Use the system sans stack throughout, with tabular numerals in tables. Headings
use 30px/18px, body text 15px, table data 14px, and labels/help text 13px. Phone
page headings reduce to 26px. Prose is limited to 74 characters; tables may be wider.

## Layout

Desktop uses a fixed 224px sidebar and 44px content gutters. At 1100px the sidebar
reduces to 190px and gutters to 26px. At 720px navigation becomes a wrapping top
bar, forms become one column, and gutters become 20px.

Scroll wide tables and bracket rounds inside their own regions. Ratings show
25 rows per page; advancement tables and bracket panels have bounded heights.
Keep headings and relevant controls grouped, with 32px between major sections.

## Elevation & Depth

Use flat tonal surfaces and one-pixel borders. The current interface has no shadows.

## Shapes

Use compact rectangular controls with 5–6px corners and forms with 8px corners.
Avoid introducing a second component shape vocabulary.

## Components

- **Buttons:** orange primary, bordered white secondary, transparent quiet actions;
  minimum height 42px. Hover changes background over 150ms. Disabled controls
  dim; focus uses a 3px orange outline with a 3px offset.
- **Inputs:** white surfaces, one-pixel borders, visible labels, matching focus rings.
- **Navigation:** orange text on a pale orange selected surface; wrapping links on phones.
- **Tables:** sticky headers, right-aligned numbers, left-aligned team names,
  sortable ratings, and visibly explained units.
- **Jobs:** announce changed status text separately from stable result buttons.
  Preserve keyed history controls and keyboard focus across polling cycles.
- **Brackets:** game headings describe the teams playing; probability headings
  describe advancement. Keep joint draws distinct from aggregate chances.

## Do's and Don'ts

- **Do** preserve focus, loading, empty, and recoverable error states.
- **Do** show the selected data source and sampling limitations.
- **Do** support reduced motion and contained scrolling on phones.
- **Don't** replace tables with decorative metric cards.
- **Don't** rebuild unchanged live regions or focused controls during polling.
