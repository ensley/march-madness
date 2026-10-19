# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Delegated by the user: a local Python server with plain HTML, CSS, and JavaScript;
no separate frontend build. Flask serves the existing Python modeling workflows.

## Users and Purpose

The repository author uses this tool to fit hand-authored Stan basketball models,
inspect team ratings, simulate matchups and tournament brackets, and evaluate
frozen forecasts against historical NCAA tournament results.

## Operating Context

This is a local research tool. Inputs live in ignored `data/`; saved fits,
simulations, diagnostics, and backtests live in ignored `output/` or `outputs/`.
The web interface reads structured artifacts rather than parsing console output.

## Capabilities and Constraints

The first version covers saved fits and backtests, team ratings, matchup and
bracket simulations, and background season fitting. Preserve the Stan equations
and existing CLI. Retain prior runs and make diagnostic warnings visible.
No public hosting, authentication, or shared-user workflow was requested.

## Product Principles

- Show real local results and identify their source.
- Make expensive work explicit and keep the interface responsive while it runs.
- Explain failures with a concrete recovery action.
- Prefer clear tables and ordinary controls to decorative dashboard elements.
