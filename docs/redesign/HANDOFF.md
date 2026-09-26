# Parallax Maker UI redesign — implementation handoff

This folder is the handoff for rebuilding the Svelte 5 frontend (`frontend/`) around a new layout. The backend API and `ProjectView` do not change unless a section below says otherwise.

- **Interactive prototype:** `prototype/Prototype.dc.html` is its source. It is a design-canvas file, not Svelte: the markup is HTML, and the behavior sits in the `class Component` script at the bottom. Read it for layout, states and copy. Do not port it line for line. Its images point at `/_blob/...` URLs that only load on the design canvas. Every image, mask and "AI" result in it is a placeholder.
- **Design tokens:** `tokens.css` replaces the `:root` / `:root.dark` blocks in `frontend/src/app.css`.
- **Live canvas (owner-only link):** https://claude.ai/artifact/AbtMWr5H9m2BUyPft25Leo — the board "Interactive prototype · press Play". Its "Prototype states ▾" menu in the status bar jumps to each of the 7 required states. That menu is a demo aid, not a product feature.

## 1. Goals (from the brief)

1. The main loop (select slice → fix mask → inpaint → preview) needs no tab switching. The layer list is always visible, and the selected slice drives a context-sensitive Inspector.
2. The image canvas dominates. View modes: Input, Depth, Slice, Composite, Parallax 2D, 3D. Canvas tools show only when they apply.
3. A workflow stepper shows progress and next steps for new users. Keyboard shortcuts serve experienced users.
4. Rare settings live in a Settings dialog. Export lives in its own dialog.
5. Inline job progress and toasts replace the log as primary feedback. The log becomes an optional drawer.
6. Dark-first, with a light mode. Primary and secondary actions are obvious: amber for the one main action, blue for selection.
7. **Segment Anything (SAM) is the main way to create slices.** Depth-threshold splitting stays as a secondary path.
8. **Leave room for outpainting** (growing a slice beyond the image edges). It is not built yet; section 7 describes the slot the UI reserves for it.

Constraints: no feature may be lost, only moved. The app fills the viewport with no page scroll, and panels scroll internally. It must work from 1280×800 up.

## 2. Layout

```
┌ AppHeader 48px ── logo · project name │ WorkflowStepper (1–7) │ undo redo theme settings [Export…] ┐
├ LayerPanel 300px ─┬ CanvasArea (flex) ─────────────────────────┬ Inspector 328px ────────────────┤
│ header + actions  │ ViewModeBar 40px (view tabs · zoom · checker) │ SelectedSliceHeader             │
│ DepthRuler + list │ ToolOptionsBar 44px (options for active tool) │ step section(s), scrolls         │
│                   │ Canvas (CanvasToolbar floats at left)       │                                  │
│ Input / Depth rows│  SelectionBar / JobCard / Toasts float over │ NextStepCard (pinned bottom)    │
├───────────────────┴─────────────────────────────────────────────┴─────────────────────────────────┤
│ LogDrawer 180px (optional, toggled)                                                               │
├ StatusBar 28px ── active job + progress + Cancel │ last message │ Log · N │ Shortcuts ?             ┤
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

At 1280px wide the canvas column gets 652px. That is enough; do not collapse panels above 1280.

## 3. Component map (Svelte names)

Put new components under `frontend/src/lib/components/<area>/`. "Replaces" names the component being retired or absorbed.

| Component | Responsibility | Replaces / absorbs |
|---|---|---|
| `AppShell` | CSS grid above; owns `data-theme` on `<html>` | `App.svelte` layout, `MainTabs` |
| `AppHeader` | Logo, project name, `WorkflowStepper`, undo/redo (selected slice), theme toggle, Settings, Export (primary) | `shell/Header` |
| `WorkflowStepper` | Steps 1–7: Image, Depth, Slices, Inpaint, Ground, Preview, Export. Each is done, current or todo. Clicking a step sets `uiStore.step` plus that step's default view and tool (§4). Export opens `ExportDialog` | `MainTabs` |
| `LayerPanel` | Header (count, Copy, Paste, Delete), `DepthRuler`, `LayerList`, pinned "Input image" and "Depth map" rows that switch the view | slice grid in `SegmentationTab` |
| `DepthRuler` | Vertical 0–255 axis, one draggable handle per slice at its depth, a curve linking each handle to its row. The ground slice's handle is outlined green and the selected one is blue. Drag calls `setSliceDepth` on mouseup | slice depth editing |
| `LayerList` / `LayerRow` | Sorted nearest first. Thumbnail, file name (`image_slice_N`), depth chip, badges (GROUND, inpainted). Click selects the slice | slice thumbnails |
| `ViewModeBar` | View tabs, checkerboard toggle, zoom −/%/+/Fit | `ViewerTabs`, zoom buttons in `InputImagePanel` |
| `CanvasToolbar` | Tools: Pan (H), Segment (S), Inpaint brush (B), Extend (O, planned), Horizon (G). Hidden in Parallax 2D and 3D | tool buttons under the image |
| `ToolOptionsBar` | Options for the active tool only (§5) | `MaskToolbar`, Invert/Feather/Multi row, camera row |
| `Canvas` | Renders the current view and overlays: SAM mask and points, brush strokes and cursor, horizon line, extend frame, candidate hover preview | `InputImagePanel`, `MaskCanvas`, `HorizonOverlay`, `PreviewOverlay`, `Model3DViewer` (3D view) |
| `SelectionBar` | Floats under the image while SAM has a mask: summary, Invert, Feather, "Segment N points" (multi-point), Add to / Remove from selected slice, **New slice** (primary, Enter) | Create/Add/Remove, Invert, Feather, Commit |
| `Inspector` | `SelectedSliceHeader` plus the section(s) for `uiStore.step` (§4), then `NextStepCard` | the five tab bodies |
| `InpaintPanel` | Mode switch Fill holes / Extend edges (planned), numbered sub-steps, prompts, sliders, Generate (primary until candidates exist), Fill/Enhance/Erase, `JobCard`, `CandidateGrid`, Apply (primary once one is picked) | `InpaintingTab` |
| `SegmentPanel` | SAM instructions with 3 sub-steps, selection summary, New slice; collapsible "Split by depth" (number of slices, Split, Balance); "Selected slice" (depth, ground toggle, replace image) | `SegmentationTab`, `ModeTab` mode selector |
| `GroundPanel` | Ground slice radio list (+ "No ground plane"), Fit ground, horizon row and pitch readout, ground distance, `SceneSideView` | Ground/Fit ground buttons, horizon readout, ground section of `ExportTab` |
| `PreviewPanel` | 2D/3D switch, camera distance, max distance, focal length | camera sliders in `ExportTab` |
| `ExportDialog` | Two sections: 3D scene (camera settings mirrored, mesh displacement, depth of field, ground summary with Edit, Upscale textures, Create → Download glTF) and Animation (number of frames, Render). Inline progress | `ExportTab` |
| `SettingsDialog` | Sections with left nav: Inpainting (model, server address + Test, ComfyUI workflow upload, API key + Test), Depth (model), Masks (padding, blur), Slicing (default number of slices), Project (Load/Save state), Appearance (theme) | `ConfigurationTab`, depth model select |
| `EmptyState` | Drop zone, "Choose image…", "try the example image", "load a saved project…"; the Inspector shows the 7-step Getting started list | empty `InputImagePanel` |
| `JobCard` | Label, %, detail line, Cancel. Rendered inline next to the control that started the job, and mirrored in `StatusBar` | `ActivityIndicator`, per-tab progress bars |
| `ToastStack` | Top-right of the canvas. Success and info dismiss after 3s; errors persist and carry actions (Retry, Open settings, View log) | log-as-feedback |
| `LogDrawer` | The existing log, toggled from the status bar or the <code>`</code> key | `LogPanel` |
| `ShortcutsDialog` | Opened with `?`; lists §6 | `HelpTooltip` (fold its help texts into step sections and hint lines) |

## 4. UI state and step → defaults

New UI-only state in `uiStore` (nothing goes to the backend): `step`, `tool`, `view`, `inpaintMode: 'holes' | 'extend'`, `segmentBy: 'object' | 'depth'`, `logOpen`, `exportOpen`, `settingsOpen`, `shortcutsOpen`, `toasts[]`, `hoverCandidate`, `zoom`.

Choosing a step sets defaults. Users can change view and tool freely afterwards.

| Step | View | Tool | Inspector section |
|---|---|---|---|
| Image | Input | Pan | Input image (replace), Project (load/save state) |
| Depth | Depth | Pan | Depth model select, Regenerate depth map |
| Slices | Input | Segment | `SegmentPanel` |
| Inpaint | Slice | Brush (or Extend) | `InpaintPanel`; selects the farthest slice if none is selected |
| Ground | Composite | Horizon | `GroundPanel` |
| Preview | Parallax 2D | — | `PreviewPanel` |
| Export | (unchanged) | — | opens `ExportDialog` |

Choosing a tool also moves the step: Segment → Slices, Brush/Extend → Inpaint, Horizon → Ground. Views Parallax 2D and 3D → Preview.

Stepper "done" rules: Image = image loaded; Depth = depth map exists; Slices = at least one slice; Inpaint = any slice has had a candidate applied (track client-side if `ProjectView` lacks a flag); Ground = a ground slice exists; Preview = user opened Parallax or 3D this session; Export = an export finished this session.

After a load, the app auto-advances: when depth finishes it goes to **Slices** with the Segment tool active, and a toast says "Depth map ready — click objects to cut them into layers, or split by depth."

## 5. Canvas tools and their options bar

- **Segment (S).** The options bar holds **Select by: Object | Depth band** (default Object), Multi-point, and Clear points (Esc).
  - The backend's `clickSegmentation(x, y, mode, shiftKey, ctrlKey)` keeps its signature: Object maps to `mode='instance'`, Depth band to `'depth'` (this replaces the Mode Selector).
  - Shift-click adds to the mask; Ctrl/Cmd-click subtracts. The prototype used Alt; implement Ctrl/Cmd to match the backend and today's help text, and accept Alt as an alias.
  - In multi-point mode, clicks queue points (`setMultiPointMode`, queued markers) and "Segment N points" calls `commitMultiPoint`.
  - While a click job runs, show a small "Segmenting…" chip over the canvas.
  - The `SelectionBar` actions map to: New slice → `createSlice`, Add to → `addMaskToSlice`, Remove from → `removeMaskFromSlice`, Invert → `invertMask`, Feather → `featherMask`.
- **Brush (B).** The options bar holds Paint/Erase (X toggles), Size, Load saved mask, Save mask, Clear. It keeps `MaskCanvas` behavior and the ROI box. The cursor becomes a brush-size ring.
- **Extend (O, planned).** The options bar holds a "±96 px sides" preset and Reset. The canvas draws a dashed frame for the requested margins. See §7.
- **Horizon (G).** Only on Composite, Input or Depth. Drag the line to call `setHorizonRow`. The options bar shows row and pitch, plus Fit ground.
- **Parallax 2D.** No toolbar. The options bar is the camera pad (← ↑ ↓ →, forward, back, Reset) via `navigateCamera`. Moving the pointer over the image may drive a *client-side* preview offset; that is optional polish, and the real camera still moves through `navigateCamera`.
- **3D.** `Model3DViewer`. The options bar shows the camera pad and an orbit hint.
- **Pan (H)** and zoom: existing zoom/pan/reset, moved to `ViewModeBar`.

## 6. Keyboard shortcuts

Register them on the app root, not `window`. Ignore them while focus is in a text field, except Ctrl+Enter in the prompt fields.

| Key | Action | Key | Action |
|---|---|---|---|
| S / B / X / O / G / H | Segment / Brush / toggle erase / Extend / Horizon / Pan | Enter | New slice from selection |
| Shift+Enter / Alt+Enter | Add to / Remove from selected slice | [ / ] | Previous / next layer (by depth) |
| Ctrl+Enter | Generate candidates | 1 2 3 | Pick candidate |
| A | Apply picked candidate | I M L C P, Shift+P | Views: Input, depth Map, sLice, Composite, Parallax, 3D |
| Ctrl+Z / Ctrl+Shift+Z | Undo / redo on selected slice (`undoSlice` / `redoSlice`) | Delete | Delete selected slice |
| Ctrl+E / Ctrl+, | Export / Settings | <code>`</code> / ? / Esc | Log / Shortcuts / clear points or close dialog |

## 7. Outpainting slot (planned, needs backend)

The UI reserves space so outpainting slots in without a relayout:

- `InpaintPanel` has a mode switch, **Fill holes | Extend edges** (the latter badged PLANNED). Both modes share the prompt, negative prompt, strength, guidance, Generate, the 3-candidate grid and Apply.
- Extend edges swaps sub-step 1 ("Paint the holes") for four margin sliders (left, right, top, bottom; 0–192 px, step 8). The canvas shows the grown frame. Candidates preview in that frame on hover.
- Needed from the backend: an endpoint such as `POST /api/v1/projects/{id}/slices/{index}/outpaint` with margins and the same prompt fields, returning candidates the way inpainting does. Slices also need a stored size or offset, so composite, glTF export and animation handle slices larger than the input. Until that exists, render Extend edges disabled with a "Coming soon" tooltip, or behind a feature flag.

## 8. Feature parity checklist

Every current control and where it goes. Keep existing `data-testid`s on the moved controls; the e2e driver depends on them.

| Today (tab / place) | New home |
|---|---|
| Mode: depth image, Depth Module Algorithm, Regenerate Depth Map | Depth step (view Depth + Inspector); model also in Settings › Depth |
| Mode: Mode Selector (depth / instance) | Segment tool option **Select by: Object / Depth band** |
| Segmentation: threshold sliders, Generate, Balance | Slices › Split by depth (collapsible). Keep threshold handles: show them on a thin bar in that section, or as amber ticks on the DepthRuler |
| Segmentation: Create, Delete, Add, Remove, Copy, Paste | SelectionBar (Create/Add/Remove), LayerPanel header (Copy/Paste/Delete) |
| Segmentation: Ground, Fit ground | Ground step (`GroundPanel`); ground toggle also in Slices › Selected slice |
| Segmentation: slice thumbnails, depth edit, undo/redo, download | LayerList + DepthRuler; header undo/redo; Inspector header download; depth input in Selected slice |
| Slice image upload | Slices › Selected slice › Replace image… |
| Under image: zoom −/reset/+, level | ViewModeBar |
| Under image: camera pad (7 buttons) | ToolOptionsBar in Parallax 2D / 3D |
| Under image: Horizon toggle + drag | Horizon tool (G) |
| Under image: checkerboard | ViewModeBar |
| Under image: Invert, Feather, Multi, Commit | SelectionBar / Segment options |
| Under image: Clear, Erase, Load, Brush size | Brush options (+ new Save mask → `saveMask`) |
| Inpainting: prompts, strength, guidance, crop to ROI, Generate, Fill, Enhance, Erase, candidates, Apply | `InpaintPanel` |
| Export: Create glTF, Export glTF, Upscale Textures, DoF, mesh displacement | ExportDialog › 3D scene |
| Export: camera distance, max distance, focal length | PreviewPanel (mirrored in ExportDialog) |
| Export: Ground distance, horizon readout, side view | GroundPanel |
| Export: Export Animation, Number of Frames | ExportDialog › Animation |
| Configuration: number of slices | Slices › Split by depth (+ Settings › Slicing default) |
| Configuration: inpainting model, server address + Test, ComfyUI workflow, API key + Test, mask padding/blur | SettingsDialog |
| Configuration: Load State / Save State | Image step › Project; SettingsDialog › Project; EmptyState link |
| 2D / 3D viewer tabs | View modes Parallax 2D / 3D |
| Header theme toggle | AppHeader + Settings › Appearance |
| Help "?" tooltips | Numbered sub-step hints + ShortcutsDialog |
| Log panel | LogDrawer + StatusBar last message |

## 9. Feedback rules

- Every long job (depth, segmentation click, split, inpainting, fill/enhance/erase, glTF, upscale, animation) shows a `JobCard` next to the control that started it. The job is mirrored in the StatusBar with % and Cancel.
- The first depth run shows "Downloading model weights (first run only)" when the job reports it.
- Only one job runs at a time (single worker). While one runs, disable buttons that would start another.
- Errors: a persistent toast with actions, *and* an inline error card where the job was started (e.g. in `InpaintPanel`: "Generation failed · Could not reach ComfyUI at … · Retry · Open settings · View log"). The StatusBar message turns red.
- Success: short toast plus a log line. No modal confirmations.

## 10. Visual system

- Tokens: `tokens.css` (dark default via `[data-theme="dark"]`, light via `[data-theme="light"]`). Remove the dashed `.panel` border style entirely.
- Type: IBM Plex Sans (UI), IBM Plex Mono (numbers, file names, depth values). 13px base, 11px uppercase section labels.
- Buttons: 32px default, 26px small. **Primary** (amber fill, dark text): exactly one per context (Export…, New slice, Generate, then Apply once candidates exist). **Secondary** (neutral fill); **ghost** (text only) for tertiary actions; **selected** state is blue.
- Selection is always blue (layer row, candidate, active tool, SAM mask tint). Ground is green. The horizon is orange. The inpaint mask is translucent red-orange.
- Icons: 16px stroke icons, 1.6 stroke width (Lucide fits if you want a library).

## 11. Backend questions and gaps

1. **Outpainting:** new endpoint plus slice size/offset (§7).
2. **Rest-of-image slice:** when the first SAM slice is created, the prototype also adds an `image_slice_N` holding everything not yet segmented, so a background exists. Either add this to `createSlice` (flag) or keep requiring Split by depth first. Decide before building `SegmentPanel`.
3. **Cancel:** Cancel buttons assume `DELETE /api/v1/jobs/{id}` or similar. If jobs can't be cancelled, hide Cancel.
4. **Inpainted flag per slice:** optional, for the stepper and the badge. Client-side tracking is fine at first.

## 12. Suggested order of work

1. Tokens + `AppShell` + header/stepper + status bar; keep the old tab bodies temporarily inside the Inspector so everything still works.
2. `LayerPanel` with `DepthRuler`; delete the slice grid from the Segmentation tab.
3. `Canvas`, `ViewModeBar`, `CanvasToolbar`, `ToolOptionsBar`; move the under-image controls.
4. `SegmentPanel` + `SelectionBar` (SAM-first); Split by depth secondary.
5. `InpaintPanel` (holes mode; Extend stubbed), `GroundPanel`, `PreviewPanel`.
6. `ExportDialog`, `SettingsDialog`, toasts, `JobCard`, `LogDrawer`, shortcuts.
7. Update `e2e/drivers/svelte.ts`. `openTab(MainTab)` must map to the new steps/dialogs (Mode → Depth step, Segmentation → Slices, Inpainting → Inpaint, Export → ExportDialog, Configuration → SettingsDialog), and `withModeTabVisible` (Mode Selector) → the Segment tool's Select-by switch. Run `npm run check:frontend`, `npm run test:frontend` and `npm run test:e2e`.
