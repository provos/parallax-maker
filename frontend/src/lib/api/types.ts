/**
 * Public API types. Generated from the backend's pydantic models by
 * `npm run gen:api-types` (see parallax_maker/api/schemas.py); CI fails if
 * `generated.ts` is stale. Only aliases for client-side naming live here.
 */

// Note: json-schema-to-typescript names positional type aliases by
// definition order, not by field name, so which generated alias ("Mode",
// "Mode1", ...) corresponds to which model can shift on regeneration; always
// re-check `generated.ts` after `npm run gen:api-types` rather than assuming
// these numbers stay stable.
import type { Mode as GeneratedInpaintingMode, Mode1 as GeneratedSegmentationMode } from './generated';

export type {
  AnimationExportRequest,
  AssetRef,
  BusyView,
  CameraSettingsRequest,
  CameraSettingsView,
  ErrorBody as ApiErrorBody,
  GltfExportRequest,
  HealthView as HealthResponse,
  ImageSize,
  InpaintingApplyRequest,
  InpaintingCandidatesView,
  InpaintingGenerateRequest,
  InpaintingPromptsRequest,
  InpaintingSelectionRequest,
  InpaintingSettingsRequest,
  InpaintingView,
  JobView as Job,
  LogEntryView as LogEntry,
  LogsView as LogsPage,
  MultiPointRequest,
  ProbeResultView,
  ProbeServerRequest,
  ProjectAssets,
  ProjectExportsView,
  ProjectSettingsRequest,
  ProjectSettingsView,
  ProjectView,
  SegmentationClickRequest,
  SegmentationPoint,
  SegmentationView,
  SelectionRequest,
  SliceView,
  Status as JobStatus,
  ValidateKeyRequest,
} from './generated';

/** Matches `InpaintingGenerateRequest.mode` ("paint" | "fill" | "enhance"). */
export type InpaintingGenerateMode = GeneratedInpaintingMode;

/**
 * Matches `SegmentationClickRequest.mode` ("depth" | "instance").
 *
 * Note: `SegmentationView.slicePixel` comes through from `generated.ts` as
 * `[unknown, unknown] | null` (`json-schema-to-typescript`'s rendering of a
 * pydantic `tuple[int, int]`); nothing in the frontend reads it yet, so it
 * is left as-is here rather than narrowed.
 */
export type SegmentationMode = GeneratedSegmentationMode;
