/**
 * Public API types. Generated from the backend's pydantic models by
 * `npm run gen:api-types` (see parallax_maker/api/schemas.py); CI fails if
 * `generated.ts` is stale. Only aliases for client-side naming live here.
 */

import type { Mode as GeneratedSegmentationMode } from './generated';

export type {
  AssetRef,
  BusyView,
  ErrorBody as ApiErrorBody,
  HealthView as HealthResponse,
  ImageSize,
  JobView as Job,
  LogEntryView as LogEntry,
  LogsView as LogsPage,
  MultiPointRequest,
  ProjectAssets,
  ProjectView,
  SegmentationClickRequest,
  SegmentationPoint,
  SegmentationView,
  SelectionRequest,
  SliceView,
  Status as JobStatus,
} from './generated';

/**
 * Matches `SegmentationClickRequest.mode` ("depth" | "instance").
 *
 * Note: `SegmentationView.slicePixel` comes through from `generated.ts` as
 * `[unknown, unknown] | null` (`json-schema-to-typescript`'s rendering of a
 * pydantic `tuple[int, int]`); nothing in the frontend reads it yet, so it
 * is left as-is here rather than narrowed.
 */
export type SegmentationMode = GeneratedSegmentationMode;
