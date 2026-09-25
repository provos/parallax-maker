/**
 * Public API types. Generated from the backend's pydantic models by
 * `npm run gen:api-types` (see parallax_maker/api/schemas.py); CI fails if
 * `generated.ts` is stale. Only aliases for client-side naming live here.
 */

export type {
  AssetRef,
  BusyView,
  ErrorBody as ApiErrorBody,
  HealthView as HealthResponse,
  ImageSize,
  JobView as Job,
  LogEntryView as LogEntry,
  LogsView as LogsPage,
  ProjectAssets,
  ProjectView,
  SliceView,
  Status as JobStatus,
} from './generated';
