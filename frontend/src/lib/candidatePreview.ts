/**
 * The inpainting candidate being previewed: the one picked in the Inpaint
 * panel, while it belongs to the selected slice. The canvas shows it in
 * place of that slice (Slice and Composite views; the Input view's server
 * display image already is the candidate) and hides the painted mask, so
 * the filled-in area can be inspected.
 */
import type { ProjectView } from './api/types';

export type CandidatePreview = { sliceIndex: number; candidate: number; url: string };

export function candidatePreview(view: ProjectView | null | undefined): CandidatePreview | null {
  const candidates = view?.inpainting.candidates;
  const picked = view?.inpainting.selectedCandidate;
  if (!view || !candidates || picked == null) return null;
  if (candidates.sliceIndex !== view.selectedSlice) return null;
  const image = candidates.images[picked];
  return image ? { sliceIndex: candidates.sliceIndex, candidate: picked, url: image.url } : null;
}
