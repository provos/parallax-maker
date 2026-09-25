<script lang="ts">
  /**
   * The scene seen from the side (``ProjectView.sceneProfile``): the camera
   * with its field of view and pitch, the cards as upright lines at their
   * depths, and the ground (with its far backdrop). World y points down, as
   * in SVG, so no flip is needed; z runs left to right.
   */
  import type { SceneProfileView } from '../../api/types';

  let { profile }: { profile: SceneProfileView } = $props();

  const PAD = 0.06;

  const bounds = $derived.by(() => {
    const zs = [profile.cameraZ, ...profile.cards.map((c) => c.z)];
    const ys = [0, ...profile.cards.flatMap((c) => [c.top, c.bottom])];
    if (profile.ground) {
      zs.push(profile.ground.nearZ, profile.ground.farZ);
      ys.push(profile.ground.height);
      if (profile.ground.backdropTop !== null && profile.ground.backdropTop !== undefined) {
        ys.push(profile.ground.backdropTop);
      }
    }
    const minZ = Math.min(...zs);
    const maxZ = Math.max(...zs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const padZ = (maxZ - minZ) * PAD || 1;
    const padY = (maxY - minY) * PAD || 1;
    return { x: minZ - padZ, y: minY - padY, w: maxZ - minZ + 2 * padZ, h: maxY - minY + 2 * padY };
  });

  // Frustum edges and optical axis from the camera out to the farthest depth.
  const rays = $derived.by(() => {
    const far = bounds.x + bounds.w - profile.cameraZ;
    const ray = (degreesUp: number) => {
      const angle = (degreesUp * Math.PI) / 180;
      return { z: profile.cameraZ + far, y: -far * Math.tan(angle) };
    };
    return {
      top: ray(profile.pitch + profile.halfFov),
      bottom: ray(profile.pitch - profile.halfFov),
      axis: ray(profile.pitch),
    };
  });

  const stroke = $derived(Math.max(bounds.w, bounds.h) / 250);
</script>

<svg
  class="side-view"
  data-testid="scene-side-view"
  viewBox={`${bounds.x} ${bounds.y} ${bounds.w} ${bounds.h}`}
  preserveAspectRatio="xMidYMid meet"
  role="img"
  aria-label="Side view of the scene: camera, cards and ground"
>
  <line class="horizon" x1={profile.cameraZ} y1="0" x2={bounds.x + bounds.w} y2="0" stroke-width={stroke} />
  <polygon
    class="frustum"
    points={`${profile.cameraZ},0 ${rays.top.z},${rays.top.y} ${rays.bottom.z},${rays.bottom.y}`}
  />
  <line class="axis" x1={profile.cameraZ} y1="0" x2={rays.axis.z} y2={rays.axis.y} stroke-width={stroke} />
  {#if profile.ground}
    <line
      class="ground"
      data-testid="side-view-ground"
      x1={profile.ground.nearZ}
      y1={profile.ground.height}
      x2={profile.ground.farZ}
      y2={profile.ground.height}
      stroke-width={stroke * 3}
    />
    {#if profile.ground.backdropTop !== null && profile.ground.backdropTop !== undefined}
      <line
        class="ground"
        x1={profile.ground.farZ}
        y1={profile.ground.backdropTop}
        x2={profile.ground.farZ}
        y2={profile.ground.height}
        stroke-width={stroke * 2}
      />
    {/if}
  {/if}
  {#each profile.cards as card (card.index)}
    <line
      class="card"
      data-testid="side-view-card"
      x1={card.z}
      y1={card.top}
      x2={card.z}
      y2={card.bottom}
      stroke-width={stroke * 2}
    />
  {/each}
  <circle class="camera" cx={profile.cameraZ} cy="0" r={stroke * 4} />
</svg>

<style>
  .side-view {
    display: block;
    width: 100%;
    height: 9rem;
    background-color: var(--color-surface-muted);
    border-radius: var(--radius-md);
  }

  .frustum {
    fill: var(--color-accent);
    opacity: 0.12;
  }

  .axis,
  .horizon {
    stroke: var(--color-text-muted);
    stroke-dasharray: 4 3;
    vector-effect: non-scaling-stroke;
  }

  .card {
    stroke: var(--color-text);
    opacity: 0.7;
  }

  .ground {
    stroke: var(--color-success);
  }

  .camera {
    fill: var(--color-accent);
  }
</style>
