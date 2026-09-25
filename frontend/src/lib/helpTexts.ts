/**
 * Dash's own contextual help texts, copied verbatim from
 * `parallax_maker/assets/scripts/utility.js`'s `setupHelper`/`helpTexts`
 * (JS-06). Dash picks one at random per idle-timeout popup; HelpTooltip.svelte
 * shows the whole list at once instead (see its own doc comment for why).
 */

export const SEGMENTATION_HELP_TEXTS: string[] = [
  'Click on the element in the image to create a mask for it',
  'Control-click to remove the element from the mask',
  'Shift-click to add the element to the mask',
];

export const INPAINTING_HELP_TEXTS: string[] = [
  'Drag Alt + Right-click to adjust brush size',
  'Use the mouse wheel to zoom in and out of the image',
  'You can clean up the image by panting over the areas you want to remove and ' +
    'pressing the erase button to remove them',
];

export const EXPORT_HELP_TEXTS: string[] = [
  'Click on the export button to download a glTF scene for Blender',
  'Using mesh displacement will give the scene more 3D depth. Try it out!',
  'Upscaling the textures will allow you to zoom in more on the scene.',
];

export const CONFIGURATION_HELP_TEXTS: string[] = [
  'State is saved automatically in a new folder in the directory you are running ' +
    'Parallax Maker in. Eventually, save state will give you a zip file.',
  'You can restore your state by clicking the load button and ' +
    'navigating to the directory on your local machine where the tool is running.',
];
