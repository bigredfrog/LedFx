import logging

import numpy as np
import voluptuous as vol
from PIL import Image, ImageOps

from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.utils.logsec import LogSec

_LOGGER = logging.getLogger(__name__)

# TODO: test with non matching
# TODO: test with 1d stretch
# TODO: Test with missing virtuals
# TODO: Introduce time scroll in 1d stretch effects
class BlendVirtual:
    """
    Get the virtual and reshape its pixels to an RGB matrix
    If the virtual does not exist or have a frame, create a black image in the target shape 
    """
    def __init__(self, virtual_id, _virtuals_config):
        self.virtual_id = virtual_id
        self.initialised = False

        for virtual_config in _virtuals_config:
            if virtual_id == virtual_config["id"]:
                return
            
        # TODO: Create a dummy, maybe deferred call
        _LOGGER.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        _LOGGER.error(f"BlendVirtual.__init__ {virtual_id} was not found, creating")
        _LOGGER.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    def init(self, _ledfx, target_shape):
       
        _virtuals = _ledfx.virtuals._virtuals
        self.target_shape = target_shape # rows 0, columns 1
        
        self.virtual = _virtuals.get(self.virtual_id)
        self.was_active = False

        if self.virtual:
            self.rows = self.virtual.config["rows"]
            self.columns = int(self.virtual.pixel_count / self.rows)
            self.shape = (self.rows, self.columns)
            self.matching = (self.shape == target_shape)
            self.initialised = True
            
            if not self.virtual.active:
                last_effect = self.virtual.virtual_cfg.get("last_effect", None)
                if last_effect:
                    effect_config = self.virtual.get_effects_config(last_effect)
                    if effect_config:
                        effect = _ledfx.effects.create(
                            ledfx=_ledfx,
                            type=last_effect,
                            config=effect_config,
                        )
                        # TODO: Is there a race breaking active virtual counts?                        
                        self.virtual.fallback_suppress_transition = True
                        self.virtual.set_effect(effect)
            else:
                self.was_active = True

            # TODO: does it have a default effect, should we be starting it!?!?
        # else just wait till next frame cycle

    def get_matrix(self):
        try:
            if hasattr(self.virtual.active_effect, "matrix"):
                self.matrix = self.virtual.active_effect.get_matrix() 
            else:
                # Reshape the 1D pixel array into (height, width, 3) for RGB
                reshaped_pixels = self.virtual.assembled_frame.reshape(
                    (self.shape[0], self.shape[1], 3)
                )
                # Convert the numpy array back into a Pillow image
                self.matrix = Image.fromarray(
                    reshaped_pixels.astype(np.uint8), "RGB"
                )
        except Exception as e:
            # TODO: Demote diagnostic to debug
            # _LOGGER.error(f"BlendVirtual.get_matrix {self.virtual_id} {e}")
            self.matrix = Image.new("RGB", self.target_shape, (0, 0, 0))
            self.shape = self.target_shape
            self.matching = True
        
        return self.matrix

    def deactivate(self):
        self.virtual.fallback_suppress_transition = False
        if self.initialised:
            if not self.was_active and self.virtual.active_effect:
                self.virtual.deactivate()
                self.virtual.clear_effect()
        # else we never got to set it up so just junmp out

def stretch_2d_full(blend_virtual):
    if blend_virtual.matching:
        return blend_virtual.matrix

    # doesn't match so stretch it to match
    return blend_virtual.matrix.resize(blend_virtual.target_shape)


def stretch_2d_tile(blend_virtual):
    if blend_virtual.matching:
        return blend_virtual.matrix

    # Get the source image dimensions
    src_width, src_height = blend_virtual.matrix.size

    # Create a new image with the target dimensions
    target_image = Image.new("RGB", blend_virtual.target_shape)

    # Tile the source image to fill the new image
    for i in range(0, blend_virtual.target_shape[1], src_width):
        for j in range(0, blend_virtual.target_shape[0], src_height):
            # Paste the source image at the current position
            target_image.paste(blend_virtual.matrix, (i, j))

    return target_image


def stretch_1d_vertical(blend_virtual):
    _LOGGER.warning("Stretch 1d vertical not implemented")


def stretch_1d_horizontal(blend_virtual):
    _LOGGER.warning("Stretch 1d horizontal not implemented")


STRETCH_FUNCS_MAPPING = {
    "2d full": stretch_2d_full,
    "2d tile": stretch_2d_tile,
    # "1d vertical": stretch_1d_vertical, TODO Implement
    # "1d horizontal": stretch_1d_horizontal, TODO: Implement
}


class Blender(AudioReactiveEffect, LogSec):
    NAME = "Blender"
    CATEGORY = "Matrix"
    HIDDEN_KEYS = ["background_color", "background_brightness", "blur"]
    ADVANCED_KEYS = LogSec.ADVANCED_KEYS + []

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "mask_stretch",
                description="How to stretch the mask source pixles to the effect pixels",
                default="2d full",
            ): vol.In(list(STRETCH_FUNCS_MAPPING.keys())),
            vol.Optional(
                "background_stretch",
                description="How to stretch the background source pixles to the effect pixels",
                default="2d full",
            ): vol.In(list(STRETCH_FUNCS_MAPPING.keys())),
            vol.Optional(
                "foreground_stretch",
                description="How to stretch the foreground source pixles to the effect pixels",
                default="2d full",
            ): vol.In(list(STRETCH_FUNCS_MAPPING.keys())),
            vol.Optional(
                "mask",
                description="The virtual from which to source the mask",
                default="",
            ): str,
            vol.Optional(
                "foreground",
                description="The virtual from which to source the foreground",
                default="",
            ): str,
            vol.Optional(
                "background",
                description="The virtual from which to source the background",
                default="",
            ): str,
            vol.Optional(
                "invert_mask",
                description="Switch Foreground and Background",
                default=False,
            ): bool,
            vol.Optional(
                "mask_cutoff",
                description="1 default = luminance as alpha, anything below 1 is mask cutoff",
                default=1.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.01, max=1.0)),
        }
    )

    def on_activate(self, pixel_count):
        # TODO: refactor to shape tuples instead of rows and columns
        self.rows = self._virtual.config["rows"]
        self.columns = int(self.pixel_count / self.rows)
        self.pixels_shape = np.shape(self.pixels)

    def deactivate(self):
        # if any source devices were off when we started, turn them off now
        self.mask.deactivate()
        self.fore.deactivate()
        self.back.deactivate()

        return super().deactivate()

    def config_updated(self, config):
        # TODO: Ensure virtual names are mangled the same as during virtual creation,
        # for now rely on exactness from user or front end
        self.initialised = False
        self.mask = BlendVirtual(self._config["mask"], self._ledfx.config["virtuals"])
        self.fore = BlendVirtual(self._config["foreground"], self._ledfx.config["virtuals"])
        self.back = BlendVirtual(self._config["background"], self._ledfx.config["virtuals"])

        self.invert_mask = self._config["invert_mask"]
        self.mask_cutoff = self._config["mask_cutoff"]

        self.mask_stretch_func = STRETCH_FUNCS_MAPPING[
            self._config["mask_stretch"]
        ]
        self.foreground_stretch_func = STRETCH_FUNCS_MAPPING[
            self._config["foreground_stretch"]
        ]
        self.background_stretch_func = STRETCH_FUNCS_MAPPING[
            self._config["background_stretch"]
        ]

    def audio_data_updated(self, data):
        pass

    def render(self):

        self.log_sec()

        # we need to keep trying to initialise as with race conditions they may not exist during start up of this blender effect
        if not self.mask.initialised:
            self.mask.init(self._ledfx, (self.rows, self.columns))
        if not self.fore.initialised:
            self.fore.init(self._ledfx, (self.rows, self.columns))
        if not self.back.initialised:
            self.back.init(self._ledfx, (self.rows, self.columns))

        self.mask.get_matrix()
        self.fore.get_matrix()
        self.back.get_matrix()

        mask_image = self.mask_stretch_func(self.mask).convert("L")
        fore_image = self.foreground_stretch_func(self.fore)
        back_image = self.background_stretch_func(self.back)

        if self.mask_cutoff < 1.0:
            cutoff = int(255 * (1 - self.mask_cutoff))
            mask_image = mask_image.point(lambda p: 255 if p > cutoff else 0)

        if self.invert_mask:
            mask_image = ImageOps.invert(mask_image)

        blend_image = Image.composite(fore_image, back_image, mask_image)

        rgb_array = np.frombuffer(blend_image.tobytes(), dtype=np.uint8)
        rgb_array = rgb_array.astype(np.float32)
        rgb_array = rgb_array.reshape(int(rgb_array.shape[0] / 3), 3)

        copy_length = min(self.pixels.shape[0], rgb_array.shape[0])
        self.pixels[:copy_length, :] = rgb_array[:copy_length, :]

        self.try_log()
