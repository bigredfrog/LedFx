import logging
import random
import timeit

import numpy as np
import vnoise
import voluptuous as vol
from PIL import Image

from ledfx.effects.gradient import GradientEffect
from ledfx.effects.twod import Twod

_LOGGER = logging.getLogger(__name__)

# this effect is inspired by the WLED soap effect found at
# https://github.com/Aircoookie/WLED/blob/f513cae66eecb2c9b4e8198bd0eb52d209ee281f/wled00/FX.cpp#L7472

def easeInOutQuad(t):
    t *= 2
    if t < 1:
        return t * t / 2
    else:
        t -= 1
        return -(t * (t - 2) - 1) / 2


class Noise2d(Twod, GradientEffect):
    NAME = "Noise"
    CATEGORY = "Matrix"
    # add keys you want hidden or in advanced here
    HIDDEN_KEYS = Twod.HIDDEN_KEYS + [
        "background_color",
        "gradient_roll",
    ]
    ADVANCED_KEYS = Twod.ADVANCED_KEYS + []

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "speed",
                description="Speed of the effect",
                default=1,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=5)),
            vol.Optional(
                "smoothness",
                description="smoothness duh",
                default=0.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
            vol.Optional(
                "stretch",
                description="Stretch of the effect",
                default=1.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.5, max=1.5)),
            vol.Optional(
                "zoom",
                description="zoom density",
                default=2,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.5, max=20)),
            vol.Optional(
                "impulse_decay",
                description="Decay filter applied to the impulse for development",
                default=0.06,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.01, max=0.3)),
            vol.Optional(
                "multiplier",
                description="audio injection multiplier, 0 is none",
                default=2.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=4.0)),
            vol.Optional(
                "soap",
                description="Add soap smear to noise",
                default=False,
            ): bool,
        }
    )

    def __init__(self, ledfx, config):
        self.first_run = True
        self.last_rotate = 0
        super().__init__(ledfx, config)

    def config_updated(self, config):
        super().config_updated(config)
        # copy over your configs here into variables
        self.speed = self._config["speed"]
        self.stretch = self._config["stretch"]
        self.zoom = self._config["zoom"]
        self.multiplier = self._config["multiplier"]
        self.soap = self._config["soap"]
        self.smoothness = self._config["smoothness"]

        self.lows_impulse_filter = self.create_filter(
            alpha_decay=self._config["impulse_decay"], alpha_rise=0.99
        )
        self.lows_impulse = 0

        if self.last_rotate != self._config["rotate"]:
            # as rotate could be non symetrical we better reseed everything
            self.first_run = True
            self.last_rotate = self._config["rotate"]

    def do_once(self):
        super().do_once()

        if self.first_run:
            self.noise_3d = np.zeros(
                (self.r_height, self.r_width), dtype=np.float64
            )
            self.noise_x = random.random()
            self.noise_y = random.random()
            self.noise_z = random.random()
            self.noise = vnoise.Noise()
            self.first_run = False

            self.amplitude_cols = self.r_width / 32
            self.amplitude_rows = self.r_height / 32

        self.scale_x = self.zoom / self.r_width
        self.scale_y = self.zoom / self.r_height

        self.seed_matrix = Image.new("RGB", (self.r_width, self.r_height))
        self.seed_image = True

    def audio_data_updated(self, data):
        self.lows_impulse = self.lows_impulse_filter.update(
            data.lows_power(filtered=False) * self.multiplier
        )

    def noise_to_image(self, noise):
        ###
        # This is the main function that will be called to generate the image
        # The noise is a 2d array of values between 0 and 1
        # The image is a PIL image that should be modified in place
        ###

        # map from 0,1 space into the gradient color space via our nicely vecotrised function
        color_array = self.get_gradient_color_vectorized2d(noise).astype(np.uint8)
        # transform the numpy array into a PIL image in one easy step
        return Image.fromarray(color_array, "RGB")

    def draw(self):

        if self.test:
            self.draw_test(self.m_draw)

        # time invariant movement throuh the noise space
        self.mov = 0.5 * self.speed * self.passed

        self.noise_x += self.mov
        self.noise_y += self.mov
        self.noise_z += self.mov

        # if we are pixel stuffing into a seed image, setup here
        # pixels = self.seed_image.load()

        # generate arrays of the X adn Y axis of our plane, with a singular Z
        # this should allow libs to use any internal acceleration for unrolling across all points

        bass_x = self.scale_x * self.lows_impulse
        bass_y = self.scale_y * self.lows_impulse

        scale_x = self.scale_x + bass_x
        scale_y = self.scale_y + bass_y

        # there is something very funky in the x and y axis here, but it works
        noise_x = self.noise_x - (scale_x * self.r_height / 2)
        noise_y = self.noise_y - (scale_y * self.r_width / 2)

        x_array = np.linspace(
            noise_x, noise_x + scale_x * self.r_height, self.r_height
        )
        y_array = np.linspace(
            noise_y, noise_y + scale_y * self.r_width, self.r_width
        )
        z_array = np.array([self.noise_z])

        ###################################################################################
        # This is where the magic happens, calling the lib of choice to get the noise plane
        ###################################################################################
        # opensimplex at 128x128 on dev machine is 200 ms per frame - Unusable
        #        self.noise_3d = opensimplex.noise3array(x_array, y_array, z_array)
        # vnoise at 128x128 on dev machine is 2.5 ms per frame - Current best candidate

        new_noise = np.squeeze(self.noise.noise3(x_array, y_array, z_array, grid_mode=True))

        if not self.soap:
            self.noise_3d = new_noise
        else:
            self.noise_3d = (self.smoothness * self.noise_3d) + ((1 - self.smoothness) * new_noise)

        # apply the stetch param to expand the range of the color space, as noise is likely not full -1 to 1
        noise_stretched = self.noise_3d * self.stretch
        # normalise the noise from -1,1 range to 0,1
        noise_normalised = (noise_stretched + 1) / 2

        if self.soap and self.seed_image:
            self.seed_matrix = self.noise_to_image(noise_normalised)
            self.seed_image = False

        # _LOGGER.info(f"matrix shape: {self.matrix.size}")
        # _LOGGER.info(f"seed_matrix shape: {self.seed_matrix.size}")
        # _LOGGER.info(f"noise_3d shape: {self.noise_3d.shape}")
        # _LOGGER.info(f"r_width: {self.r_width}, r_height: {self.r_height}")

        if self.soap:
            leds_buff = np.tile(np.array([0, 0, 0]), (self.r_width, 1))

            for y in range(self.r_height):
                amount = self.noise_3d[y,0] * self.amplitude_cols
                delta = int(abs(amount))
                fraction = abs(amount) - delta
                for x in range(self.r_width):
                    if amount < 0:
                        zD = x - delta
                        zF = zD - 1
                    else:
                        zD = x + delta
                        zF = zD + 1

                    if zD >= 0 and zD < self.r_width:
                        pixel_a = np.array(self.seed_matrix.getpixel((zD, y)))
                    else:
                        pixel_a = self.get_gradient_color(noise_normalised[y, abs(zD) % self.r_width])

                    if zF >= 0 and zF < self.r_width:
                        pixel_b = np.array(self.seed_matrix.getpixel((zF, y)))
                    else:
                        pixel_b = self.get_gradient_color(noise_normalised[y, abs(zF) % self.r_width])

                    leds_buff[x] = pixel_a * easeInOutQuad(1 - fraction) + pixel_b * easeInOutQuad(fraction)
                for x in range(self.r_width):
                    self.seed_matrix.putpixel((x, y), tuple(leds_buff[x].clip(0, 255).astype(np.uint8)))

            leds_buff = np.tile(np.array([0, 0, 0]), (self.r_height, 1))

            for x in range(self.r_width):
                amount = self.noise_3d[0,x] * self.amplitude_rows
                delta = int(abs(amount))
                fraction = abs(amount) - delta
                for y in range(self.r_height):
                    if amount < 0:
                        zD = y - delta
                        zF = zD - 1
                    else:
                        zD = y + delta
                        zF = zD + 1

                    if zD >= 0 and zD < self.r_height:
                        pixel_a = np.array(self.seed_matrix.getpixel((x, zD)))
                    else:
                        pixel_a = self.get_gradient_color(noise_normalised[abs(zD) % self.r_height, x])

                    if zF >= 0 and zF < self.r_height:
                        pixel_b = np.array(self.seed_matrix.getpixel((x, zF)))
                    else:
                        pixel_b = self.get_gradient_color(noise_normalised[abs(zF) % self.r_height, x])

                    leds_buff[y] = pixel_a * easeInOutQuad(1 - fraction) + pixel_b * easeInOutQuad(fraction)
                for y in range(self.r_height):
                    self.seed_matrix.putpixel((x, y), tuple(leds_buff[y].clip(0, 255).astype(np.uint8)))

        # _LOGGER.info(f"x_array: {x_array}")
        # _LOGGER.info(f"y_array: {y_array}")
        # _LOGGER.info(f"z_array: {z_array}")
        # _LOGGER.info(f"simple_noise3d: {self.simple_n3d}")
        # _LOGGER.info(f"shape: {self.simple_n3d.shape}")
        # _LOGGER.info(f"min {np.min(self.simple_n3d)}, max {np.max(self.simple_n3d)}")

        if not self.soap:
            self.matrix = self.noise_to_image(noise_normalised)
        else:
            self.matrix = self.seed_matrix
