import numpy as np

WIDTH = 1200
HEIGHT = 800

SENSOR_WIDTH = 10
FOCAL_LENGTH = np.sqrt(WIDTH**2 + HEIGHT**2) * SENSOR_WIDTH / WIDTH

CATEGORY2RENDERCONFIG = {
    "default": {
        "focal_length": FOCAL_LENGTH,
        "scale": 1.5,
        "elevation": 5.0,
        "azimuth": 45,
        "z_transl": 0.2,
        "radius": 1.75,
    },
    "barbell": {
        "focal_length": FOCAL_LENGTH,
        "scale": 1.4,
        "elevation": 5.0,
        "azimuth": 45,
        "z_transl": 0.325,
        "radius": 1.4,
    },
}

CATEGORY2GENERATIONCONFIG = {
    "barbell": {
        "prompt": "1 person pulling a loaded barbell from the ground in a powerful deadlift, hips hinged back and spine neutral with eyes looking forward. The person's hands grip the bar firmly, legs driving into the floor as the barbell rises close to the shins, showcasing controlled strength and precise form, full body"
    },
}