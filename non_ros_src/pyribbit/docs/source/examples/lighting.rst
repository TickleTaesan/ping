.. _lighting_guide:

Creating Lights
===============

Pyribbit supports three types of punctual light:

- :class:`.PointLight`: Point-based light sources, such as light bulbs.
- :class:`.SpotLight`: A conical light source, like a flashlight.
- :class:`.DirectionalLight`: A general light that does not attenuate with
  distance.

Creating lights is easy -- just specify their basic attributes:

>>> pl = pyribbit.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
>>> sl = pyribbit.SpotLight(color=[1.0, 1.0, 1.0], intensity=2.0,
...                         innerConeAngle=0.05, outerConeAngle=0.5)
>>> dl = pyribbit.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)

For more information about how these lighting models are implemented,
see their class documentation.
