# Dual Teleop for Astra Arm

## Calibrate the intrinsic parameters of the camera 

TODO do better with the doc

```bash
# collect calibration image
python -m astra_teleop.calibration_collect -d /dev/video0 -c ./calibration_images
# process calibration image
python -m astra_teleop.calibration_process -c ./calibration_images
```

## Teleop

```bash
python -m astra_teleop.process -d /dev/video0 -c ./calibration_images
```

## Acknowledgment

Special thanks to Hello Robot team and their stretch dex teleop project!

<https://github.com/hello-robot/stretch_dex_teleop>

## License

TODO 