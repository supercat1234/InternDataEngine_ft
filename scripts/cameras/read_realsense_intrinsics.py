#!/usr/bin/env python3
"""Read a RealSense RGB profile's calibration without starting video streams."""
import argparse
import json
from pathlib import Path


def main():
    import pyrealsense2 as rs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    context = rs.context()
    devices = [d for d in context.query_devices()
               if d.get_info(rs.camera_info.serial_number) == args.serial]
    if not devices:
        parser.error(f"RealSense {args.serial} was not found")
    device = devices[0]
    for sensor in device.sensors:
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != rs.stream.color or profile.format() != rs.format.rgb8:
                continue
            video = profile.as_video_stream_profile()
            if (video.width(), video.height(), profile.fps()) != (args.width, args.height, args.fps):
                continue
            intr = video.get_intrinsics()
            result = dict(device_name=device.get_info(rs.camera_info.name), serial_number=args.serial,
                firmware_version=device.get_info(rs.camera_info.firmware_version), stream="color", format="rgb8",
                resolution_width=intr.width, resolution_height=intr.height, frequency=profile.fps(),
                camera_params=[intr.fx, intr.fy, intr.ppx, intr.ppy],
                distortion_model=str(intr.model), distortion_coefficients=list(intr.coeffs))
            output = json.dumps(result, indent=2) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(output)
            print(output, end="")
            return
    parser.error("The requested RGB resolution/fps is not supported")


if __name__ == "__main__":
    main()
