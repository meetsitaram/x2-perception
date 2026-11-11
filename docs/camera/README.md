# Training camera reference: RealSense on the X2's chest

`chest_realsense_depth_reference.jpg` is the user's reference for the camera all camera tracks train with (user, 2026-11-11:
"This one"; "Color image is not wide enough, but depth one is"; "Use the depth one as reference and add some measurement
overlay and keep that in the agent repo for reference"). It is an infrared frame from an Intel RealSense held on the X2's chest,
centred, just above the chest's built-in sensor window and below the logo, pitched down. The infrared imager has the depth
stream's field of view (87 x 58 degrees), so the frame shows what depth covers. Policies use depth only; the colour stream
(69 x 42 degrees) is too narrow and is not used.

## What the overlay says

| Line | Distance ahead of the toe tips | Source |
|---|---|---|
| Bottom edge | 0 m, the robot's own toe tips | seen in the frame |
| Rug inner border | about 0.55 m | the user's tape on the floor (read from a photo, +-5 cm) |
| Concrete strip | 0.96-1.00 m, at about 27% of the image height from the top | the user's tape |
| Top edge | about 1.9 m | fitted camera pose |
| Dashed lines | 0.25 to 1.75 m | fitted camera pose |

Fitted pose (the coordinator's estimate, to be replaced by measured values): camera 0.90 m above the floor, lens 0.12 m behind
the toe tips, optical axis 53 degrees below horizontal. With it the 1.0 m strip falls at 26.5% from the top and the rug border
at 50%; the photo shows about 27% and about 52-54%.

## How the tracks use it

A track's simulated camera is accepted when, at the standing pose on flat ground, its depth view reproduces the two measured
landmarks: toe tips just inside the bottom edge, and a floor line 1.0 m ahead of the toes at 27% (+-3%) from the top. Each
track renders that view with the two lines marked and keeps it next to its state file.

## Limits

- The picture is a phone photo of a laptop screen, rectified by hand-picked corners (`make_overlay.py`): positions are good to
  a few percent of image height. It is assumed the screen showed the whole frame.
- Camera height, tilt and model (D435-class) are not measured yet.
- Rebuild: `python make_overlay.py <photo.jpg> chest_realsense_depth_reference.jpg`. The original photo is not in the repo.
