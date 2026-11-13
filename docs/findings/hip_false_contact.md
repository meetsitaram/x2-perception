# False hip contact: the pictures

Finding A1 in `docs/FINDINGS.md`. The simulator collided the convex hulls of the pelvis and each hip-roll link; the real
parts never touch.

## In the simulator

![MuJoCo render of the pelvis and hip-roll links: real meshes and convex hulls at three hip-roll angles](hip_false_contact.jpg)

MuJoCo, not the training simulator. Static poses, no policy running. Model: AgiBot X2 (`x2_ultra.xml`). Hull distance
between pelvis and hip-roll link: +2.7 mm standing, -0.6 mm at 0.06 rad inward, -3.9 mm at 0.12 rad, -10.9 mm at the
0.235 rad joint limit (negative = overlap).

## On the real robot (user's photos, 2026-11-12)

Legs hanging apart:

| Front | Right side | Gap | Into the recess |
|---|---|---|---|
| ![front](hip_real_1_front.jpg) | ![right side](hip_real_2_right_side.jpg) | ![gap](hip_real_3_gap.jpg) | ![recess](hip_real_4_recess.jpg) |

Leg pushed inward:

| From below | Side | Rear |
|---|---|---|
| ![inward, from below](hip_real_5_inward_below.jpg) | ![inward, side](hip_real_6_inward_side.jpg) | ![inward, rear](hip_real_7_inward_rear.jpg) |

Test by the user with the leg held at its inward stop: "a playing card nicely slides all around without being blocked".
The parts do not touch; the joint is the stop.
