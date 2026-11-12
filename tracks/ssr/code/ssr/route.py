"""The apparatus mobility route, hand-placed waypoints from loco-manip-apparatus sim/x2_route.py (x, y in inches, apparatus frame)."""
IN = 0.0254
WALK, SLOW = 1.0, 0.0
SPAWN = (30 * IN, 30 * IN, 0.0)
# name, x, y, heading deg, speed class, hold s
MOBILITY = [
    ("to the door", 84, 47, 0, WALK, 0.0),
    ("through the door", 104, 46, 0, SLOW, 0.0),
    ("clear of the door wall", 107, 27, -70, SLOW, 0.0),
    ("over the front 4x4", 130, 27, 0, SLOW, 0.0),
    ("button 1", 130, 13, -90, SLOW, 1.5),
    ("square to the front 4x4", 130, 27, 90, SLOW, 0.0),
    ("back over the front 4x4", 108, 27, 180, SLOW, 0.0),
    ("door side", 108, 60, 90, SLOW, 0.0),
    ("over the back 4x4", 129, 64, 0, SLOW, 0.0),
    ("button 2", 129, 79, 90, SLOW, 1.5),
    ("ramps, back side", 150, 72, 0, SLOW, 0.0),
    ("over the back hill", 196, 72, 0, SLOW, 0.0),
    ("button 3", 226, 83, 90, SLOW, 1.5),
    ("ramps, diagonal", 196, 46, -130, SLOW, 0.0),
    ("button 4", 162, 14, -90, SLOW, 1.5),
    ("ramps, front side", 214, 24, 0, SLOW, 0.0),
    ("around the wall", 278, 54, 0, WALK, 0.0),
    ("past the wall", 312, 54, 0, WALK, 0.0),
    ("foot of the stairs", 326, 21, 0, SLOW, 1.0),
    ("up the stairs", 380, 21, 0, SLOW, 0.0),
    ("landing, turn left", 404, 34, 90, SLOW, 0.0),
    ("button 5", 410, 83, 90, SLOW, 2.0),
]
