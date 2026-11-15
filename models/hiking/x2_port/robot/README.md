x2_ultra_pelvis_root.urdf = /mnt/filesystem-g4/data/robots/x2_ultra/urdf/x2_ultra.urdf with the massless base_link and its fixed joint
to pelvis removed (nothing else), so the articulation root in Isaac is named pelvis. meshes -> symlink to the shared meshes.
Also removed: the XML declaration line (pytorch_kinematics cannot parse a str with an encoding declaration).
