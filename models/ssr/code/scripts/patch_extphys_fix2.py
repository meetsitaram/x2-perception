p = "/mnt/filesystem-g4/tracks/ssr/code/ssr/env.py"
e = open(p).read()
a = "        self.g_rb[wi, gi] = B[..., 3:6].norm(dim=-1)\n"
assert e.count(a) == 1
e = e.replace(a, a + """        gxp, gxm = wp.to_torch(self.dd.geom_xpos), wp.to_torch(self.dd.geom_xmat)   # world-body geoms are not re-posed by kinematics: write the pose
        gxp[wi, gi] = B[..., 0:3]
        gxm[wi, gi] = quat_to_mat(B[..., 6:10].reshape(-1, 4)).reshape(len(ids), self.ext_phys, 3, 3).to(gxm.dtype).reshape(gxm[wi, gi].shape)
""")
a = "        self.ext_phys = int(ext_phys)\n"
assert e.count(a) == 1
e = e.replace(a, a + "        if self.ext_phys:\n            nconmax = max(nconmax, 30)                                 # wall boxes add broadphase pairs\n")
open(p, "w").write(e)
print("ok")
