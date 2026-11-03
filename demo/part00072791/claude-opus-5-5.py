import cadquery as cq

# Circular base
base_r = 20.0
base_h = 3.0
base = cq.Workplane("XY").circle(base_r).extrude(base_h)

# Slight recess on top of base
recess = (
    cq.Workplane("XY")
    .workplane(offset=base_h - 0.5)
    .circle(base_r - 2.0)
    .extrude(0.5)
)
base = base.cut(recess)

# Small elongated cylinder on top
cyl_r = 3.0
cyl_h = 25.0
cyl = (
    cq.Workplane("XY")
    .workplane(offset=base_h - 0.5)
    .circle(cyl_r)
    .extrude(cyl_h)
    .faces(">Z")
    .edges()
    .fillet(1.5)
)

result = base.union(cyl)