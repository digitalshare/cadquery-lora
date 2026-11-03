import cadquery as cq

L = 0.75
W = 0.4
T = 0.05
H = 0.3
r = 0.04

base = cq.Workplane("XY").box(L, W, T, centered=(True, True, False))
wall = (
    cq.Workplane("XY")
    .box(L, T, H, centered=(True, False, False))
    .translate((0, W / 2 - T, 0))
)
result = base.union(wall)

holes = (
    cq.Workplane("XY")
    .pushPoints([(-L / 4, -0.03), (L / 4, -0.03)])
    .circle(r)
    .extrude(T * 3)
    .translate((0, 0, -T))
)
result = result.cut(holes)