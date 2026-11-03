import cadquery as cq

result = (
    cq.Workplane('XY', origin=(-100, 0, -10))
    .center(100, 0)
    .circle(100)
    .extrude(20)
    .union(
        cq.Workplane('XY', origin=(-100, 0, -10))
        .center(100, 0)
        .circle(100)
        .extrude(20)
    )
)