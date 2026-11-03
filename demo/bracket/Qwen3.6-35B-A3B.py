import cadquery as cq

result = (
    cq.Workplane('ZX', origin=(-100, 7, -53))
    .moveTo(0, 0)
    .lineTo(0, 200)
    .lineTo(107, 200)
    .lineTo(107, 0)
    .lineTo(0, 0)
    .close()
    .extrude(-13)
    .union(
        cq.Workplane('ZX', origin=(-100, 7, -53))
        .moveTo(0, 0)
        .lineTo(0, 200)
        .lineTo(107, 200)
        .lineTo(107, 0)
        .lineTo(0, 0)
        .close()
        .extrude(80)
    )
    .cut(
        cq.Workplane('ZX', origin=(-100, 7, -53))
        .center(13, 13)
        .circle(11)
        .extrude(-13)
    )
    .cut(
        cq.Workplane('ZX', origin=(-100, 7, -53))
        .center(13, 187)
        .circle(11)
        .extrude(-13)
    )
)