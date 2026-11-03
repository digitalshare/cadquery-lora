import cadquery as cq

# Parameters
base_radius = 20.0
base_height = 5.0
recess_radius = 15.0
recess_depth = 1.0

cyl_radius = 5.0
cyl_height = 30.0
fillet_radius = 2.0

# Base with a shallow recessed area
base = (
    cq.Workplane("XY")
    .cylinder(base_height, base_radius)
    .faces(">Z")
    .workplane()
    .circle(recess_radius)
    .cutBlind(recess_depth)
)

# Elongated cylinder with rounded edges
cyl = (
    cq.Workplane("XY")
    .cylinder(cyl_height, cyl_radius)
    .edges("|Z")
    .fillet(fillet_radius)
    .translate((0, 0, base_height))
)

# Combine the parts
result = base.union(cyl)