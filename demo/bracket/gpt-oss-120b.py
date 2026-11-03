import cadquery as cq

# Base plate
base = cq.Workplane("XY").box(0.75, 0.4, 0.05)

# Through-holes in the base (radius 0.04 → diameter 0.08)
base = (
    base.faces(">Z")
    .workplane()
    .pushPoints([(-0.2, 0), (0.2, 0)])
    .hole(0.08)
)

# Vertical wall on one long edge (same length, thickness 0.05, height 0.3)
wall = (
    cq.Workplane("XY")
    .box(0.75, 0.05, 0.3)
    .translate((0, 0.225, 0.175))  # align with base top and outer edge
)

# Combine base and wall
result = base.union(wall)