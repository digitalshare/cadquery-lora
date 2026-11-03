import cadquery as cq

# Parameters
outer_diameter = 0.75
thickness = 0.04
central_hole_diameter = 0.20   # radius 0.1
bolt_hole_diameter = 0.06      # radius 0.03
bolt_radius_from_center = (outer_diameter / 2) - 0.05  # positioned near the rim

# Build the flange
result = (
    cq.Workplane("XY")
    .circle(outer_diameter / 2)
    .extrude(thickness)
    .faces(">Z")
    .workplane()
    .hole(central_hole_diameter)
    .pushPoints([
        (bolt_radius_from_center, 0),
        (0, bolt_radius_from_center),
        (-bolt_radius_from_center, 0),
        (0, -bolt_radius_from_center)
    ])
    .hole(bolt_hole_diameter)
)