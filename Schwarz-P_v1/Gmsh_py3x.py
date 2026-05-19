import sys
import numpy as np
import gmsh

from scipy.io import loadmat
import matplotlib.pyplot as plt


def TPMS(p, size):
    return np.sum(np.cos(2 * np.pi * p / size), axis=1)


# -------------------------------------------------------------------
# Load MATLAB data
# -------------------------------------------------------------------

mat = loadmat("to_gmsh.mat")

n_quad = int(mat["n_quad"][0, 0])
n_pyr = int(mat["n_pyr"][0, 0])
n_lvl = int(mat["n_layers"][0, 0])

p = mat["p_array"]
quads = mat["quads"].astype(int)
pyrs = mat["pyrs"].astype(int)

I = int(mat["I"][0, 0])

Size = float(mat["Size"][0, 0])
Ds = float(mat["Ds"][0, 0])

factor = int(mat["factor_streamwise"][0, 0])
mesh_dir = int(mat["mesh_dir"][0, 0])

# -------------------------------------------------------------------
# Initialize Gmsh
# -------------------------------------------------------------------

gmsh.initialize()
gmsh.model.add("Schwarz-P")

# -------------------------------------------------------------------
# Create bounding geometry
# -------------------------------------------------------------------

if mesh_dir == 1:

    for i in range(8):
        x = (Size / 2) * (-1) ** (i // 4)
        y = (Size / 2) * (-1) ** (i // 2)
        z = (Size / 2) * (-1) ** ((i + 1) // 2)

        gmsh.model.geo.addPoint(
            x,
            y,
            z,
            Size * factor * Ds,
            i + 1
        )

    for i in range(12):

        if i < 8:
            a = i + 1
            b = ((i + 1) % 4) + (i + 1) - (i % 4)
        else:
            a = (i % 4) + 1
            b = a + 4

        gmsh.model.geo.addLine(a, b, i + 1)

    # Transfinite curves
    for i in [2, 4, 6, 8]:
        gmsh.model.geo.mesh.setTransfiniteCurve(
            i,
            int(1 / Ds) + 1
        )

    for i in [1, 3, 5, 7, 9, 10, 11, 12]:
        gmsh.model.geo.mesh.setTransfiniteCurve(
            i,
            int(1 / (factor * Ds)) + 1
        )

    # Curve loops
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)
    gmsh.model.geo.addCurveLoop([1, 10, -5, -9], 3)
    gmsh.model.geo.addCurveLoop([3, 12, -7, -11], 4)
    gmsh.model.geo.addCurveLoop([4, 9, -8, -12], 5)
    gmsh.model.geo.addCurveLoop([2, 11, -6, -10], 6)

# -------------------------------------------------------------------
# Add TPMS points
# -------------------------------------------------------------------

p = np.hstack((np.zeros((p.shape[0], 1), dtype=int), p))

for i in range(1, I + n_pyr + 1):

    p[-i, 0] = gmsh.model.geo.addPoint(
        float(p[-i, 1]),
        float(p[-i, 2]),
        float(p[-i, 3])
    )

gmsh.model.geo.synchronize()

# -------------------------------------------------------------------
# Create boundary surfaces
# -------------------------------------------------------------------

lvl_int = n_lvl * I + np.arange(I)

for axis in range(3):

    for sign in range(2):

        mask = (
            np.power(
                p[lvl_int, axis + 1] - ((-1) ** sign) * Size / 2,
                2
            )
            <= (Size * Ds / 10) ** 2
        )

        face_pts = p[lvl_int, :][mask]

        face_pts = face_pts[
            np.argsort(
                np.arctan2(
                    face_pts[:, (axis + 1) % 3 + 1],
                    face_pts[:, (axis + 2) % 3 + 1]
                )
            )
        ]

        face_pts = np.vstack((face_pts, face_pts[0, :]))

        outer_lines = []

        for i in range(face_pts.shape[0] - 1):

            line = gmsh.model.geo.addLine(
                int(face_pts[i, 0]),
                int(face_pts[i + 1, 0])
            )

            outer_lines.append(line)

            gmsh.model.geo.mesh.setTransfiniteCurve(line, 2)

        outer_loop = gmsh.model.geo.addCurveLoop(outer_lines)

        sid = 2 * axis + sign + 1

        if mesh_dir == 0:
            gmsh.model.geo.addPlaneSurface([outer_loop], sid)
        else:
            gmsh.model.geo.addPlaneSurface(
                [sid, outer_loop],
                sid
            )

# -------------------------------------------------------------------
# Pyramid surfaces
# -------------------------------------------------------------------

pyr_surfs = []

for i in range(pyrs.shape[0]):

    q = pyrs[i, :] - 1

    for j in range(4):

        l1 = gmsh.model.geo.addLine(
            int(p[q[j], 0]),
            int(p[q[(j + 1) % 4], 0])
        )

        l2 = gmsh.model.geo.addLine(
            int(p[q[(j + 1) % 4], 0]),
            int(p[q[4], 0])
        )

        l3 = gmsh.model.geo.addLine(
            int(p[q[4], 0]),
            int(p[q[j], 0])
        )

        for line in [l1, l2, l3]:
            gmsh.model.geo.mesh.setTransfiniteCurve(line, 2)

        loop = gmsh.model.geo.addCurveLoop([l1, l2, l3])

        surf = gmsh.model.geo.addPlaneSurface([loop])

        pyr_surfs.append(surf)

# -------------------------------------------------------------------
# Volume
# -------------------------------------------------------------------

gmsh.model.geo.addSurfaceLoop(
    [1, 2, 3, 4, 5, 6] + pyr_surfs,
    1
)

fluid = gmsh.model.geo.addVolume([1], 1)

gmsh.model.geo.synchronize()

# -------------------------------------------------------------------
# Generate mesh
# -------------------------------------------------------------------

gmsh.model.mesh.generate(3)

node_tags, node_coords, _ = gmsh.model.mesh.getNodes(
    3,
    1,
    includeBoundary=True
)

nodes = np.hstack((
    node_tags.reshape((-1, 1)),
    node_coords.reshape((-1, 3))
))

# -------------------------------------------------------------------
# Find last layer nodes
# -------------------------------------------------------------------

lvl_min = np.min(
    TPMS(p[n_lvl * I:(n_lvl + 1) * I, 1:], Size)
)

lvl_max = np.max(
    TPMS(p[n_lvl * I:(n_lvl + 1) * I, 1:], Size)
)

a = np.logical_and(
    TPMS(nodes[:, 1:4], Size) > lvl_min - 1e-6,
    TPMS(nodes[:, 1:4], Size) < lvl_max + 1e-6
)

for i in range(n_lvl * I, (n_lvl + 1) * I):

    b = (
        np.sum(
            (nodes[a, 1:4] - p[i, 1:4]) ** 2,
            axis=1
        )
        < (Size * (Ds / 10)) ** 2
    )

    idx = np.argmax(b)

    p[i, 0] = int(nodes[a, 0][idx])

print("Obtained last layer node IDs")

# -------------------------------------------------------------------
# Find pyramid peak nodes
# -------------------------------------------------------------------

lvl_min = np.min(TPMS(p[-n_pyr:, 1:], Size))
lvl_max = np.max(TPMS(p[-n_pyr:, 1:], Size))

a = np.logical_and(
    TPMS(nodes[:, 1:4], Size) > lvl_min - 1e-6,
    TPMS(nodes[:, 1:4], Size) < lvl_max + 1e-6
)

for i in range((n_lvl + 1) * I, (n_lvl + 1) * I + n_pyr):

    b = (
        np.sum(
            (nodes[a, 1:4] - p[i, 1:4]) ** 2,
            axis=1
        )
        < (Size * (Ds / 10)) ** 2
    )

    idx = np.argmax(b)

    p[i, 0] = int(nodes[a, 0][idx])

print("Obtained pyramid node IDs")

# -------------------------------------------------------------------
# Remove temporary pyramid surfaces
# -------------------------------------------------------------------

for surf in pyr_surfs:
    gmsh.model.geo.remove([(2, surf)])

# -------------------------------------------------------------------
# Add structured hex elements
# -------------------------------------------------------------------

p_array = p[0:n_lvl * I, 1:4].reshape(-1)

gmsh.model.mesh.addNodes(3, 1, [], p_array)

p[0:n_lvl * I, 0] = gmsh.model.mesh.getNodes(3, 1)[0][-n_lvl * I:]

hex_nodes = np.zeros(8 * n_lvl * n_quad, dtype=np.int64)

for i in range(n_lvl):

    for j in range(n_quad):

        base = (n_quad * i + j) * 8

        hex_nodes[base + 0] = int(p[quads[j, 0] - 1 + i * I, 0])
        hex_nodes[base + 1] = int(p[quads[j, 1] - 1 + i * I, 0])
        hex_nodes[base + 2] = int(p[quads[j, 2] - 1 + i * I, 0])
        hex_nodes[base + 3] = int(p[quads[j, 3] - 1 + i * I, 0])

        hex_nodes[base + 4] = int(p[quads[j, 0] - 1 + (i + 1) * I, 0])
        hex_nodes[base + 5] = int(p[quads[j, 1] - 1 + (i + 1) * I, 0])
        hex_nodes[base + 6] = int(p[quads[j, 2] - 1 + (i + 1) * I, 0])
        hex_nodes[base + 7] = int(p[quads[j, 3] - 1 + (i + 1) * I, 0])

gmsh.model.mesh.addElementsByType(
    fluid,
    5,
    [],
    hex_nodes
)

print("Added fluid zone hex elements")

# -------------------------------------------------------------------
# Add pyramid elements
# -------------------------------------------------------------------

pyr_nodes = np.zeros(5 * n_pyr, dtype=np.int64)

for i in range(n_pyr):

    pyr_nodes[5 * i + 0] = int(p[pyrs[i, 0] - 1, 0])
    pyr_nodes[5 * i + 1] = int(p[pyrs[i, 1] - 1, 0])
    pyr_nodes[5 * i + 2] = int(p[pyrs[i, 2] - 1, 0])
    pyr_nodes[5 * i + 3] = int(p[pyrs[i, 3] - 1, 0])
    pyr_nodes[5 * i + 4] = int(p[pyrs[i, 4] - 1, 0])

gmsh.model.mesh.addElementsByType(
    fluid,
    7,
    [],
    pyr_nodes
)

print("Added pyramid elements")

# -------------------------------------------------------------------
# Physical groups
# -------------------------------------------------------------------

gmsh.model.geo.addPhysicalGroup(
    3,
    [fluid],
    1,
    name="Fluid"
)

gmsh.model.geo.synchronize()

# -------------------------------------------------------------------
# Surface elements
# -------------------------------------------------------------------

for axis in range(3):

    for sign in range(2):

        face_pts = np.hstack((
            p[:I, :],
            np.arange(I).reshape((-1, 1))
        ))

        mask = (
            (p[:I, axis + 1] - ((-1) ** sign) * Size / 2) ** 2
            <= (Size * Ds / 10) ** 2
        )

        face_pts = face_pts[mask]

        face_pts = face_pts[
            np.argsort(
                np.arctan2(
                    face_pts[:, (axis + 1) % 3 + 1],
                    face_pts[:, (axis + 2) % 3 + 1]
                )
            ),
            -1
        ]

        face_pts = np.append(face_pts, face_pts[0])

        surf_nodes = np.zeros(
            4 * n_lvl * (face_pts.shape[0] - 1),
            dtype=np.int64
        )

        for i in range(n_lvl):

            for j in range(face_pts.shape[0] - 1):

                base = 4 * ((face_pts.shape[0] - 1) * i + j)

                surf_nodes[base + 0] = int(p[int(face_pts[j]) + i * I, 0])
                surf_nodes[base + 1] = int(p[int(face_pts[j + 1]) + i * I, 0])
                surf_nodes[base + 2] = int(p[int(face_pts[j + 1]) + (i + 1) * I, 0])
                surf_nodes[base + 3] = int(p[int(face_pts[j]) + (i + 1) * I, 0])

        entity = 2 * axis + sign + 1

        gmsh.model.mesh.addElementsByType(
            entity,
            3,
            [],
            surf_nodes
        )

        namestr = (
            ["x", "y", "z"][axis]
            + "_"
            + ["plus", "minus"][sign]
        )

        gmsh.model.geo.addPhysicalGroup(
            2,
            [entity],
            name=namestr
        )

print("Added unit cell face elements")

# -------------------------------------------------------------------
# TPMS surface elements
# -------------------------------------------------------------------

s = gmsh.model.add_discrete_entity(2)

surf_nodes = np.zeros(4 * n_quad, dtype=np.int64)

for i in range(n_quad):

    surf_nodes[4 * i + 0] = int(p[quads[i, 0] - 1, 0])
    surf_nodes[4 * i + 1] = int(p[quads[i, 1] - 1, 0])
    surf_nodes[4 * i + 2] = int(p[quads[i, 2] - 1, 0])
    surf_nodes[4 * i + 3] = int(p[quads[i, 3] - 1, 0])

gmsh.model.mesh.addElementsByType(
    s,
    3,
    [],
    surf_nodes
)

gmsh.model.geo.addPhysicalGroup(
    2,
    [s],
    name="Surface"
)

print("Added surface elements")

# -------------------------------------------------------------------
# Finalize
# -------------------------------------------------------------------

gmsh.model.geo.synchronize()

if "-popup" in sys.argv:
    gmsh.fltk.run()

gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

gmsh.write("Schwarz-P.msh")

gmsh.finalize()
