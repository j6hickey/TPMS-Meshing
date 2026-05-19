import numpy as np
import gmsh
import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt

def TPMS(p,Size):
    val = np.sum(np.cos(2*np.pi*p/Size),axis=1)
    return val

mat = loadmat("to_gmsh.mat")
n_quad = int(mat["n_quad"][0][0])
n_pyr = int(mat["n_pyr"][0][0])
n_lvl = int(mat["n_layers"][0][0])
p = mat["p_array"]
quads = mat["quads"]
pyrs = mat["pyrs"]
I = int(mat["I"][0][0])
Size = float(mat["Size"][0][0])
Ds = float(mat["Ds"][0][0])
factor = int(mat["factor_streamwise"][0][0])
dir = int(mat["mesh_dir"][0][0])

gmsh.initialize()
gmsh.model.add("Schwarz-P")

if dir == 1:
    for i in range(8):
        gmsh.model.geo.addPoint((Size/2)*(-1)**(i//4), (Size/2)*(-1)**(i//2), (Size/2)*(-1)**((i+1)//2), Size*factor*Ds, i+1)
    for i in range(12):
        if i < 8:
            a = i+1
            b = ((i+1)%4)+(i+1)-(i%4)
        else:
            a = (i%4)+1
            b = a+4
        gmsh.model.geo.addLine(a,b,i+1)

    for i in [2,4,6,8]:
        gmsh.model.geo.mesh.setTransfiniteCurve(i,int(1/Ds)+1)
    for i in [1,3,5,7,9,10,11,12]:
        gmsh.model.geo.mesh.setTransfiniteCurve(i,int(1/(factor*Ds))+1)

    gmsh.model.geo.addCurveLoop([1,2,3,4],1)
    gmsh.model.geo.addCurveLoop([5,6,7,8],2)
    gmsh.model.geo.addCurveLoop([1,10,-5,-9],3)
    gmsh.model.geo.addCurveLoop([3,12,-7,-11],4)
    gmsh.model.geo.addCurveLoop([4,9,-8,-12],5)
    gmsh.model.geo.addCurveLoop([2,11,-6,-10],6)

p = np.hstack((np.zeros((p.shape[0],1)),p))
for i in range(1,I+n_pyr+1):
    p[-i,0] = gmsh.model.geo.addPoint(p[-i,1],p[-i,2],p[-i,3])
gmsh.model.geo.synchronize()

lvl_int = n_lvl*I+np.arange(I)
for axis in range(0,3):
    for sign in range(0,2):
        face_pts = p[lvl_int,:][np.pow(p[lvl_int,axis+1]-((-1)**sign)*Size/2,2) <= (Size*Ds/10)**2,:]
        face_pts = face_pts[np.argsort(np.arctan2(face_pts[:,(axis+1)%3+1],face_pts[:,(axis+2)%3+1])),:]
        face_pts = np.vstack((face_pts,face_pts[0,:]))
        outer_lines = []
        for i in range(face_pts.shape[0]-1):
            outer_lines.append(gmsh.model.geo.addLine(int(face_pts[i,0]),
                                int(face_pts[i+1,0])))
            gmsh.model.geo.mesh.setTransfiniteCurve(outer_lines[-1], 2)
        outer_loop = gmsh.model.geo.addCurveLoop(outer_lines)
        if dir == 0:
            gmsh.model.geo.addPlaneSurface([outer_loop],2*axis+sign+1)
        elif dir == 1:
            gmsh.model.geo.addPlaneSurface([2*axis+sign+1,outer_loop],2*axis+sign+1)

pyr_surfs = []
for i in range(pyrs.shape[0]):
    q = pyrs[i,:]-1
    for j in range(0,4):
        l1 = gmsh.model.geo.addLine(int(p[q[j],0]),int(p[q[(j+1)%4],0]))
        gmsh.model.geo.mesh.setTransfiniteCurve(l1, 2)
        l2 = gmsh.model.geo.addLine(int(p[q[(j+1)%4],0]),int(p[q[4],0]))
        gmsh.model.geo.mesh.setTransfiniteCurve(l2, 2)
        l3 = gmsh.model.geo.addLine(int(p[q[4],0]),int(p[q[j],0]))
        gmsh.model.geo.mesh.setTransfiniteCurve(l3, 2)
        s = gmsh.model.geo.addCurveLoop([l1,l2,l3])
        pyr_surfs.append(gmsh.model.geo.addPlaneSurface([s]))

gmsh.model.geo.addSurfaceLoop([1,2,3,4,5,6]+pyr_surfs, 1)
fluid = gmsh.model.geo.addVolume([1], 1)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(3)

nodes = gmsh.model.mesh.getNodes(3,1,includeBoundary=True)
nodes = np.hstack((nodes[0].reshape((-1,1)),nodes[1].reshape((-1,3))))

#Finding last layer nodes
lvl_min = np.min(TPMS(p[n_lvl*I:(n_lvl+1)*I,1:],Size))
lvl_max = np.max(TPMS(p[n_lvl*I:(n_lvl+1)*I:,1:],Size))
a = np.logical_and(TPMS(nodes[:,1:4],Size) > lvl_min - 1e-6,
                   TPMS(nodes[:,1:4],Size) < lvl_max + 1e-6)
for i in range(n_lvl*I,(n_lvl+1)*I):
    b = np.sum(np.pow(nodes[a,1:4]-p[i,1:4],2),axis=1) < np.pow(Size*(Ds/10),2)
    b = np.argmax(b)
    p[i,0] = int(nodes[a,0][b])
print("Obtained last layer node IDs")

#Finding pyramid peak nodes
lvl_min = np.min(TPMS(p[-n_pyr:,1:],Size))
lvl_max = np.max(TPMS(p[-n_pyr:,1:],Size))
a = np.logical_and(TPMS(nodes[:,1:4],Size) > lvl_min - 1e-6,
                   TPMS(nodes[:,1:4],Size) < lvl_max + 1e-6)
for i in range((n_lvl+1)*I,(n_lvl+1)*I+n_pyr):
    b = np.sum(np.pow(nodes[a,1:4]-p[i,1:4],2),axis=1) < np.pow(Size*(Ds/10),2)
    b = np.argmax(b)
    p[i,0] = int(nodes[a,0][b])
print("Obtained pyramid node IDs")

for i in pyr_surfs:
    gmsh.model.geo.remove([(2,i)])

p_array = p[0:n_lvl*I,1:4].reshape(1,-1)[0]
gmsh.model.mesh.addNodes(3,1,[],p_array)
p[0:n_lvl*I,0] = gmsh.model.mesh.getNodes(3,1)[0][-n_lvl*I:]

nodes = np.zeros(8*n_lvl*n_quad)
for i in range(n_lvl):
    for j in range(n_quad):
        nodes[(n_quad*i+j)*8] = int(p[quads[j,0]-1+i*I,0])
        nodes[(n_quad*i+j)*8+1] = int(p[quads[j,1]-1+i*I,0])
        nodes[(n_quad*i+j)*8+2] = int(p[quads[j,2]-1+i*I,0])
        nodes[(n_quad*i+j)*8+3] = int(p[quads[j,3]-1+i*I,0])
        nodes[(n_quad*i+j)*8+4] = int(p[quads[j,0]-1+(i+1)*I,0])
        nodes[(n_quad*i+j)*8+5] = int(p[quads[j,1]-1+(i+1)*I,0])
        nodes[(n_quad*i+j)*8+6] = int(p[quads[j,2]-1+(i+1)*I,0])
        nodes[(n_quad*i+j)*8+7] = int(p[quads[j,3]-1+(i+1)*I,0])
gmsh.model.mesh.addElementsByType(fluid,5,[],nodes)
print("Added fluid zone hex elements")

nodes = np.zeros(5*n_pyr)
for i in range(n_pyr):
    nodes[5*i] = int(p[pyrs[i,0]-1,0])
    nodes[5*i+1] = int(p[pyrs[i,1]-1,0])
    nodes[5*i+2] = int(p[pyrs[i,2]-1,0])
    nodes[5*i+3] = int(p[pyrs[i,3]-1,0])
    nodes[5*i+4] = int(p[pyrs[i,4]-1,0])
gmsh.model.mesh.addElementsByType(fluid,7,[],nodes)
print("Added pyramid elements")

gmsh.model.geo.addPhysicalGroup(3,[fluid],1,name="Fluid")
gmsh.model.geo.synchronize()

for axis in range(0,3):
    for sign in range(0,2):
        face_pts = np.hstack((p[:I,:],np.arange(I).reshape((-1,1)),))
        face_pts = face_pts[np.pow(p[:I,axis+1]-((-1)**sign)*Size/2,2) <= (Size*Ds/10)**2,:]
        face_pts = face_pts[np.argsort(np.arctan2(face_pts[:,(axis+1)%3+1],face_pts[:,(axis+2)%3+1])),-1]
        face_pts = np.append(face_pts,face_pts[0])
        nodes = np.zeros(4*n_lvl*(face_pts.shape[0]-1))
        for i in range(n_lvl):
            for j in range(face_pts.shape[0]-1):
                nodes[4*((face_pts.shape[0]-1)*i+j)] = int(p[int(face_pts[j])+i*I,0])
                nodes[4*((face_pts.shape[0]-1)*i+j)+1] = int(p[int(face_pts[j+1])+i*I,0])
                nodes[4*((face_pts.shape[0]-1)*i+j)+2] = int(p[int(face_pts[j+1])+(i+1)*I,0])
                nodes[4*((face_pts.shape[0]-1)*i+j)+3] = int(p[int(face_pts[j])+(i+1)*I,0])
        gmsh.model.mesh.addElementsByType(2*axis+sign+1,3,[],nodes)
        namestr = ["x","y","z"][axis]+"_"+["plus","minus"][sign]
        gmsh.model.geo.addPhysicalGroup(2,[2*axis+sign+1],name=namestr)
print("Added unit cell face elements")

s = gmsh.model.add_discrete_entity(2)
nodes = np.zeros(4*n_quad)
for i in range(n_quad):
    nodes[4*i] = int(p[quads[i,0]-1,0])
    nodes[4*i+1] = int(p[quads[i,1]-1,0])
    nodes[4*i+2] = int(p[quads[i,2]-1,0])
    nodes[4*i+3] = int(p[quads[i,3]-1,0])
gmsh.model.mesh.addElementsByType(s,3,[],nodes)
gmsh.model.geo.addPhysicalGroup(2,[s],name="Surface")
print("Added surface elements")

gmsh.model.geo.synchronize()

if '-popup' in sys.argv:
    gmsh.fltk.run()

gmsh.option.setNumber("Mesh.MshFileVersion",2.2)   
gmsh.write("Schwarz-P.msh")
gmsh.finalize()