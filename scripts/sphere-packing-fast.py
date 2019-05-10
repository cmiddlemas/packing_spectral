# Author: Michael Klatt
# Usage: blender --python sphere-packing.py <infile>
import bpy
import bmesh
import mathutils
import numpy as np
import math

#Global parameter
sphere_subdivisions = 4

#code snippet from blender.stackexchange.com/questions/6817
import sys
argv = sys.argv
try:
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
except ValueError:
    print("ERROR: missing sphere packing file")
    print("Expecting blender --python sphere-packing.py -- <infile>")
    sys.exit(-1)
if len(argv) == 0:
    print("ERROR: missing sphere packing file")
    print("Expecting blender --python sphere-packing.py -- <infile>")
    sys.exit(-1)
else:
    infile = argv[0]
try:
    f = open(infile,"r")
except FileNotFoundError:
    print("ERROR: could not find file '"+infile+"'")
    sys.exit(-1)

def line2floatarray(line):
    strl = (line[:-1]).replace('\t',' ').split(' ')
    if strl[-1] == '':
        strl = strl[:-1]
    print(strl)
    return(np.array([float(c) for c in strl]))

lnr = 0
particles = []
for line in f:
    lnr += 1
    if lnr == 3:
        Nsph = int(line[:-1])
    elif lnr == 4:
        x = line2floatarray(line)
    elif lnr == 5:
        y = line2floatarray(line)
    elif lnr == 6:
        z = line2floatarray(line)
    elif lnr > 7:
        particles += [line2floatarray(line)]
particles = np.vstack(particles)
R_max = np.max(particles[:,-1])
R_min = np.min(particles[:,-1])

def R2col(R):
    gr = (R-R_min)/(R_max-R_min)
    return(0,gr,1.0-gr)

scene = bpy.context.scene
mesh = bpy.data.meshes.new('HS')

bm = bmesh.new()
bmesh.ops.create_icosphere(bm, subdivisions=sphere_subdivisions, diameter=1.0) # Note bug, diameter is actually radius (at least in my version)
bm.to_mesh(mesh)
bm.free()

for i in range(Nsph):
    #code snippet from gist.github.com/anonymous/7799e5f41198e41d0825
    name = "sphere.%07d" % (i)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = particles[i,:-1]
    R = particles[i,-1]
    obj.scale = [R,R,R]
    scene.objects.link(obj)

scene.update()

