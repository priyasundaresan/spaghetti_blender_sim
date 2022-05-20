import gym
import torch
import cv2

import bpy, bpy_extras
import time
import numpy as np
import os
from mathutils import Vector
import bmesh

from math import pi
import os
import sys
sys.path.append(os.getcwd())

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.neighbors import NearestNeighbors

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def add_camera_light():
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0,0,0))
    bpy.ops.object.camera_add(location=(0,0,8), rotation=(0,0,0))
    bpy.context.scene.camera = bpy.context.object

def clear_scene():
    '''Clear existing objects in scene'''
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def set_render_settings(engine, render_size, generate_masks=True):
    # Set rendering engine, dimensions, colorspace, images settings
    scene = bpy.context.scene
    scene.world.color = (1, 1, 1)
    scene.render.resolution_percentage = 100
    scene.render.engine = engine
    render_width, render_height = render_size
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    scene.use_nodes = True
    scene.render.image_settings.file_format='JPEG'
    scene.view_settings.exposure = 1.3
    scene.render.image_settings.file_format='JPEG'
    #scene.cycles.samples = 10
    scene.cycles.samples = 1
    scene.view_settings.view_transform = 'Raw'
    scene.cycles.max_bounces = 1
    scene.cycles.min_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transmission_bounces = 1
    scene.cycles.volume_bounces = 1
    scene.cycles.transparent_max_bounces = 1
    scene.cycles.transparent_min_bounces = 1
    scene.view_layers[0].use_pass_object_index = True
    #scene.render.tile_x = 256
    #scene.render.tile_y = 256
    scene.render.use_persistent_data = True
    scene.cycles.device = 'GPU'

def initialize_renderer():
    scene = bpy.context.scene
    tree = bpy.context.scene.node_tree
    links = tree.links
    render_node = tree.nodes["Render Layers"]

    #rl = tree.nodes.new('CompositorNodeRLayers')   
    vl = tree.nodes.new('CompositorNodeViewer')   
    #vl.use_alpha = False

    id_mask_node = tree.nodes.new(type="CompositorNodeIDMask")

    id_mask_node.use_antialiasing = True
    id_mask_node.index = 1
    composite = tree.nodes.new(type = "CompositorNodeComposite")
    links.new(render_node.outputs['IndexOB'], id_mask_node.inputs["ID value"])
    links.new(id_mask_node.outputs[0], composite.inputs["Image"])

    links.new(id_mask_node.outputs[0], vl.inputs[0])  # link Renger Image to Viewer Image
    #links.new(id_mask_node.outputs[2], vl.inputs[1])  # link Render Z to Viewer Alpha

def render(episode):
    bpy.context.scene.render.filepath = 'masks/%05d.jpg'%episode
    bpy.ops.render.render(write_still=True)
    image = None
    #bpy.ops.render.render(write_still=False)
    #pixels = np.array(bpy.data.images['Viewer Node'].pixels)
    #width = bpy.context.scene.render.resolution_x
    #height = bpy.context.scene.render.resolution_y
    #image = pixels.reshape(height,width,4)
    #image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #cv2.imwrite('masks/%05d.jpg'%episode, image)
    return image

def make_fork(path_to_fork_stl):
    bpy.ops.import_mesh.stl(filepath=path_to_fork_stl, filter_glob="*.stl")
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.rigidbody.object_add()
    fork = bpy.context.object
    fork.rigid_body.mass = 5
    fork.location = (0,0,15)
    fork.collision.thickness_inner = 0.2
    fork.collision.damping = 1.0
    fork.collision.stickiness = 0.5
    fork.rigid_body.kinematic = True
    fork.rigid_body.collision_shape = 'MESH'
    fork.rigid_body.use_deform = True
    fork.rigid_body.friction = 1
    fork.rigid_body.collision_margin = 0.01

    return fork

def make_table(params):
    #bpy.ops.import_mesh.stl(filepath='assets/tray.stl', filter_glob="*.stl")
    bpy.ops.mesh.primitive_plane_add(size=params["table_size"], location=(0,0,0))
    bpy.ops.rigidbody.object_add()
    table = bpy.context.object
    table.rigid_body.type = 'PASSIVE'
    #table.rigid_body.friction = 1.2
    table.rigid_body.friction = 1.8
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.object.select_all(action='DESELECT')

def make_pusher():
    location = (0,0,15)
    rotation = (0,np.pi/2,0)
    bpy.ops.mesh.primitive_plane_add(size=1, location=location, rotation=rotation)
    pusher = bpy.context.object
    #pusher.scale = (1,1,0.1)
    #bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.rigidbody.object_add()
    pusher.rigid_body.mass = 25
    pusher.rigid_body.kinematic = True
    pusher.rigid_body.type = 'PASSIVE'

    #pusher.collision.damping = 0.1
    #pusher.collision.thickness_outer = 0.3
    #pusher.collision.thickness_inner = 0.5
    pusher.collision.damping = 0.5
    pusher.collision.thickness_outer = 0.3
    pusher.collision.thickness_inner = 0.5
    bpy.ops.object.select_all(action='DESELECT')
    return pusher

def make_noodle_rig():
    bpy.ops.curve.primitive_bezier_circle_add(radius=0.04, location=(5,0,0))

def freeze_softbody_physics(noodles):
    bpy.context.view_layer.objects.active = noodles
    bpy.ops.object.modifier_apply(modifier="Softbody")

def add_softbody_physics(noodles):
    bpy.context.view_layer.objects.active = noodles
    bpy.ops.object.modifier_add(type='SOFT_BODY')
    noodles.modifiers["Softbody"].settings.mass = 1.3
    #noodles.modifiers["Softbody"].settings.ball_stiff = 1
    noodles.modifiers["Softbody"].settings.ball_stiff = 1.2
    noodles.modifiers["Softbody"].settings.ball_damp = 0.8

    noodles.modifiers["Softbody"].show_in_editmode = True
    noodles.modifiers["Softbody"].show_on_cage = True

    noodles.modifiers["Softbody"].settings.use_goal = False
    noodles.modifiers["Softbody"].settings.use_edges = True
    noodles.modifiers["Softbody"].settings.use_self_collision = True
    noodles.data.bevel_object = bpy.data.objects["BezierCircle"]
    bpy.context.view_layer.objects.active = noodles
    bpy.context.object.data.bevel_mode = 'OBJECT'

    bpy.ops.object.select_all(action='DESELECT')

    #noodles.modifiers["Softbody"].settings.step_max = 250

    return noodles

def annotate(points):
    scene = bpy.context.scene
    render_width = scene.render.resolution_x 
    render_height = scene.render.resolution_y 
    pixels = []
    for pt in points:
        camera_coord = bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, bpy.context.scene.camera, Vector(pt))
        pixel = [round(camera_coord.x * render_width), round(render_height - camera_coord.y * render_height)]
        pixels.append(pixel)
    return pixels

def make_noodle():
    location = np.random.uniform(-0.3,0.3,3)
    location[2] = np.random.uniform(0.25,1.00)
    rotation = np.array([np.random.uniform(-0.02, 0.02),np.random.uniform(-0.02, 0.02),np.random.uniform(0, np.pi)])
    bpy.ops.curve.primitive_nurbs_path_add(radius=1.0, enter_editmode=False, align='WORLD', location=location, rotation=rotation, scale=(1,1,1))
    bpy.ops.object.editmode_toggle()

    bpy.ops.curve.subdivide(number_cuts=2) 
    #bpy.ops.curve.subdivide(number_cuts=3) 
    bpy.ops.object.editmode_toggle()
    path = bpy.context.object

    bpy.ops.object.modifier_add(type='SOFT_BODY')
    path.modifiers["Softbody"].settings.mass = 2
    path.modifiers["Softbody"].settings.ball_stiff = 1
    path.modifiers["Softbody"].settings.ball_damp = 0.8

    path.modifiers["Softbody"].show_in_editmode = True
    path.modifiers["Softbody"].show_on_cage = True

    path.modifiers["Softbody"].settings.use_goal = False
    path.modifiers["Softbody"].settings.use_edges = True
    path.modifiers["Softbody"].settings.use_self_collision = True
    path.data.bevel_object = bpy.data.objects["BezierCircle"]
    bpy.context.view_layer.objects.active = path
    bpy.context.object.data.bevel_mode = 'OBJECT'

    bpy.ops.object.select_all(action='DESELECT')

    for curve in path.data.splines:
        for point in curve.points:
            point.select = False

    return path

def make_noodle_pile(n_noodles):
    noodles = []
    bpy.ops.object.select_all(action='DESELECT')
    for i in range(n_noodles):
        noodle = make_noodle()
        noodles.append(noodle)
    for noodle in noodles:
        noodle.select_set(True)
    bpy.ops.object.join()
    noodles = bpy.context.object
    noodles.pass_index = 1

    for step in range(0, 30, 1):
        bpy.context.scene.frame_set(step)

    empties = None
    return noodles, empties

def delete_objs(objs):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.delete()


def push(pusher, start_frame, push_duration, lift_duration, push_start_2d, push_end_2d, hull_2d, densest_3d, annot_dir='annots'):
    #push_end_2d = densest_3d[:-1]
    push_end_2d = [0,0]

    offset = push_end_2d - push_start_2d
    angle = np.arctan(offset[1]/offset[0])

    push_start_2d -= offset*0.1
    pusher.location = np.array([push_start_2d[0], push_start_2d[1], 0.5])
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame)

    pusher.location = np.array([push_end_2d[0], push_end_2d[1], 0.6])
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame+push_duration)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame+push_duration)

    pusher.location = np.array([push_end_2d[0], push_end_2d[1], 0.75])
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame+push_duration+lift_duration)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame+push_duration+lift_duration)

    for step in range(start_frame, start_frame+push_duration+lift_duration):
        bpy.context.scene.frame_set(step)
        x,y,z = pusher.matrix_world.translation
        pixels = annotate([[x,y,z]] + [[h[0],h[1],0] for h in hull_2d] + [densest_3d])
        #bpy.context.scene.camera.location = [densest_2d[0], densest_2d[1], bpy.context.scene.camera.location[2]]
        render(step-30)
        np.save('%s/%03d.npy'%(annot_dir,step-30), np.array([pixels]))

    return start_frame+push_duration

def wait(start_frame, wait_duration):
    for step in range(start_frame, start_frame+wait_duration):
        bpy.context.scene.frame_set(step)
        render(step-30)
    return start_frame+wait_duration

def twirl(fork, start_frame, down_duration, twirl_duration, scoop_duration, twirl_start_3d, angle, annot_dir='annots'):
    x,y,z = twirl_start_3d

    fork.location = (x,y,2.0)
    fork.rotation_euler = (0,0,angle)
    fork.keyframe_insert(data_path="location", frame=start_frame)
    fork.keyframe_insert(data_path="rotation_euler", frame=start_frame)

    fork.location = (x,y,0.5)
    fork.rotation_euler = (0,0,angle)
    fork.keyframe_insert(data_path="location", frame=start_frame+down_duration)
    fork.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration)

    fork.location = (x,y,0.5)
    fork.rotation_euler = (0,0,angle+6*np.pi)
    fork.keyframe_insert(data_path="location", frame=start_frame+down_duration+twirl_duration)
    fork.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration+twirl_duration)

    fork.location = (x,y,2.0)
    fork.rotation_euler = (0,np.pi/2,angle+6*np.pi)
    fork.keyframe_insert(data_path="location", frame=start_frame+down_duration+twirl_duration+scoop_duration)
    fork.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration+twirl_duration+scoop_duration)

    for step in range(start_frame, start_frame+down_duration+twirl_duration+scoop_duration):
        bpy.context.scene.frame_set(step)
        pixels = annotate([twirl_start_3d])
        render(step-30)
        np.save('%s/%03d.npy'%(annot_dir,step-30), np.array([pixels]))

def densest_point(noodles):
    start = time.time()
    points = []
    for curve in noodles.data.splines:
        for point in curve.points:
            point = noodles.matrix_world@point.co
            point = point[:3]
            points.append(point)
    points = np.array(points)
    points_2d = points[:,:2]

    neigh = NearestNeighbors()
    neigh.fit(points)

    min_dist = float('inf')
    densest_point = None
    for idx, point in enumerate(points):
        dists, match_idxs = neigh.kneighbors([point], len(points), return_distance=True) 
        cumulative_dist = sum(dists.squeeze())
        if cumulative_dist <= min_dist:
            densest_point = point
            min_dist = cumulative_dist
    end = time.time()

    return densest_point

def densest_point_angle(noodles):
    points = []
    for curve in noodles.data.splines:
        for point in curve.points:
            point = noodles.matrix_world@point.co
            point = point[:3]
            points.append(point)
    points = np.array(points)

    neigh = NearestNeighbors()
    neigh.fit(points)

    min_dist = float('inf')
    densest_point = None

    densest_noodle_idx = None
    densest_point_idx = None

    for noodle_idx, curve in enumerate(noodles.data.splines):
        for point_idx, point in enumerate(curve.points):
            point = noodles.matrix_world@point.co
            point = point[:3]
            dists, match_idxs = neigh.kneighbors([point], len(points), return_distance=True) 
            cumulative_dist = sum(dists.squeeze())
            if cumulative_dist <= min_dist:
                densest_noodle_idx = noodle_idx
                densest_point_idx = point_idx
                densest_point = point
                min_dist = cumulative_dist

    densest_point_neighbor_idx = densest_point_idx + 1 if (densest_point_idx + 1 < len(curve.points)) else densest_point_idx - 1
    densest_point_neighbor = noodles.data.splines[densest_noodle_idx].points[densest_point_neighbor_idx]
    densest_point_neighbor = noodles.matrix_world@densest_point_neighbor.co
    densest_point_neighbor = densest_point_neighbor[:3]

    #neighboring_noodle_points = [densest_point, densest_point_neighbor]
    offset = np.array(densest_point_neighbor[:2]) - np.array(densest_point[:2])
    angle = np.arctan(offset[1]/offset[0])

    return densest_point, angle

def reset_pusher(pusher, frame):
    pusher.location = (0,0,15)
    pusher.rotation_euler = (0,np.pi/2,0)
    #pusher.rigid_body.kinematic = False
    #pusher.keyframe_insert(data_path="location", frame=frame)
    #pusher.keyframe_insert(data_path="rotation_euler", frame=frame)

def reset_fork(fork, frame):
    fork.location = (0,0,15)
    fork.rotation_euler = (0,0,0)
    #fork.keyframe_insert(data_path="location", frame=frame)
    #fork.keyframe_insert(data_path="rotation_euler", frame=frame)

def noodle_state(noodles):
    points = []
    for curve in noodles.data.splines:
        for point in curve.points:
            point = noodles.matrix_world@point.co
            point = point[:3]
            points.append(point)
    points = np.array(points)
    points_2d = points[:,:2]
    start = time.time()
    hull = ConvexHull(points_2d)
    end = time.time()

   # center_2d = np.mean(points_2d, axis=0)
    center_2d = np.zeros(2)
    hull_points_2d = points_2d[hull.vertices]

    neigh = NearestNeighbors()
    neigh.fit(hull_points_2d)
    match_idxs = neigh.kneighbors([center_2d], len(hull.vertices), return_distance=False) 
    furthest_idx = match_idxs.squeeze().tolist()[-1]
    furthest_2d = hull_points_2d[furthest_idx]

    print('hull area', hull.volume)
    print('center', center_2d)
    print('furthest_2d', furthest_2d)
    
    return hull_points_2d, center_2d, furthest_2d, hull.volume, densest_point(noodles)

def remove_picked_up(noodles):
    bpy.ops.object.select_all(action='DESELECT')
    z_thresh = 0.5
    points = []
    pickedup_noodle_idxs  = []
    for curve_idx, curve in enumerate(noodles.data.splines):
        picked_up = False
        for point in curve.points:
            point = noodles.matrix_world@point.co
            point = point[:3]
            if point[2]  > z_thresh:
                picked_up = True
                break   
        if picked_up:
            pickedup_noodle_idxs.append(curve_idx)

    curve_idx = 0
    for curve_idx in pickedup_noodle_idxs:
        for point_idx, point in enumerate(noodles.data.splines[curve_idx].points):
            noodles.data.splines[curve_idx].points[point_idx].select = True

    bpy.ops.object.editmode_toggle()
    bpy.ops.curve.delete(type='VERT')
    bpy.ops.object.editmode_toggle()
    return noodles



def generate_dataset(episodes):
    if not os.path.exists('masks'):
        os.mkdir('masks')
    if not os.path.exists('annots'):
        os.mkdir('annots')

    render_size = (140,140)
    #render_size = (64,64)
    set_render_settings('CYCLES', render_size)
    clear_scene()
    camera = add_camera_light()
    params = {"table_size":10}
    make_table(params)
    make_pusher()
    make_noodle_rig()
    initialize_renderer()

    for episode in range(episodes):
        noodles, empties = make_noodle_pile(15)
        render(episode)
        #delete_objs([noodles])

    # Grouping
    pusher = make_pusher()
    areas = []
    i = 0
    area = float('inf')
    while area > 6:
        if i>5:
            break
        freeze_softbody_physics(noodles)
        hull_2d, center_2d, furthest_2d, area, densest_3d = noodle_state(noodles)
        add_softbody_physics(noodles)
        areas.append(area)
        start = push(pusher, 30+(i*15), 10, 5, furthest_2d, center_2d, hull_2d, densest_3d)
        i+=1
    reset_pusher(pusher, bpy.context.scene.frame_current)

    start_frame = 30+(i*15)
    freeze_softbody_physics(noodles)
    densest_3d, angle = densest_point_angle(noodles)
    add_softbody_physics(noodles)
    fork = make_fork('assets/fork.stl')

    twirl(fork, 30+start_frame, 5, 20, 20, densest_3d, angle)
    freeze_softbody_physics(noodles)
    remove_picked_up(noodles)
    reset_fork(fork, bpy.context.scene.frame_current)

    wait(30+start_frame+45,30)

if __name__ == '__main__':
    generate_dataset(1)
