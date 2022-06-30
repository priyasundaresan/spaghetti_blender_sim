import random
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

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.neighbors import NearestNeighbors

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
    scene.render.use_persistent_data = True
    scene.cycles.device = 'GPU'

def initialize_renderer():
    scene = bpy.context.scene
    tree = bpy.context.scene.node_tree
    links = tree.links
    render_node = tree.nodes["Render Layers"]

    vl = tree.nodes.new('CompositorNodeViewer')   

    id_mask_node = tree.nodes.new(type="CompositorNodeIDMask")

    id_mask_node.use_antialiasing = True
    id_mask_node.index = 1
    composite = tree.nodes.new(type = "CompositorNodeComposite")
    links.new(render_node.outputs['IndexOB'], id_mask_node.inputs["ID value"])
    links.new(id_mask_node.outputs[0], composite.inputs["Image"])

    links.new(id_mask_node.outputs[0], vl.inputs[0])  # link Renger Image to Viewer Image

def render(episode):
    bpy.context.scene.render.filepath = 'masks/%05d.jpg'%episode
    bpy.ops.render.render(write_still=True)
    #image = cv2.imread('masks/%05d.jpg'%episode)
    pixels = np.array(bpy.data.images['Viewer Node'].pixels)
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    image = pixels.reshape(height,width,4)[:,:,:3]
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    #cv2.imshow('img', image)
    #cv2.waitKey(0)
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
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.rigidbody.object_add()
    pusher.rigid_body.mass = 25
    pusher.rigid_body.kinematic = True
    pusher.rigid_body.type = 'PASSIVE'

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

def get_coverage_pickup_stats(noodles):
    try:
        freeze_softbody_physics(noodles)
    except:
        pass
        #print('here')
    hull_2d, center_2d, furthest_2d, area, densest_3d = noodle_state(noodles)
    add_softbody_physics(noodles)
    coverage = area
    num_noodles_left = len(noodles.data.splines)
    return coverage, num_noodles_left 

def generate_heldout_noodle_state(num_noodles):
    locations = []
    rotations = []
    for i in range(num_noodles):
        np.random.seed(num_noodles+i)
        location = np.random.uniform(-1.1,1.1,3)
        location[2] = np.random.uniform(0.25,1.00)
        np.random.seed(num_noodles+i)
        rotation = np.array([np.random.uniform(-0.4, 0.4),np.random.uniform(-0.4, 0.4),np.random.uniform(0, np.pi)])
        locations.append(location)
        rotations.append(rotation)
    return locations, rotations

def make_noodle(location=None, rotation=None):
    np.random.seed()
    if location is None:
        location = np.random.uniform(-1.1,1.1,3)
        location[2] = np.random.uniform(0.25,1.00)
    if rotation is None:
        rotation = np.array([np.random.uniform(-0.4, 0.4),np.random.uniform(-0.4, 0.4),np.random.uniform(0, np.pi)])
    bpy.ops.curve.primitive_nurbs_path_add(radius=1.0, enter_editmode=False, align='WORLD', location=location, rotation=rotation, scale=(1,1,1))
    bpy.ops.object.editmode_toggle()

    bpy.ops.curve.subdivide(number_cuts=2) 
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

def make_noodle_pile(n_noodles, deterministic=False, settle_time=30):
    noodles = []
    bpy.ops.object.select_all(action='DESELECT')

    if deterministic:
        locations, rotations = generate_heldout_noodle_state(n_noodles)

    for i in range(n_noodles):
        if deterministic:
            loc, rot = locations[i], rotations[i]
        else:
            loc, rot = None, None
        noodle = make_noodle(location=loc, rotation=rot)
        noodles.append(noodle)
    for noodle in noodles:
        noodle.select_set(True)
    bpy.ops.object.join()
    noodles = bpy.context.object
    noodles.pass_index = 1

    for step in range(bpy.context.scene.frame_current, bpy.context.scene.frame_current + settle_time):
        bpy.context.scene.frame_set(step)

    return noodles

def delete_objs(objs):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
    bpy.ops.object.delete()


def push(pusher, push_duration, lift_duration, push_start_2d, push_end_2d, hull_2d=None, densest_3d=None, annot_dir='annots'):
    start_frame = bpy.context.scene.frame_current
    
    #push_end_2d = [0,0]

    offset = push_end_2d - push_start_2d
    angle = np.arctan(offset[1]/offset[0])

    push_start_2d -= offset*0.15
    pusher.location = np.array([push_start_2d[0], push_start_2d[1], 0.45])
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame)

    pusher.location = np.array([push_end_2d[0], push_end_2d[1], 0.45])
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame+push_duration)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame+push_duration)

    pusher.location = np.array([push_end_2d[0], push_end_2d[1], 0.75])
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame+push_duration+lift_duration)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame+push_duration+lift_duration)
    
    pixels = annotate([[push_start_2d[0], push_start_2d[1], 0], [push_end_2d[0], push_end_2d[1], 0]])
    for step in range(start_frame, start_frame+push_duration+lift_duration):
        bpy.context.scene.frame_set(step)
        #x,y,z = pusher.matrix_world.translation
        #pixels = annotate([[x,y,z]] + [[h[0],h[1],0] for h in hull_2d] + [densest_3d])
        #render(step-30)
        #np.save('%s/%03d.npy'%(annot_dir,step-30), np.array([pixels]))

    #render(step-30)

    #bpy.context.scene.camera.location[0] = push_end_2d[0]
    #bpy.context.scene.camera.location[1] = push_end_2d[1]
    return pixels

def wait(wait_duration):
    start_frame = bpy.context.scene.frame_current
    for step in range(start_frame, start_frame+wait_duration):
        bpy.context.scene.frame_set(step)

def twirl(fork, down_duration, twirl_duration, scoop_duration, wait_duration, twirl_start_3d, angle, annot_dir='annots'):
    start_frame = bpy.context.scene.frame_current
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

    pixels = annotate([twirl_start_3d])
    for step in range(start_frame, start_frame+down_duration+twirl_duration+scoop_duration):
        bpy.context.scene.frame_set(step)
        #pixels = annotate([twirl_start_3d])
        #render(step-30)
        #np.save('%s/%03d.npy'%(annot_dir,step-30), np.array([pixels]))

    wait(wait_duration)

    #bpy.context.scene.camera.location[0] = twirl_start_3d[0]
    #bpy.context.scene.camera.location[1] = twirl_start_3d[1]

    #render(step+wait_duration-30)
    return pixels

def densest_point(noodles, return_avg_density=False):
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
    all_dists = []
    for idx, point in enumerate(points):
        dists, match_idxs = neigh.kneighbors([point], len(points), return_distance=True) 
        cumulative_dist = sum(dists.squeeze())
        all_dists.append(cumulative_dist)
        if cumulative_dist <= min_dist:
            densest_point = point
            min_dist = cumulative_dist
    end = time.time()

    if not return_avg_density:
        return densest_point
    else:
        return densest_point, np.mean(all_dists)

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

    offset = np.array(densest_point_neighbor[:2]) - np.array(densest_point[:2])
    angle = np.arctan(offset[1]/offset[0])

    return densest_point, angle

def reset_pusher(pusher):
    pusher.location = (15,0,0)
    pusher.rotation_euler = (0,np.pi/2,0)
    pusher.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)
    pusher.keyframe_insert(data_path="rotation_euler", frame=bpy.context.scene.frame_current)

def reset_fork(fork):
    fork.location = (15,0,0)
    fork.rotation_euler = (0,0,0)
    fork.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)
    fork.keyframe_insert(data_path="rotation_euler", frame=bpy.context.scene.frame_current)

def noodle_state(noodles):
    points = []
    for curve in noodles.data.splines:
        for point in curve.points:
            point = noodles.matrix_world@point.co
            point = point[:3]
            points.append(point)
    points = np.array(points)

    if len(points):
        densest_3d = densest_point(noodles)
        points_2d = points[:,:2]
        start = time.time()
        hull = ConvexHull(points_2d)
        end = time.time()

        #center_2d = np.mean(points_2d, axis=0)
        #center_2d = np.zeros(2)
        #print(center_2d)
        center_2d = densest_3d[:2]
        hull_points_2d = points_2d[hull.vertices]

        neigh = NearestNeighbors()
        neigh.fit(hull_points_2d)
        match_idxs = neigh.kneighbors([center_2d], len(hull.vertices), return_distance=False) 
        furthest_idx = match_idxs.squeeze().tolist()[-1]
        furthest_2d = hull_points_2d[furthest_idx]

        #print('hull area', hull.volume)
        #print('center', center_2d)
        #print('furthest_2d', furthest_2d)
        
        return hull_points_2d, center_2d, furthest_2d, hull.volume, densest_3d
    else:
        return np.zeros(2), np.zeros(2), np.zeros(2), 0, np.zeros(2)

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

def clear_actions_frames():
    bpy.context.scene.frame_set(0)
    for a in bpy.data.actions:
        bpy.data.actions.remove(a)

def clear_noodles():
    objs = []
    for obj in bpy.data.objects:
        if "Nurbs" in obj.name:
            objs.append(obj)
    delete_objs(objs)

def initialize_sim():
    if not os.path.exists('masks'):
        os.mkdir('masks')
    if not os.path.exists('annots'):
        os.mkdir('annots')

    render_size = (256,256)
    set_render_settings('CYCLES', render_size)
    clear_scene()
    clear_actions_frames()
    camera = add_camera_light()
    params = {"table_size":10}
    make_table(params)
    make_noodle_rig()
    initialize_renderer()
    pusher = make_pusher()
    reset_pusher(pusher)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fork = make_fork('%s/assets/fork.stl'%dir_path)
    reset_fork(fork)

    return pusher, fork

def reset_sim(pusher, fork, num_noodles, deterministic=False):
    reset_pusher(pusher)
    reset_fork(fork)
    noodles = make_noodle_pile(num_noodles, deterministic=deterministic)
    clear_actions_frames()
    return noodles

def take_push_action(pusher, noodles, push_start_2d=None, push_end_2d=None):
    freeze_softbody_physics(noodles)
    if push_start_2d is None and push_end_2d is None:
        hull_2d, center_2d, furthest_2d, area, densest_3d = noodle_state(noodles)
    add_softbody_physics(noodles)
    push_duration = 15
    lift_duration = 5
    pixels = push(pusher, push_duration, lift_duration, furthest_2d, center_2d, hull_2d, densest_3d)
    return pixels

def take_twirl_action(fork, noodles):
    freeze_softbody_physics(noodles)
    densest_3d, angle = densest_point_angle(noodles)
    add_softbody_physics(noodles)
    #down_duration = 5
    #twirl_duration = 20
    down_duration = 10
    twirl_duration = 13
    lift_duration = 10
    wait_duration = 10
    pixels = twirl(fork, down_duration, twirl_duration, lift_duration, wait_duration, densest_3d, angle)
    freeze_softbody_physics(noodles)
    remove_picked_up(noodles)
    add_softbody_physics(noodles)
    reset_fork(fork)
    return pixels


def generate_dataset(episodes, pusher, fork):
    initial_noodles = 15
    noodles = reset_sim(pusher, fork, initial_noodles)

    initial_area, initial_num_noodles = get_coverage_pickup_stats(noodles)
    rewards = []
    for i in range(10):
        render(bpy.context.scene.frame_current-30)
        if random.random()<0.8:
            take_push_action(pusher, noodles)
        else:
            take_twirl_action(fork, noodles)
        area, num_noodles = get_coverage_pickup_stats(noodles)
        rewards.append([initial_area-area, initial_num_noodles-num_noodles])
        initial_area = area
        initial_num_noodles = num_noodles

    #print('rewards: area, noodles')
    #for r in rewards:
    #    print(r)

    delete_objs([noodles])

if __name__ == '__main__':
    pusher, fork = initialize_sim()
    generate_dataset(1, pusher, fork)
    generate_dataset(1, pusher, fork)
    #delete_objs([noodles])
    #noodles = generate_dataset(1, pusher, fork)
