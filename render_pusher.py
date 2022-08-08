import random
#import torch
#import cv2

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
    #bpy.ops.object.camera_add(location=(0,0,8), rotation=(0,0,0))
    bpy.ops.object.camera_add(location=(0,0,12), rotation=(0,0,0))
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

    bpy.ops.rigidbody.world_add()
    bpy.context.scene.rigidbody_world.point_cache.frame_end = 2000
    bpy.context.scene.frame_end = 2000


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
    #image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #cv2.imshow('img', image)
    #cv2.waitKey(0)
    #cv2.imwrite('masks/%05d.jpg'%episode, image)
    return image

def make_walls(params):
    dim = params["table_size"]

    bpy.ops.mesh.primitive_plane_add(size=1, location=(dim/2,0,0.5), rotation=(0,np.pi/2,0))
    bpy.ops.rigidbody.object_add()
    wall = bpy.context.object
    wall.rigid_body.type = 'PASSIVE'
    wall.rigid_body.friction = 1.0
    wall.scale = (1,dim,1)
    bpy.ops.object.modifier_add(type='COLLISION')

    bpy.ops.mesh.primitive_plane_add(size=1, location=(-dim/2,0,0.5), rotation=(0,np.pi/2,0))
    bpy.ops.rigidbody.object_add()
    wall = bpy.context.object
    wall.rigid_body.type = 'PASSIVE'
    wall.rigid_body.friction = 1.0
    wall.scale = (1,dim,1)
    bpy.ops.object.modifier_add(type='COLLISION')

    bpy.ops.mesh.primitive_plane_add(size=1, location=(0,dim/2,0.5), rotation=(np.pi/2,0,0))
    bpy.ops.rigidbody.object_add()
    wall = bpy.context.object
    wall.rigid_body.type = 'PASSIVE'
    wall.rigid_body.friction = 1.0
    wall.scale = (dim,1,1)
    bpy.ops.object.modifier_add(type='COLLISION')

    bpy.ops.mesh.primitive_plane_add(size=1, location=(0,-dim/2,0.5), rotation=(np.pi/2,0,0))
    bpy.ops.rigidbody.object_add()
    wall = bpy.context.object
    wall.rigid_body.type = 'PASSIVE'
    wall.rigid_body.friction = 1.0
    wall.scale = (dim,1,1)
    bpy.ops.object.modifier_add(type='COLLISION')

    bpy.ops.object.select_all(action='DESELECT')

def make_table(params):
    bpy.ops.mesh.primitive_plane_add(size=params["table_size"], location=(0,0,0))
    bpy.ops.rigidbody.object_add()
    table = bpy.context.object
    table.rigid_body.type = 'PASSIVE'
    #table.rigid_body.friction = 1.2
    table.rigid_body.friction = 1.0
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.object.select_all(action='DESELECT')

def make_scooper(path_to_scooper_stl):
    bpy.ops.import_mesh.stl(filepath=path_to_scooper_stl, filter_glob="*.stl")
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.rigidbody.object_add()
    scooper = bpy.context.object
    scooper.location = (0,0,15)
    scooper.rigid_body.kinematic = True
    scooper.rigid_body.collision_shape = 'MESH'
    scooper.pass_index = 1
    return scooper

def make_pusher():
    location = (0,0,15)
    rotation = (0,np.pi/2,0)
    bpy.ops.mesh.primitive_plane_add(size=2, location=location, rotation=rotation)
    pusher = bpy.context.object
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.rigidbody.object_add()
    pusher.rigid_body.kinematic = True
    bpy.ops.object.select_all(action='DESELECT')
    pusher.pass_index = 1
    return pusher

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

def get_coverage_pickup_stats(items):
    hull_2d, center_2d, furthest_2d, area, densest_3d = items_state(items)
    coverage = area
    num_items_left = len(items)
    return coverage, num_items_left 

def generate_heldout_pile_state(num_items):
    RANDOM_SEED = 3
    locations = []
    rotations = []
    for i in range(num_items):
        np.random.seed(num_items+i+RANDOM_SEED)
        location = np.random.uniform(-1.1,1.1,3)
        #location[2] = np.random.uniform(0.25,1.00)
        location[2] = np.random.uniform(0.15,0.25)
        np.random.seed(num_items+i+RANDOM_SEED)
        rotation = np.array([np.random.uniform(-0.4, 0.4),np.random.uniform(-0.4, 0.4),np.random.uniform(0, np.pi)])
        locations.append(location)
        rotations.append(rotation)
    return locations, rotations

def push(pusher, down_duration, push_duration, lift_duration, wait_duration, push_start_2d, push_end_2d, hull_2d=None, densest_3d=None, annot_dir='annots'):
    start_frame = bpy.context.scene.frame_current
    
    offset = push_end_2d - push_start_2d
    angle = np.arctan(offset[1]/offset[0])

    #push_start_2d -= offset*0.15
    push_start_2d -= offset*0.05

    pusher.location = np.array([push_start_2d[0], push_start_2d[1], 2.0])
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame)

    pusher.location = np.array([push_start_2d[0], push_start_2d[1], 1.0])
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame+down_duration)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration)

    pusher.location = np.array([push_end_2d[0], push_end_2d[1], 1.0])
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame+down_duration+push_duration)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration+push_duration)

    pusher.location = np.array([push_end_2d[0], push_end_2d[1], 2.0])
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame+down_duration+push_duration+lift_duration)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration+push_duration+lift_duration)
    
    pixels = annotate([[push_start_2d[0], push_start_2d[1], 0], [push_end_2d[0], push_end_2d[1], 0]])

    wait(wait_duration)

    for step in range(start_frame, start_frame+down_duration+push_duration+lift_duration+wait_duration):
        bpy.context.scene.frame_set(step)
        render(step-30)
        np.save('%s/%03d.npy'%(annot_dir,step-30), np.array([pixels]))
    return pixels

def scoop(pusher, scooper, down_duration, scoop_duration, lift_duration, wait_duration, densest_3d, annot_dir='annots'):
    start_frame = bpy.context.scene.frame_current
    x,y,z = densest_3d
    angle = 0

    pusher.location = (x-3.0,y,2.0)
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame)

    scooper.location = (x+3.0,y,2.0)
    scooper.rotation_euler = (0,0,angle)
    scooper.keyframe_insert(data_path="location", frame=start_frame)
    scooper.keyframe_insert(data_path="rotation_euler", frame=start_frame)

    pusher.location = (x-3.0,y,1.0)
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame+down_duration)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration)

    scooper.location = (x+3.0,y,1.3)
    scooper.rotation_euler = (0,0,angle)
    scooper.keyframe_insert(data_path="location", frame=start_frame+down_duration)
    scooper.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration)

    pusher.location = (x,y,1.0)
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame+down_duration+scoop_duration)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration+scoop_duration)

    scooper.location = (x+2,y,1.3)
    scooper.rotation_euler = (0,0,angle)
    scooper.keyframe_insert(data_path="location", frame=start_frame+down_duration+scoop_duration)
    scooper.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration+scoop_duration)

    pusher.location = (x,y,2.0)
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame+down_duration+scoop_duration+lift_duration)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration+scoop_duration+lift_duration)

    scooper.location = (x+2,y,2.0)
    scooper.rotation_euler = (0,np.pi/12,angle)
    scooper.keyframe_insert(data_path="location", frame=start_frame+down_duration+scoop_duration+lift_duration)
    scooper.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration+scoop_duration+lift_duration)

    wait(wait_duration)

    pixels = annotate([densest_3d])

    for step in range(start_frame, start_frame+down_duration+scoop_duration+lift_duration+wait_duration):
        np.save('%s/%03d.npy'%(annot_dir,step-30), np.array([pixels]))
        bpy.context.scene.frame_set(step)
        render(step-30)
    return pixels

def wait(wait_duration):
    start_frame = bpy.context.scene.frame_current
    for step in range(start_frame, start_frame+wait_duration):
        bpy.context.scene.frame_set(step)

def clear_actions_frames():
    bpy.context.scene.frame_set(0)
    for a in bpy.data.actions:
        bpy.data.actions.remove(a)

def clear_items():
    objs = []
    for obj in bpy.data.objects:
        #if "Sphere" in obj.name:
        if "Cube" in obj.name:
            objs.append(obj)
    delete_objs(objs)

def reset_pusher(pusher):
    pusher.location = (0,0,15)
    pusher.rotation_euler = (0,np.pi/2,0)
    pusher.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)
    pusher.keyframe_insert(data_path="rotation_euler", frame=bpy.context.scene.frame_current)

def reset_scooper(scooper):
    scooper.location = (0,0,15)
    scooper.rotation_euler = (0,0,0)
    scooper.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)
    scooper.keyframe_insert(data_path="rotation_euler", frame=bpy.context.scene.frame_current)

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
    params = {"table_size":6}
    make_table(params)
    make_walls(params)
    initialize_renderer()
    pusher = make_pusher()
    reset_pusher(pusher)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    scooper = make_scooper('%s/assets/scooper.stl'%dir_path)
    reset_scooper(scooper)

    return pusher, scooper

def reset_sim(pusher, scooper, num_items, deterministic=False):
    reset_pusher(pusher)
    reset_scooper(scooper)
    clear_actions_frames()
    items = make_pile(num_items, deterministic=deterministic, settle_time=30)
    print(bpy.context.scene.frame_current)
    return items

def take_push_action(pusher, scooper, items, push_start_2d=None, push_end_2d=None):
    if push_start_2d is None and push_end_2d is None:
        hull_2d, center_2d, furthest_2d, area, densest_3d = items_state(items)
    down_duration = lift_duration = 5
    push_duration = 45
    wait_duration = 20
    #pixels = push(pusher, push_duration, lift_duration, furthest_2d, center_2d, hull_2d, densest_3d)
    pixels = push(pusher, down_duration, push_duration, lift_duration, wait_duration, furthest_2d, center_2d, hull_2d, densest_3d)
    reset_pusher(pusher)
    reset_scooper(scooper)
    return pixels

def take_scoop_action(pusher, scooper, items):
    densest_3d = densest_point(items)
    down_duration = 10
    scoop_duration = 45
    lift_duration = 10
    wait_duration = 5
    pixels = scoop(pusher, scooper, down_duration, scoop_duration, lift_duration, wait_duration, densest_3d)
    items = remove_picked_up(items)
    reset_pusher(pusher)
    reset_scooper(scooper)
    return items, pixels

def remove_picked_up(items):
    bpy.ops.object.select_all(action='DESELECT')
    z_lower = 0.0
    z_upper = 0.5
    remaining_items = []
    for item in items:
        x,y,z = item.matrix_world.translation
        picked_up = (z < z_lower) or (z > z_upper)
        if picked_up:
            item.select_set(True)
        else:
            remaining_items.append(item)
    bpy.ops.object.delete()
    return remaining_items

def items_state(items):
    points = []
    for item in items:
        point = item.matrix_world.translation
        points.append(point)
    points = np.array(points)
    if len(points) > 2:
        densest_3d = densest_point(items)
        points_2d = points[:,:2]

        hull = ConvexHull(points_2d)
        center_2d = densest_3d[:2]
        hull_points_2d = points_2d[hull.vertices]

        neigh = NearestNeighbors()
        neigh.fit(hull_points_2d)
        match_idxs = neigh.kneighbors([center_2d], len(hull.vertices), return_distance=False) 
        furthest_idx = match_idxs.squeeze().tolist()[-1]
        furthest_2d = hull_points_2d[furthest_idx]

        return hull_points_2d, center_2d, furthest_2d, hull.volume, densest_3d
    else:
        return np.zeros(2), np.zeros(2), np.zeros(2), 0, np.zeros(2)

def densest_point(items):
    if len(items) == 1:
        print('here priya')
        return items[0].matrix_world.translation

    points = []
    for item in items:
        point = item.matrix_world.translation
        points.append(point)
    points = np.array(points)

    neigh = NearestNeighbors()
    neigh.fit(points)

    min_dist = float('inf')
    densest_point = None

    for item in items:
        point = item.matrix_world.translation
        dists, match_idxs = neigh.kneighbors([point], len(points), return_distance=True) 
        cumulative_dist = sum(dists.squeeze())
        if cumulative_dist <= min_dist:
            densest_point = point
            min_dist = cumulative_dist

    return densest_point

def delete_objs(objs):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
    bpy.ops.object.delete()

def make_item(location=None, rotation=None):
    np.random.seed()
    if location is None:
        location = np.random.uniform(-1.1,1.1,3)
        location[2] = np.random.uniform(0.25,1.00)
    if rotation is None:
        rotation = np.array([np.random.uniform(-0.4, 0.4),np.random.uniform(-0.4, 0.4),np.random.uniform(0, np.pi)])

    #bpy.ops.mesh.primitive_ico_sphere_add(radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(0.3, 0.3, 0.3))
    #bpy.ops.mesh.primitive_uv_sphere_add(radius=1, enter_editmode=False, align='WORLD', location=location, rotation=rotation, scale=(0.3, 0.3, 0.3))
    bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=location, rotation=rotation, scale=(0.3, 0.3, 0.3))
    item = bpy.context.object
    bpy.ops.rigidbody.object_add()
    #item.rigid_body.collision_shape = 'MESH'
    #item.rigid_body.mass = 0.5
    item.rigid_body.friction = 10
    return item

def make_pile(num_items, deterministic=False, settle_time=30):
    items = []
    bpy.ops.object.select_all(action='DESELECT')

    if deterministic:
        locations, rotations = generate_heldout_pile_state(num_items)

    for i in range(num_items):
        if deterministic:
            loc, rot = locations[i], rotations[i]
        else:
            loc, rot = None, None
        item = make_item(location=loc, rotation=rot)
        item.pass_index = 1
        items.append(item)
    #for item in items:
    #    item.select_set(True)
    #bpy.ops.object.join()
    #items = bpy.context.object
    #items.pass_index = 1
    #items = None

    for step in range(bpy.context.scene.frame_current, bpy.context.scene.frame_current + settle_time):
        bpy.context.scene.frame_set(step)

    return items

def generate_dataset(episodes, pusher, scooper):
    initial_items= 20
    items = reset_sim(pusher, scooper, initial_items)

    initial_area, initial_num_items = get_coverage_pickup_stats(items)
    rewards = []

    for i in range(5):
        render(bpy.context.scene.frame_current-30)
        #if random.random()<0.5:
        if i%2==0:
            take_push_action(pusher, scooper, items)
        else:
            items, _ = take_scoop_action(pusher, scooper, items)

        render(bpy.context.scene.frame_current-30)
        if not len(items):
            break

        area, num_items = get_coverage_pickup_stats(items)
        rewards.append([initial_area-area, initial_num_items - num_items])
        initial_area = area
        initial_num_items = num_items
        
    print(rewards)
    #delete_objs(items)

if __name__ == '__main__':
    pusher, scooper = initialize_sim()
    generate_dataset(10, pusher, scooper)

    #clear_scene()
    #params = {"table_size":10}
    #make_table(params)
    #make_pile(20, deterministic=True)
    #pusher = make_pusher()
    #scooper = make_scooper('assets/scooper.stl')
