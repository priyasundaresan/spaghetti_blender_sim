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

RED_RECEPTACLE_LOC = (5,0,0)
BLUE_RECEPTACLE_LOC = (-5,0,0)

def add_camera_light():
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0,0,0))
    #bpy.ops.object.camera_add(location=(0,0,8), rotation=(0,0,0))
    #bpy.ops.object.camera_add(location=(0,0,20), rotation=(0,0,0))
    bpy.ops.object.camera_add(location=(0,-5,20), rotation=(np.deg2rad(18.5),0,0))
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
    scene.view_settings.view_transform = 'Raw'
    scene.eevee.taa_samples = 10
    scene.eevee.taa_render_samples = 10
    scene.view_settings.exposure = 1.3
    bpy.ops.rigidbody.world_add()
    bpy.context.scene.rigidbody_world.point_cache.frame_end = 2000
    bpy.context.scene.frame_end = 2000

def initialize_renderer():
    scene = bpy.context.scene
    tree = bpy.context.scene.node_tree
    links = tree.links
    render_node = tree.nodes["Render Layers"]
    vl = tree.nodes.new('CompositorNodeViewer')   
    composite = tree.nodes.new(type = "CompositorNodeComposite")
    links.new(render_node.outputs['Image'], composite.inputs["Image"])
    links.new(render_node.outputs['Image'], vl.inputs[0])  # link Render Image to Viewer Image

def render(episode):
    bpy.context.scene.render.filepath = 'masks/%05d.jpg'%episode
    bpy.ops.render.render(write_still=True)
    #image = cv2.imread('masks/%05d.jpg'%episode)
    pixels = np.array(bpy.data.images['Viewer Node'].pixels)
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    image = pixels.reshape(height,width,4)[:,:,:3]

    #cv2.imshow('img', image)
    #cv2.waitKey(0)
    #cv2.imwrite('masks/%05d.jpg'%episode, image)
    return image

def make_walls(params):
    dim = params["table_size"]

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

    dir_path = os.path.dirname(os.path.realpath(__file__))
    bpy.ops.import_mesh.stl(filepath='%s/assets/receptacle_wide_tall.stl'%dir_path, filter_glob="*.stl")
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.rigidbody.object_add()
    receptacle_right = bpy.context.object
    receptacle_right.rigid_body.collision_shape = 'MESH'
    receptacle_right.rigid_body.type = 'PASSIVE'
    receptacle_right.rigid_body.friction = 1.0

    dir_path = os.path.dirname(os.path.realpath(__file__))
    bpy.ops.import_mesh.stl(filepath='%s/assets/receptacle_wide_tall.stl'%dir_path, filter_glob="*.stl")
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.rigidbody.object_add()
    receptacle_left = bpy.context.object
    receptacle_left.rigid_body.collision_shape = 'MESH'
    receptacle_left.rotation_euler = (0,0,np.pi)
    receptacle_left.rigid_body.type = 'PASSIVE'
    receptacle_left.rigid_body.friction = 1.0
    
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

def push(pusher, down_duration, push_duration, lift_duration, wait_duration, push_start, push_end, annot_dir='annots'):
    start_frame = bpy.context.scene.frame_current
    print('START_FRAME', start_frame)

    push_start_2d = push_start[:2]
    push_end_2d = push_end[:2].astype('float')

    offset = push_end_2d - push_start_2d
    angle = np.arctan(offset[1]/offset[0])

    push_start_2d -= offset*0.25
    push_end_2d -= offset*0.15

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
    np.save('%s/%03d.npy'%(annot_dir,start_frame-30), np.array([pixels]))

    for step in range(start_frame, start_frame+down_duration+push_duration+lift_duration+wait_duration):
        bpy.context.scene.frame_set(step)
        #render(step)
        #np.save('%s/%03d.npy'%(annot_dir,step), np.array([pixels]))
    return pixels

def pick_place(pick_item, transport_time, wait_duration, place_point, annot_dir='annots'):
    start_frame = bpy.context.scene.frame_current
    print('START_FRAME', start_frame)

    pick_2d = pick_item.matrix_world.translation[:2]
    place_2d = place_point[:2]

    pick_item.rigid_body.kinematic = True
    pick_item.keyframe_insert(data_path="rigid_body.kinematic", frame=start_frame)
    pick_item.location = pick_item.matrix_world.translation
    pick_item.keyframe_insert(data_path="location", frame=start_frame)
    pick_item.rotation_euler = pick_item.matrix_world.to_euler()
    pick_item.keyframe_insert(data_path="rotation_euler", frame=start_frame)

    pick_item.location = np.array(pick_item.matrix_world.translation) + np.array([0,0,1])
    pick_item.keyframe_insert(data_path="location", frame=start_frame+transport_time/2)
    pick_item.keyframe_insert(data_path="rotation_euler", frame=start_frame+transport_time/2)

    pick_item.location = np.array(place_point)
    pick_item.keyframe_insert(data_path="location", frame=start_frame+transport_time)
    pick_item.keyframe_insert(data_path="rotation_euler", frame=start_frame+transport_time)

    pixels = annotate([[pick_2d[0], pick_2d[1], 0], [place_2d[0], place_2d[1], 0]])
    np.save('%s/%03d.npy'%(annot_dir,start_frame-30), np.array([pixels]))

    for step in range(start_frame, start_frame + transport_time):
        bpy.context.scene.frame_set(step)
        #render(step)
        #np.save('%s/%03d.npy'%(annot_dir,step), np.array([pixels]))

    pick_item.rigid_body.kinematic = False
    pick_item.keyframe_insert(data_path="rigid_body.kinematic", frame=step)

    for step in range(start_frame+transport_time, start_frame+transport_time+wait_duration):
        bpy.context.scene.frame_set(step)
        #render(step)
        #np.save('%s/%03d.npy'%(annot_dir,step), np.array([pixels]))

    return pixels

def take_pick_place_action(pick_item, place_point):
    settle_time = 10
    transport_time = 30
    wait_duration = 20
    pixels = pick_place(pick_item, transport_time, wait_duration, place_point)
    return pixels

def take_push_action(pusher, push_start, push_end, items):
    down_duration = 30
    lift_duration = 5
    push_duration = 45
    wait_duration = 20
    pixels = push(pusher, down_duration, push_duration, lift_duration, wait_duration, np.array(push_start), np.array(push_end))
    reset_pusher(pusher)
    return pixels

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

def generate_heldout_pile_state(num_items):
    RANDOM_SEED = 3
    locations = []
    rotations = []
    for i in range(num_items):
        np.random.seed(num_items+i+RANDOM_SEED)
        location = np.random.uniform(-3,3,3)
        location[2] = np.random.uniform(0.05,0.1)
        np.random.seed(num_items+i+RANDOM_SEED)
        rotation = np.array([np.random.uniform(-0.4, 0.4),np.random.uniform(-0.4, 0.4),np.random.uniform(0, np.pi)])
        locations.append(location)
        rotations.append(rotation)
    return locations, rotations


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

def initialize_sim():
    if not os.path.exists('masks'):
        os.mkdir('masks')
    if not os.path.exists('annots'):
        os.mkdir('annots')

    render_size = (256,256)
    set_render_settings('BLENDER_EEVEE', render_size)
    clear_scene()
    clear_actions_frames()
    camera = add_camera_light()
    params = {"table_size":6}
    make_table(params)
    make_walls(params)
    initialize_renderer()
    pusher = make_pusher()
    reset_pusher(pusher)

    return pusher

def reset_sim(pusher, num_items, deterministic=False):
    reset_pusher(pusher)
    clear_actions_frames()
    items, colors = make_pile(num_items, deterministic=deterministic, settle_time=30)
    print(bpy.context.scene.frame_current)
    return items, colors

def make_pile(num_items, deterministic=False, settle_time=30):
    items = []
    colors = []
    bpy.ops.object.select_all(action='DESELECT')

    if deterministic:
        locations, rotations = generate_heldout_pile_state(num_items)

    for i in range(num_items):
        if deterministic:
            loc, rot = locations[i], rotations[i]
        else:
            loc, rot = None, None
        item, color = make_item(location=loc, rotation=rot)
        items.append(item)
        colors.append(color)

    for step in range(bpy.context.scene.frame_current, bpy.context.scene.frame_current + settle_time):
        bpy.context.scene.frame_set(step)

    return items, colors

def delete_objs(objs):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
    bpy.ops.object.delete()

def set_viewport_shading(mode):
    '''Makes color/texture viewable in viewport'''
    areas = bpy.context.workspace.screens[0].areas
    for area in areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = mode

def colorize(obj, color):
    r,g,b = np.array(color)
    color = [r,g,b,1]
    if '%.2f-%.2f-%.2f'%(r,g,b) in bpy.data.materials:
        mat = bpy.data.materials['%.2f-%.2f-%.2f'%(r,g,b)]
    else:
        mat = bpy.data.materials.new(name='%.2f-%.2f-%.2f'%(r,g,b))
        mat.use_nodes = False
    mat.diffuse_color = color
    #mat.specular_intensity = np.random.uniform(0, 0.1)
    #mat.roughness = np.random.uniform(0.5, 1)
    mat.roughness = 1
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
    set_viewport_shading('MATERIAL')

def make_item(location=None, rotation=None):
    np.random.seed()
    if location is None:
        location = np.random.uniform(-1.1,1.1,3)
        location[2] = np.random.uniform(0.25,1.00)
    if rotation is None:
        rotation = np.array([np.random.uniform(-0.4, 0.4),np.random.uniform(-0.4, 0.4),np.random.uniform(0, np.pi)])

    bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=location, rotation=rotation, scale=(0.3, 0.3, 0.3))
    item = bpy.context.object
    bpy.ops.rigidbody.object_add()
    #item.rigid_body.collision_shape = 'MESH'
    item.rigid_body.mass = 15
    item.rigid_body.friction = 10

    COLOR_RED= (1,0,0)
    COLOR_BLUE = (0,0,1)

    if random.random() < 0.5:
        COLOR = COLOR_BLUE
    else:
        COLOR = COLOR_RED
    colorize(item, COLOR)

    start_frame = bpy.context.scene.frame_current
    item.keyframe_insert(data_path="rigid_body.kinematic", frame=start_frame)

    return item, 'blue' if COLOR==COLOR_BLUE else 'red'

def closest_point(items, point):
    points = []
    for item in items:
        points.append(item.matrix_world.translation)
    points = np.array(points)

    neigh = NearestNeighbors()
    if len(points) == 1:
        points = points.reshape(1, -1)
    neigh.fit(points)

    dists, match_idxs = neigh.kneighbors([point], 1, return_distance=True) 
    closest_idx = match_idxs.squeeze()
    closest_dist = dists.squeeze()
    closest_point = points[closest_idx]
    closest_item = items[closest_idx]

    return closest_point, closest_dist, closest_item

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

def get_action_candidates(items, colors):

    red_items = [items[i] for i in range(len(items)) if colors[i] == 'red' and items[i].matrix_world.translation[2] >= 0]
    blue_items = [items[i] for i in range(len(items)) if colors[i] == 'blue' and items[i].matrix_world.translation[2] >= 0]

    if len(red_items) + len(blue_items) == 0:
        return None, None, None, None

    # for pick place
    if len(red_items):
        red_closest_point, red_closest_dist, red_closest_item = closest_point(red_items, RED_RECEPTACLE_LOC)

    if len(blue_items):
        blue_closest_point, blue_closest_dist, blue_closest_item = closest_point(blue_items, BLUE_RECEPTACLE_LOC)

    if len(red_items) and len(blue_items):
        if (red_closest_dist < blue_closest_dist):
            pick_item= red_closest_item
            pick_point = red_closest_point
            place_point = RED_RECEPTACLE_LOC
        else:
            pick_item= blue_closest_item
            pick_point = blue_closest_point
            place_point = BLUE_RECEPTACLE_LOC
    else:
        pick_item= red_closest_item if len(red_items) else blue_closest_item
        pick_point = red_closest_point if len(red_items) else blue_closest_point
        place_point = RED_RECEPTACLE_LOC if len(red_items) else BLUE_RECEPTACLE_LOC

    # for push
    if len(red_items) > len(blue_items):
        push_candidate_items = red_items
        push_start = densest_point(push_candidate_items)
        push_end = RED_RECEPTACLE_LOC
    else:
        push_candidate_items = blue_items
        push_start = densest_point(push_candidate_items)
        push_end = BLUE_RECEPTACLE_LOC

    return pick_item, place_point, push_start, push_end

def get_reward_stats(items, colors):

    red_items = [items[i] for i in range(len(items)) if colors[i] == 'red']
    blue_items = [items[i] for i in range(len(items)) if colors[i] == 'blue']

    red_correct = 0
    red_incorrect = 0

    blue_correct = 0
    blue_incorrect = 0

    THRESH = 3.75
    for item, color in zip(items,colors):
        dist_to_red_receptacle = np.linalg.norm(np.array(item.matrix_world.translation) - RED_RECEPTACLE_LOC)
        dist_to_blue_receptacle = np.linalg.norm(np.array(item.matrix_world.translation) - BLUE_RECEPTACLE_LOC)
        print('dist', dist_to_red_receptacle, dist_to_blue_receptacle)
        if dist_to_red_receptacle < THRESH:
            if color == 'red':
                red_correct += 1
            else:
                blue_incorrect += 1
        if dist_to_blue_receptacle < THRESH:
            if color == 'blue':
                blue_correct += 1
            else:
                red_incorrect += 1
    return red_correct, red_incorrect, blue_correct, blue_incorrect

def remove_picked_up(items, colors):
    bpy.ops.object.select_all(action='DESELECT')
    z_lower = 0.0
    remaining_items = []
    remaining_colors = []
    for item, color in zip(items, colors):
        x,y,z = item.matrix_world.translation
        picked_up = (z < z_lower) 
        if picked_up:
            item.select_set(True)
        else:
            remaining_items.append(item)
            remaining_colors.append(color)
    bpy.ops.object.delete()
    return remaining_items, remaining_colors

def generate_dataset(episodes, pusher):
    initial_items= 10
    items, colors = reset_sim(pusher, initial_items)

    rewards = []
    for i in range(5):
        render(bpy.context.scene.frame_current-30)
        pick_item, place_point, push_start, push_end = get_action_candidates(items, colors)

        if pick_item is None:
            break

        if i%2==0:
        #if False:
            take_push_action(pusher, push_start, push_end, items)
        else:
            take_pick_place_action(pick_item, place_point)

        result = get_reward_stats(items, colors)
        rewards.append(result)
    print('REWARDS')

    print('red_correct, red_incorrect, blue_correct, blue_incorrect')
    print(rewards)
        #area, num_items = get_coverage_pickup_stats(items)
        #rewards.append([initial_area-area, initial_num_items - num_items])
        #initial_area = area
        #initial_num_items = num_items
        
    #print(rewards)
    #delete_objs(items)

if __name__ == '__main__':
    pusher = initialize_sim()
    generate_dataset(10, pusher)
