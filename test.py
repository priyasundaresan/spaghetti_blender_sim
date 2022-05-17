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
    if os.path.exists("./images"):
        os.system('rm -r ./images')
    os.makedirs('./images')
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
    scene.view_layers["View Layer"].use_pass_object_index = True
    scene.render.tile_x = 16
    scene.render.tile_y = 16


def initialize_renderer():
    scene = bpy.context.scene
    tree = bpy.context.scene.node_tree
    links = tree.links
    render_node = tree.nodes["Render Layers"]
    id_mask_node = tree.nodes.new(type="CompositorNodeIDMask")
    id_mask_node.use_antialiasing = True
    id_mask_node.index = 1.0
    composite = tree.nodes.new(type = "CompositorNodeComposite")
    links.new(render_node.outputs['IndexOB'], id_mask_node.inputs["ID value"])
    links.new(id_mask_node.outputs[0], composite.inputs["Image"])

def render(episode):
    bpy.context.scene.render.filepath = 'masks/%05d.jpg'%episode
    bpy.ops.render.render(write_still=True)

def make_fork(path_to_fork_stl):
    bpy.ops.import_mesh.stl(filepath=path_to_fork_stl, filter_glob="*.stl")
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.rigidbody.object_add()
    fork = bpy.context.object
    fork.rigid_body.mass = 5
    fork.location = (0,0,2)
    fork.collision.thickness_inner = 0.2
    fork.collision.damping = 1.0
    fork.collision.stickiness = 0.5
    fork.rigid_body.kinematic = True
    fork.rigid_body.collision_shape = 'MESH'
    fork.rigid_body.use_deform = True
    fork.rigid_body.friction = 1
    fork.rigid_body.collision_margin = 0.01

    return fork

def make_walls():
    bpy.ops.import_mesh.stl(filepath='assets/walls.stl', filter_glob="*.stl")
    bpy.ops.rigidbody.object_add()
    walls = bpy.context.object
    walls.rigid_body.type = 'PASSIVE'
    walls.rigid_body.friction = 1.2
    bpy.ops.object.modifier_add(type='COLLISION')
    #walls.rigid_body.collision_shape = 'MESH'
    bpy.ops.object.select_all(action='DESELECT')

def make_table(params):
    #bpy.ops.import_mesh.stl(filepath='assets/tray.stl', filter_glob="*.stl")
    bpy.ops.mesh.primitive_plane_add(size=params["table_size"], location=(0,0,0))
    bpy.ops.rigidbody.object_add()
    table = bpy.context.object
    table.rigid_body.type = 'PASSIVE'
    table.rigid_body.friction = 1.2
    bpy.ops.object.modifier_add(type='COLLISION')
    #table.rigid_body.collision_shape = 'MESH'
    bpy.ops.object.select_all(action='DESELECT')

def make_pusher():
    location = (0,0,5)
    rotation = (0,np.pi/2,0)
    bpy.ops.mesh.primitive_plane_add(size=1, location=location, rotation=rotation)
    #bpy.ops.mesh.primitive_cube_add(size=1, location=location, rotation=rotation)
    pusher = bpy.context.object
    #pusher.scale = (1,1,0.1)
    #bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.modifier_add(type='COLLISION')
    bpy.ops.rigidbody.object_add()
    pusher.rigid_body.kinematic = True
    pusher.rigid_body.type = 'PASSIVE'

    pusher.collision.thickness_outer = 0.2
    pusher.collision.damping = 0.1
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
    noodles.modifiers["Softbody"].settings.mass = 1
    noodles.modifiers["Softbody"].settings.ball_stiff = 1
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

def annotate(points, render_size=(400,400)):
    pixels = []
    for pt in points:
        camera_coord = bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, bpy.context.scene.camera, Vector(pt))
        pixel = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
        pixels.append(pixel)
    return pixels

def make_noodle():
    location = np.random.uniform(-0.3,0.3,3)
    location[2] = np.random.uniform(0.25,1.00)
    rotation = np.array([np.random.uniform(-0.02, 0.02),np.random.uniform(-0.02, 0.02),np.random.uniform(0, np.pi)])
    bpy.ops.curve.primitive_nurbs_path_add(radius=1.0, enter_editmode=False, align='WORLD', location=location, rotation=rotation, scale=(1,1,1))
    bpy.ops.object.editmode_toggle()
    bpy.ops.curve.subdivide(number_cuts=3) 
    bpy.ops.object.editmode_toggle()
    path = bpy.context.object

    bpy.ops.object.modifier_add(type='SOFT_BODY')
    #path.modifiers["Softbody"].settings.mass = 2
    path.modifiers["Softbody"].settings.mass = 1
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
    noodles.pass_index = 1.0

    #empties = []
    #curve_idx = 0
    #while curve_idx < len(noodles.data.splines):
    #    for point_idx, point in enumerate(noodle.data.splines[curve_idx].points):
    #        noodles.data.splines[curve_idx].points[point_idx].select = True
    #        coord = noodles.matrix_world@noodles.data.splines[curve_idx].points[point_idx].co
    #        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=coord[:-1])
    #        empty = bpy.context.object
    #        empty.empty_display_size = 0.15
    #        empties.append(empty)
    #        bpy.context.view_layer.objects.active = noodles
    #        empty.select_set(True)
    #        noodles.select_set(True)
    #        bpy.ops.object.editmode_toggle()
    #        bpy.ops.object.vertex_parent_set()
    #        bpy.ops.object.editmode_toggle()
    #        noodles.data.splines[curve_idx].points[point_idx].select = False
    #        empty.select_set(False)
    #        noodles.select_set(False)
    #    curve_idx += 1
    
    for step in range(0, 30, 1):
        bpy.context.scene.frame_set(step)

    empties = None
    return noodles, empties

def delete_objs(objs):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.delete()

def twirl(fork, start_frame, down_duration, twirl_duration, scoop_duration):
    fork.keyframe_insert(data_path="location", frame=start_frame)
    fork.keyframe_insert(data_path="rotation_euler", frame=start_frame)

    fork.location = (0,0,0.5)
    fork.rotation_euler = (0,0,0)
    fork.keyframe_insert(data_path="location", frame=start_frame+down_duration)
    fork.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration)

    fork.location = (0,0,0.5)
    fork.rotation_euler = (0,0,2*np.pi)
    fork.keyframe_insert(data_path="location", frame=start_frame+down_duration+twirl_duration)
    fork.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration+twirl_duration)

    fork.location = (0,0,1.0)
    fork.rotation_euler = (0,np.pi/2,2*np.pi)
    fork.keyframe_insert(data_path="location", frame=start_frame+down_duration+twirl_duration+scoop_duration)
    fork.keyframe_insert(data_path="rotation_euler", frame=start_frame+down_duration+twirl_duration+scoop_duration)

    for step in range(start_frame, start_frame+down_duration+twirl_duration+scoop_duration):
        bpy.context.scene.frame_set(step)
        

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def push(pusher, start_frame, push_duration, lift_duration, push_start_2d, push_end_2d, hull_2d):
    offset = push_end_2d - push_start_2d
    angle = np.arctan(offset[1]/offset[0])

    push_start_2d -= offset*0.1
    pusher.location = np.array([push_start_2d[0], push_start_2d[1], 0.5])
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame)

    pusher.location = np.array([push_end_2d[0], push_end_2d[1], 0.5])
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame+push_duration)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame+push_duration)

    pusher.location = np.array([push_end_2d[0], push_end_2d[1], 1.0])
    pusher.rotation_euler = (0,np.pi/2,angle)
    pusher.keyframe_insert(data_path="location", frame=start_frame+push_duration+lift_duration)
    pusher.keyframe_insert(data_path="rotation_euler", frame=start_frame+push_duration+lift_duration)

    annot_dir = 'annots'
    for step in range(start_frame, start_frame+push_duration+lift_duration):
        bpy.context.scene.frame_set(step)
        x,y,z = pusher.matrix_world.translation
        pixels = annotate([[x,y,z]] + [[h[0],h[1],0] for h in hull_2d])
        render(step-30)
        np.save('%s/%03d.npy'%(annot_dir,step-30), np.array([pixels]))

    return start_frame+push_duration

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

    print('hull area', hull.volume)
    center_2d = np.mean(points_2d, axis=0)
    print('center', center_2d)
    hull_points_2d = points_2d[hull.vertices]
    neigh = NearestNeighbors()
    neigh.fit(hull_points_2d)
    match_idxs = neigh.kneighbors([center_2d], len(hull.vertices), return_distance=False) 
    furthest_idx = match_idxs.squeeze().tolist()[-1]
    furthest_2d = hull_points_2d[furthest_idx]
    print('furthest_2d', furthest_2d)
    
    return hull_points_2d, center_2d, furthest_2d, hull.volume

def generate_dataset(episodes):
    
    if not os.path.exists('annots'):
        os.mkdir('annots')

    render_size = (400,400)
    set_render_settings('CYCLES', render_size)
    clear_scene()
    camera = add_camera_light()
    params = {"table_size":5}
    #make_walls()
    make_table(params)
    make_pusher()
    make_noodle_rig()
    initialize_renderer()

    for episode in range(episodes):
        noodles, empties = make_noodle_pile(10)
        render(episode)
        #delete_objs([noodles])

    pusher = make_pusher()

    areas = []
    i=0
    area = float('inf')
    while area > 6:
        freeze_softbody_physics(noodles)
        hull_2d, center_2d, furthest_2d, area = noodle_state(noodles)
        add_softbody_physics(noodles)
        areas.append(area)
        start = push(pusher, 30+(i*15), 10, 5, furthest_2d, center_2d, hull_2d)
        i+=1

    print('areas', areas)

if __name__ == '__main__':
    generate_dataset(1)
    #clear_scene()
    #add_camera_light()
    #params = {"table_size":5}
    #make_table(params)
    #make_noodle_rig()
    ##noodle = make_noodle()
    #noodles = make_noodle_pile(10)
