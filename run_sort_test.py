#!/usr/bin/env python3

if __name__ == '__main__':
    from pyvirtualdisplay import Display
    import os
    Display().start()
    #os.system('blender -P env_pusher.py')
    os.system('blender -P render_sort.py')
