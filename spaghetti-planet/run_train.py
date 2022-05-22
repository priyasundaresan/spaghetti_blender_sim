#!/usr/bin/env python3

'''
Solution for headless rendering is to "fake" a display:
https://blender.stackexchange.com/questions/150526/blender-2-8-aws-ec2-command-line-eevee-render
'''

if __name__ == '__main__':
    from pyvirtualdisplay import Display
    import os
    Display().start()
    os.system('blender -P main.py')
