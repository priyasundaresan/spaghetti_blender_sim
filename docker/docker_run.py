#!/usr/bin/env python
import os

if __name__=="__main__":
    #cmd = "docker run -it -v %s:/tmp priya-blender" % (os.getcwd())
    #cmd = "docker run  -v %s:/tmp priya-blender" % (os.getcwd())
    cmd = "nvidia-docker run -it -v %s:/host priya-blender" % (os.path.join(os.getcwd(), '..'))
    code = os.system(cmd)
