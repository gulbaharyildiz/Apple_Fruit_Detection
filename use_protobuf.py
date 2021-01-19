# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 21:51:47 2021

@author: aad
"""

import os
import sys
args = sys.argv
directory = args[1]
protoc_path = args[2]
for file in os.listdir(directory):
    if file.endswith(".proto"):
        os.system(protoc_path+" "+directory+"/"+file+" --python_out=.")