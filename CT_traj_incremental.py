#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gtsam
import numpy as np
from typing import List

from gtsam import Pose3, StateData
from gtsam.symbol_shorthand import X, V, L, Y, W
import time
import os
import argparse

from collections import defaultdict


# In[2]:


TOL_NS = 1_000_000        # 1 ms timestamp matching tolerance
FRAME_STRIDE = 4          # use every 4th non-KF frame to start
MIN_OBS = 3               # min observations per landmark
MAX_REPROJ_PX = 5.0       # init outlier threshold
STATE_DT = 0.5  # GP state spacing [s]
FRAME_DT = 0.05           # EuRoC camera period (20 Hz)

# EuRoC cam0 intrinsics. ORB-SLAM3's EuRoC yaml resizes images 752x480 -> 600x350
# (Camera.newWidth/newHeight), so the pixels in the dumps live in the resized image.
# Pixels are already undistorted (mvKeysUn), so no distortion model here.
SX, SY = 600 / 752, 350 / 480
K = gtsam.Cal3_S2(458.654 * SX, 457.296 * SY, 0.0, 367.215 * SX, 248.375 * SY)


# In[3]:


def s_to_ns(s):            # "1403636579.963555574"
    a, b = s.split('.')
    return int(a) * 10**9 + int(b.ljust(9, '0')[:9])

def ns_to_s(ns):           # 1403636579963555574 -> "1403636579.963555574"
    ns = int(ns)
    sign = '-' if ns < 0 else ''
    a, b = divmod(abs(ns), 10**9)
    return f"{sign}{a}.{b:09d}"


# In[40]:


def load_f_dataset(path):
    """Parse a TUM file with lines: t_ns tx ty tz qx qy qz qw -> (int64 ns array, [Pose3])."""
    times_ns, poses = [], []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok or tok[0].startswith('#'):
                continue
            times_ns.append(int(tok[0].split('.')[0]))
            tx, ty, tz, qx, qy, qz, qw = map(float, tok[1:8])
            poses.append(Pose3(gtsam.Rot3.Quaternion(qw, qx, qy, qz), gtsam.Point3(tx, ty, tz)))
    return np.array(times_ns, dtype=np.int64), poses

# ---------- load landmarks + keyframe observations ----------
def build_landmark_lookup_dict(path):
    """ Return dict of Landmark_id and its 3D world Pose"""
    P = {}
    for line in open(path):
        s = line.split()
        if s[0] == "P":
            P[int(s[1])] = np.array(list(map(float, s[2:5])))
    return P

def load_per_frame_observations(path):
    frame_obs = {}
    with open(path) as f:
        for line in f:
            s = line.split()
            temp = [int(s[0]), float(s[2]), float(s[3])]
            t_ns = s_to_ns(s[1])
            if(t_ns in frame_obs.keys()):
                frame_obs[t_ns].append(temp)
            else:
                frame_obs[t_ns] = [temp]
    return frame_obs


# ## 1. Load and filter (unchanged)

# In[51]:


# kf_times_ns, kf_poses = load_kf_dataset("MH_01_data/kf_dataset-MH01_mono.txt")

# Loading all frames with timestamps and  
f_times_ns, f_poses = load_f_dataset("MH_01_data/f_dataset-MH01_mono.txt")
P = build_landmark_lookup_dict("MH_01_data/MapPoints_KF_UVobs_dataset-MH01_mono.txt")

# load frame observations
per_frame_obs = load_per_frame_observations("MH_01_data/Frame_obs_MH01_mono.txt")

# Other objects
landmarks_in_graph = []
pending = []
last_GP_state_t = 0


# In[49]:


params = gtsam.ISAM2Params(); params.setRelinearizeThreshold(0.1); params.relinearizeSkip = 1
isam = gtsam.ISAM2(params)


# In[50]:


for i in range(len(f_times_ns)):
    print

