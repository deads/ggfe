#!/usr/bin/python

# Grammar-Guided Feature Extraction (GGFE)
#
# Author:   Damian Eads
#
# File:     bioid.py
#
# Purpose:  This module parses data stored in the BioID format.
#
# Date:     August 4, 2010

import sys
import types

def get_label_dictionary():
    """
    Returns a dictionary mapping human-understable names like eyes or mouth
    to a list of indexes.

    0: right eye pupil
    1: left eye pupil
    2: right mouth corner
    3: left mouth corner
    4: outer end of right eye brow
    5: inner end of right eye brow
    6: inner end of left eye brow
    7: outer end of left eye brow
    8: right temple
    9: outer corner of right eye
    10: inner corner of right eye
    11: inner corner of left eye
    12: outer corner of left eye
    13: left temple
    14: tip of nose
    15: right nostril
    16: left nostril
    17: centre point on outer edge of upper lip
    18: centre point on outer edge of lower lip
    19: tip of chin
    """
    labels = {"reye": [0],
              "leye": [1],
              "eyes": ["reye","leye"],
              "rmouth": [2],
              "lmouth": [3],
              "mouth": ["rmouth","lmouth"],
              "orbrow": [4],
              "irbrow": [5],
              "rbrow": ["orbrow","irbrow"],
              "olbrow": [6],
              "olbrow": [7],
              "lbrow": ["olbrow","ilbrow"],
              "rtemple": [8],
              "ocreye": [9],
              "icreye": [10],
              "icleye": [11],
              "ocleye": [12],
              "eyecorners": ["ocreye", "icreye", "icleye", "ocleye"],
              "ltemple": [13],
              "nosetip": [14],
              "rnostril": [15],
              "lnostril": [16],
              "nostrils": ["lnostril","rnostril"],
              "nose": ["nostrils","nosetips"],
              "ulip": [17],
              "llip": [18],
              "chintip": [19],
              "chin": [19]}
    return labels

def get_points(filename):
    """
    Grabs a list of points stored in the bioid format.
    """
    fid = open(filename, 'r')
    lines = fid.readlines()

    versionline=lines[0]

    parts = versionline.split(":")
    version = [part.strip() for part in parts]
    version = version[1]

    if version != "1":
        raise RuntimeError("unsupport version")
        exit(1)

    npointsline=lines[1]

    parts = npointsline.split(":")
    npoints = [part.strip() for part in parts]
    npoints = int(npoints[1])

    for k in xrange(2, len(lines)):
        if "{" in lines[k]:
            break

    pts = []

    for j in xrange(k+1, len(lines)):
        if "}" in lines[j]:
            break
        line = lines[j]
        parts = line.split(" ")
        parts = [float(part) for part in parts]
        pts.append(parts)

    if len(pts) != npoints:
        raise ValueError("Number of points in list not equal to the number specified in npoints")
    return pts

def get_indices(keyword):
    """
    Returns a list of centroid indices corresponding to a particular facial
    feature (e.g. 'eyes', 'mouth', 'nose')
    """
    labels = get_label_dictionary()
    if type(keyword) == types.IntType:
        return keyword
    elif type(keyword) == types.StringType:
        if keyword == "":
            indices = range(0, 20)
        else:
            if keyword in labels:
                rindices = [get_indices(kw) for kw in labels[keyword]]
                indices = []
                for index in rindices:
                    if type(index) == types.ListType:
                        indices = indices + index
                    else:
                        indices.append(index)
            else:
                raise ValueError("convert_points.py: unknown keyword '%s'" % keyword)
        return indices


def filter_points_by_index(pts, indices):
    """
    Given a full list of points containing all the facial points marked
    in the BioID data set, this function only selects the points corresponding
    to particular indices.
    """
    newpts = []
    for k in indices:
        newpts.append(pts[k])
    return newpts

def get_points_by_keyword(filename, keyword):
    """
    Returns all the centroids of all those points matching a keyword.
    """
    pts = get_points(filename)
    indices = get_indices(keyword)
    newpts = filter_points_by_index(pts, indices)
    return newpts

if __name__ == "__main__":
    keyword = ""
    if len(sys.argv) == 1:
        print "usage: convert_points.py filename [keyword]"
        sys.exit(1);
        
    if len(sys.argv) >= 2:
        filename = sys.argv[1]

    if len(sys.argv) >= 3:
        keyword = sys.argv[2]

    try:
        print get_points_by_keyword(filename, keyword)
    except RuntimeError:
        print "error (convert_points.py): " + e
