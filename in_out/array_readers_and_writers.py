import logging
logger = logging.getLogger(__name__)

import os.path

import numpy as np


def write_2D_array(array, output_dir, name, fmt='%f'):
    """
    Assuming 2-dim array here e.g. control points
    save_name = os.path.join(Settings().output_dir, name)
    np.savetxt(save_name, array)
    """
    save_name = os.path.join(output_dir, name)
    if len(array.shape) == 0:
        array = array.reshape(1,)
    np.savetxt(save_name, array, fmt=fmt)


def write_3D_array(array, output_dir, name):
    """
    Saving an array has dim (numsubjects, numcps, dimension), using deformetrica format
    """
    s = array.shape
    if len(s) == 2:
        array = np.array([array])
    save_name = os.path.join(output_dir, name)
    with open(save_name, "w") as f:
        f.write(str(len(array)) + " " + str(len(array[0])) + " " + str(len(array[0, 0])) + "\n")
        for elt in array:
            f.write("\n")
            for elt1 in elt:
                for elt2 in elt1:
                    f.write(str(elt2) + " ")
                f.write("\n")


def read_2D_list(path):
    """
    Reading a list of list.
    """
    with open(path, "r") as f:
        output_list = [[float(x) for x in line.split()] for line in f]
    return output_list


def write_2D_list(input_list, output_dir, name):
    """
    Saving a list of list.
    """
    save_name = os.path.join(output_dir, name)
    with open(save_name, "w") as f:
        for elt_i in input_list:
            for elt_i_j in elt_i:
                f.write(str(elt_i_j) + " ")
            f.write("\n")

def read_3D_list(path):
    """
    Reading a list of list of list.
    """
    with open(path, "r") as f:
        output_list = []
        subject_list = []
        for line in f:
            if not line == '\n':
                subject_list.append([float(x) for x in line.split()])
            else:
                output_list.append(subject_list)
                subject_list = []
        if not line == '\n':
            output_list.append(subject_list)
        return output_list

def write_3D_list(list, output_dir, name):
    """
    Saving a list of list of list.
    """
    save_name = os.path.join(output_dir, name)
    with open(save_name, "w") as f:
        for elt_i in list:
            for elt_i_j in elt_i:
                for elt_i_j_k in elt_i_j:
                    f.write(str(elt_i_j_k) + " ")
                if len(elt_i_j) > 1: f.write("\n")
            f.write("\n\n")

def flatten_3D_list(list3):
    out = []
    for list2 in list3:
        for list1 in list2:
            for elt in list1:
                out.append(elt)
    return out


def read_3D_array(name):
    """
    Loads a file containing momenta, old deformetrica syntax assumed
    """
    try:
        with open(name, "r") as f:
            lines = f.readlines()
            line0 = [int(elt) for elt in lines[0].split()]
            nbSubjects, nbControlPoints, dimension = line0[0], line0[1], line0[2]
            momenta = np.zeros((nbSubjects, nbControlPoints, dimension))
            lines = lines[2:]
            for i in range(nbSubjects):
                for c in range(nbControlPoints):
                    foo = lines[c].split()
                    assert (len(foo) == dimension)
                    foo = [float(elt) for elt in foo]
                    momenta[i, c, :] = foo
                lines = lines[1 + nbControlPoints:]
        if momenta.shape[0] == 1:
            return momenta[0]
        else:
            return momenta

    except ValueError:
        return read_2D_array(name)


def read_2D_array(name):
    """
    Assuming 2-dim array here e.g. control points
    """
    return np.loadtxt(name)
