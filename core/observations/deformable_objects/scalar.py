# import logging
# logger = logging.getLogger(__name__)
#
# import numpy as np
# import torch
# from torch.autograd import Variable
#
#
# class Scalar:
#     """
#     Light-weight class for scalar observations e.g. cognitive scores.
#     """
#
#     ####################################################################################################################
#     ### Constructor:
#     ####################################################################################################################
#
#     # Constructor.
#     def __init__(self, x):
#         self.type = 'Scalar'
#         self.points = x  # Numpy array.
#         self.points_torch = Variable(torch.from_numpy(x).type(Settings().tensor_scalar_type))
#         self.bounding_box = None
#
#     # Clone.
#     def clone(self):
#         clone = Scalar(np.copy(self.points))
#         return clone
#
#     ####################################################################################################################
#     ### Encapsulation methods:
#     ####################################################################################################################
#
#     def set_points(self, points):
#         """
#         Here points is torch tensor !
#         """
#         self.points = points.data.numpy()
#         self.points_torch = points
#
#     def get_points(self):
#         return self.points
#
#     def get_points_torch(self):
#         return self.points_torch
#
#     ####################################################################################################################
#     ### Public methods:
#     ####################################################################################################################
#
#     # Update the relevant information.
#     def update(self):
#         return
