import numpy as np

import logging
logger = logging.getLogger(__name__)


class LongitudinalDataset:
    """
    A longitudinal data set is a collection of sets of deformable objects
    for a series of subjects at multiple time-points.

    """

    ################################################################################
    ### Constructor:
    ################################################################################

    def __init__(self, subject_ids, times=None, deformable_objects=None):

        self.subject_ids = subject_ids
        self.times = times
        self.deformable_objects = deformable_objects

        self.number_of_subjects = len(subject_ids)

        # assert self.number_of_subjects == len(self.times)

        # Total number of observations.
        if times is not None:
            self.total_number_of_observations = 0
            for i in range(self.number_of_subjects):
                self.total_number_of_observations += len(self.times[i])
        elif deformable_objects is not None:
            self.total_number_of_observations = 0
            for i in range(self.number_of_subjects):
                self.total_number_of_observations += len(self.deformable_objects[i])

        # Order the observations.
        if times is not None and len(times) > 0 and len(times[0]) > 0 and deformable_objects is not None:
            self.order_observations()

    ################################################################################
    ### Public methods:
    ################################################################################

    def is_cross_sectional(self):
        """
        Checks whether there is a single visit per subject
        """
        b = True
        for elt in self.deformable_objects: b = (b and len(elt) == 1)
        return b

    def is_time_series(self):
        """
        Checks whether there is a single visit per subject
        """
        return len(self.deformable_objects) == 1 and len(self.deformable_objects[0]) > 1 and \
               len(self.times) == 1 and len(self.deformable_objects[0]) == len(self.times[0])

    def check_image_shapes(self):
        """
        In the case of non deformable objects, checks the dimension of the images are the same.
        """
        shape = None
        for subj in self.deformable_objects:
            for img in subj:
                if shape is None:
                    shape = img.shape
                else:
                    assert img.shape == shape, \
                        "Different images dimensions were detected."

    def order_observations(self):
        """ sort the visits for each individual, by time"""
        for i in range(len(self.times)):
            arg_sorted_times = np.argsort(self.times[i])
            sorted_times = np.sort(self.times[i])
            sorted_deformable_objects = []
            for j in arg_sorted_times:
                sorted_deformable_objects.append(self.deformable_objects[i][j])
            self.times[i] = sorted_times
            self.deformable_objects[i] = sorted_deformable_objects
