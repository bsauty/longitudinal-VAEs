import numpy as np

import logging
logger = logging.getLogger(__name__)


class DeformableMultiObject:
    """
    Collection of deformable objects, i.e. landmarks or images.
    The DeformableMultiObject class is used to deal with collections of deformable objects embedded in
    the current 2D or 3D space. It extends for such collections the methods of the deformable object class
    that are designed for a single object at a time.

    """

    ####################################################################################################################
    ### Constructors:
    ####################################################################################################################

    def __init__(self, object_list):
        self.object_list = object_list

        self.dimension = max([elt.dimension for elt in self.object_list])
        self.number_of_objects = len(self.object_list)  # TODO remove

        self.number_of_objects = len(self.object_list)
        assert (self.number_of_objects > 0)
        self.update_bounding_box(self.dimension)

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    # Accessor
    def __getitem__(self, item):
        return self.object_list[item]

    def get_data(self):
        landmark_points = []
        image_intensities = None
        for elt in self.object_list:
            if elt.type.lower() in ['surfacemesh', 'polyline', 'pointcloud', 'landmark']:
                landmark_points.append(elt.get_points())
            elif elt.type.lower() == 'image':
                assert image_intensities is None, 'A deformable_multi_object cannot contain more than one image object.'
                image_intensities = elt.get_intensities()

        data = {}
        if len(landmark_points) > 0:
            data = {'landmark_points': np.array(np.concatenate(landmark_points))}
        if image_intensities is not None:
            data['image_intensities'] = image_intensities
        return data

    def set_data(self, data):
        if 'landmark_points' in data.keys():
            landmark_object_list = [elt for elt in self.object_list
                                    if elt.type.lower() in ['surfacemesh', 'polyline', 'pointcloud', 'landmark']]
            assert len(data['landmark_points']) == np.sum([elt.get_number_of_points()
                                                           for elt in landmark_object_list]), \
                "Number of points differ in template and data given to template"
            pos = 0
            for i, elt in enumerate(landmark_object_list):
                elt.set_points(data['landmark_points'][pos:pos + elt.get_number_of_points()])
                pos += elt.get_number_of_points()

        if 'image_intensities' in data.keys():
            image_object_list = [elt for elt in self.object_list if elt.type.lower() == 'image']
            assert len(image_object_list) == 1, 'That\'s unexpected.'
            image_object_list[0].set_intensities(data['image_intensities'])

    def get_points(self):
        """
        Gets the geometrical data that defines the deformable multi object, as a concatenated array.
        We suppose that object data share the same last dimension (e.g. the list of list of points of vtks).
        """
        landmark_points = []
        image_points = None
        for elt in self.object_list:
            if elt.type.lower() in ['surfacemesh', 'polyline', 'pointcloud', 'landmark']:
                landmark_points.append(elt.get_points())
            elif elt.type.lower() == 'image':
                assert image_points is None, 'A deformable_multi_object cannot contain more than one image object.'
                image_points = elt.get_points()

        points = {}
        if len(landmark_points) > 0:
            # points['landmark_points'] = landmark_points
            points = {'landmark_points': np.concatenate(landmark_points)}
        if image_points is not None:
            points['image_points'] = image_points
        return points

    def get_deformed_data(self, deformed_points, template_data):

        deformed_data = {}

        if 'landmark_points' in deformed_points.keys():
            # assert 'landmark_points' in template_data.keys(), 'That\'s unexpected.'       # TODO check this
            # template_data['landmark_points'] = deformed_points['landmark_points']  # For torch gradients to circulate.
            deformed_data['landmark_points'] = deformed_points['landmark_points']

        if 'image_points' in deformed_points.keys():
            assert 'image_intensities' in template_data.keys(), 'That\'s unexpected.'
            image_object_list = [elt for elt in self.object_list if elt.type.lower() == 'image']
            assert len(image_object_list) == 1, 'That\'s unexpected.'
            deformed_data['image_intensities'] = image_object_list[0].get_deformed_intensities(
                deformed_points['image_points'], template_data['image_intensities'])

        return deformed_data

    # def set_points(self, points):
    #     """
    #     points is a numpy array containing the new position of all the landmark points
    #     """
    #     if 'landmark_points' in data.keys():
    #         landmark_object_list = [elt for elt in self.object_list
    #                                 if elt.type.lower() in ['surfacemesh', 'polyline', 'pointcloud', 'landmark']]
    #         assert len(data) == np.sum([elt.get_number_of_points() for elt in landmark_object_list]), \
    #             "Number of points differ in template and data given to template"
    #         pos = 0
    #         for i, elt in enumerate(landmark_object_list):
    #             elt.set_points(data[pos:pos + elt.get_number_of_points()])
    #             pos += elt.get_number_of_points()
    #
    #     if 'image_intensities' in data.keys():
    #         image_object = [elt for elt in self.object_list if elt.type.lower() == 'image'][0]
    #         image_object.set_intensities(data['image_intensities'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    # # Update the relevant information.
    # def update(self):
    #     self.number_of_objects = len(self.object_list)
    #     assert (self.number_of_objects > 0)
    #
    #     for elt in self.object_list:
    #         elt.update()
    #
    #     self.update_bounding_box(self.dimension)

    # Compute a tight bounding box that contains all objects.
    def update_bounding_box(self, dimension):
        assert (self.number_of_objects > 0)

        self.bounding_box = self.object_list[0].bounding_box
        for k in range(1, self.number_of_objects):
            for d in range(dimension):
                if self.object_list[k].bounding_box[d, 0] < self.bounding_box[d, 0]:
                    self.bounding_box[d, 0] = self.object_list[k].bounding_box[d, 0]
                if self.object_list[k].bounding_box[d, 1] > self.bounding_box[d, 1]:
                    self.bounding_box[d, 1] = self.object_list[k].bounding_box[d, 1]

    def write(self, output_dir, names, data=None):
        """
        Save the list of objects with the given names
        """
        assert len(names) == len(self.object_list), "Give as many names as objects to save multi-object"

        pos = 0
        for elt, name in zip(self.object_list, names):
            if data is None:
                elt.write(output_dir, name)

            else:
                if elt.type.lower() in ['surfacemesh', 'polyline', 'pointcloud', 'landmark']:
                    elt.write(output_dir, name, data['landmark_points'][pos:pos + elt.get_number_of_points()])
                    pos += elt.get_number_of_points()

                elif elt.type.lower() == 'image':
                    elt.write(output_dir, name, data['image_intensities'])
