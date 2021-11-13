from polygons import Polygons

class Energy(Polygons):

    def __init__(self, polygon_coords):
        self.polygon_coords = polygon_coords
    def scopes_of_polygons(self):
        return list(map(self.polygon_scope, self.polygon_coords))
    def area_of_polygons(self):
        return list(map(self.polygon_area, self.polygon_coords))
    
    # TODO: energie terme aufstellen

    def __call__(self):
        energy = sum(self.area_of_polygons())
        return energy
