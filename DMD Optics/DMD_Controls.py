import numpy as np

def x_1(n):
  """
  Returns the x-coord of nth spot in the middle vertical line (counting from the bottom)
  """
  # n = 0 is bottom spot
  return 10*n + 442
def y_1(n):
  """
  Returns the y-coord of nth spot in the middle vertical line (counting from the bottom)
  """
  # n = 0 is bottom spot
  return -10*n + 846

def x_2(n):
  """
  Returns the x-coord of nth spot in the middle horizontal line (counting from the left)
  """
  # n = 0 is left spot
  return round(-11.*n + 547)
def y_2(n):
  """
  Returns the y-coord of nth spot in the middle horizontal line (counting from the left)
  """
  # n = 0 is left spot
  return round(-11.2*n + 852)

def get_individual_spot_locations():
    a = np.zeros((11, 11, 2), dtype=int)
    x_vert = np.array([x_1(n) for n in range(10, -1, -1)]) # start from the top
    y_vert = np.array([y_1(n) for n in range(10, -1, -1)])

    for i in range(11):
        x_center = x_vert[i]
        y_center = y_vert[i]

        x_coords = [11*j + x_center for j in range(5, -6, -1)]
        y_coords = [round(11.2*j) + y_center for j in range(5, -6, -1)]
        # x_coords = [11*j + x_center for j in range(-5, 6)]
        # y_coords = [round(11.2*j) + y_center for j in range(-5, 6)]

        for j, (x_coord,y_coord) in enumerate(zip(x_coords, y_coords)):
            a[i, j] = [x_coord, y_coord]
    return a