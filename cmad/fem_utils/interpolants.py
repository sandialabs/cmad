import numpy as np

class ShapeFunctions():
    """

    Shape functions and shape function gradients (in the parametric space).

    Attributes:
        values: Values of the shape functions at a discrete set of points.
            Shape is ``(num_eval_points, num_nodes_elem)``, where ``num_eval_points`` is the number of
            points at which the shape functions are evaluated, and ``num_nodes_elem``
            is the number of nodes in the element (which is equal to the
            number of shape functions).
        gradients: Values of the parametric gradients of the shape functions.
            Shape is ``(num_eval_points, dof_node, num_nodes_elem)``, where ``dof_node`` is the number
            of spatial dimensions. Line elements are an exception, which
            have shape ``(num_eval_points, num_nodes_elem)``.

    """
    def __init__(self, values, gradients):
        self.values = values
        self.gradients = gradients

def shape1d(evaluationPoints):
    """

    Evaluate 1D shape functions and derivatives at points in the master element.

    Args:
      evaluationPoints: Array of points in the master element domain at
        which to evaluate the shape functions and derivatives.

    Returns:
      Shape function values and shape function derivatives at ``evaluationPoints``,
      in a tuple (``shape``, ``dshape``).
      shapes: [num_eval_points, num_nodes_elem]
      dshapes: [num_eval_points, num_nodes_elem]

    """

    shape = np.vstack(((1 - evaluationPoints[:,0])/2.0, (1 + evaluationPoints[:,0])/2.0)).T
    dshape = np.vstack((-0.5 * np.ones(len(evaluationPoints)), 0.5 * np.ones(len(evaluationPoints)))).T

    return ShapeFunctions(shape, dshape)

def shape_triangle(evaluationPoints):
    """

    Evaluate triangle shape functions and derivatives at points in the master element.

    Args:
      evaluationPoints: Array of points in the master element domain at
        which to evaluate the shape functions and derivatives.

    Returns:
      Shape function values and shape function derivatives at ``evaluationPoints``,
      in a tuple (``shape``, ``dshape``).
      shapes: [num_eval_points, num_nodes_elem]
      dshapes: [num_eval_points, dof_node, num_nodes_elem]

    """
    num_eval_points = len(evaluationPoints)
    dof_node = 2
    num_nodes_elem = 3

    shape = np.vstack((1 - evaluationPoints[:,0] - evaluationPoints[:,1],
                       evaluationPoints[:,0], evaluationPoints[:,1])).T
    
    dshape = np.zeros((num_eval_points, dof_node, num_nodes_elem))

    for i in range(num_eval_points):
        dshape[i, :, :] = np.array([[-1, 1, 0],[-1, 0, 1]])
    
    return ShapeFunctions(shape, dshape)

def shape_quad(evaluationPoints):
    """

    Evaluate quad shape functions and derivatives at points in the master element.

    Args:
      evaluationPoints: Array of points in the master element domain at
        which to evaluate the shape functions and derivatives.

    Returns:
      Shape function values and shape function derivatives at ``evaluationPoints``,
      in a tuple (``shape``, ``dshape``).
      shapes: [num_eval_points, num_nodes_elem]
      dshapes: [num_eval_points, dof_node, num_nodes_elem]

    """

    num_eval_points = len(evaluationPoints)
    dof_node = 2
    num_nodes_elem = 4

    l0x = 1 - evaluationPoints[:,0]
    l1x = 1 + evaluationPoints[:,0]
    l0y = 1 - evaluationPoints[:,1]
    l1y = 1 + evaluationPoints[:,1]

    shape = np.vstack((l0x * l0y / 4, l1x * l0y / 4, l1x * l1y / 4, l0x * l1y / 4)).T
    dshape = np.zeros((num_eval_points, dof_node, num_nodes_elem))

    for i in range(num_eval_points):
        point = evaluationPoints[i]
        l0x = 1 - point[0]
        l1x = 1 + point[0]
        l0y = 1 - point[1]
        l1y = 1 + point[1]
        dshape[i, :, :] = np.array([[-l0y, l0y, l1y, -l1y],[-l0x, -l1x, l1x, l0x]])
    
    return ShapeFunctions(shape, dshape)

def shape_tetrahedron(evaluationPoints):
    """

    Evaluate tetrahedron shape functions and derivatives at points in the master element.

    Args:
      evaluationPoints: Array of points in the master element domain at
        which to evaluate the shape functions and derivatives.

    Returns:
      Shape function values and shape function derivatives at ``evaluationPoints``,
      in a tuple (``shape``, ``dshape``).
      shapes: [num_eval_points, num_nodes_elem]
      dshapes: [num_eval_points, dof_node, num_nodes_elem]

    """

    num_eval_points = len(evaluationPoints)
    dof_node = 3
    num_nodes_elem = 4

    shape = np.vstack((1 - evaluationPoints[:,0] - evaluationPoints[:,1] - evaluationPoints[:,2],
                       evaluationPoints[:,0], evaluationPoints[:,1], evaluationPoints[:,2])).T
    
    dshape = np.zeros((num_eval_points, dof_node, num_nodes_elem))

    for i in range(num_eval_points):
        dshape[i, :, :] = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    
    return ShapeFunctions(shape, dshape)


def shape_brick(evaluationPoints):
    """

    Evaluate brick shape functions and derivatives at points in the master element.

    Args:
      evaluationPoints: Array of points in the master element domain at
        which to evaluate the shape functions and derivatives.

    Returns:
      Shape function values and shape function derivatives at ``evaluationPoints``,
      in a tuple (``shape``, ``dshape``).
      shapes: [num_eval_points, num_nodes_elem]
      dshapes: [num_eval_points, dof_node, num_nodes_elem]
      
    """

    num_eval_points = len(evaluationPoints)
    dof_node = 3
    num_nodes_elem = 8

    m1 = 1 - evaluationPoints[:,0]
    p1 = 1 + evaluationPoints[:,0]
    m2 = 1 - evaluationPoints[:,1]
    p2 = 1 + evaluationPoints[:,1]
    m3 = 1 - evaluationPoints[:,2]
    p3 = 1 + evaluationPoints[:,3]

    shape = np.vstack((m1 * m2 * m3 / 8, p1 * m2 * m3 / 8, p1 * p2 * m3 / 8, m1 * p2 * m3 / 8,
                       m1 * m2 * p3 / 8, p1 * m2 * p3 / 8, p1 * p2 * p3 / 8, m1 * p2 * p3 / 8)).T

    dshape = np.zeros((num_eval_points, dof_node, num_nodes_elem))

    for i in range(num_eval_points):
        point = evaluationPoints[i]
        m1 = 1 - point[0]
        p1 = 1 + point[0]
        m2 = 1 - point[1]
        p2 = 1 + point[1]
        m3 = 1 - point[2]
        p3 = 1 + point[2]
        dshape[i, :, :] = np.array([[-m2 * m3 / 8, m2 * m3 / 8, p2 * m3 / 8, -p2 * m3 / 8, -m2 * p3 / 8, m2 * p3 / 8, p2 * p3 / 8, -p2 * p3 / 8],
                                   [-m1 * m3 / 8, -p1 * m3 / 8, p1 * m3 / 8, m1 * m3 / 8, -m1 * p3 / 8, -p1 * p3 / 8, p1 * p3 / 8, m1 * p3 / 8],
                                   [-m1 * m2 / 8, -p1 * m2 / 8, -p1 * p2 / 8, -m1 * p2 / 8, m1 * m2 / 8, p1 * m2 / 8, p1 * p2 / 8, m1 * p2 / 8]])
    
    return ShapeFunctions(shape, dshape)


