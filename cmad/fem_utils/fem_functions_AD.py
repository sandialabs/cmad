import jax.numpy as jnp

def compute_shape_jacobian(elem_points, dshape_tetra):

    J = (dshape_tetra @ elem_points).T

    dv = jnp.linalg.det(J)

    # derivatives of shape functions with respect to spacial coordinates
    gradphiXYZ = jnp.linalg.inv(J).T @ dshape_tetra

    return dv, gradphiXYZ

def interpolate(u, shape_tetra, gradphiXYZ, num_nodes_elem):
    
    ux = u[0:num_nodes_elem]
    uy = u[num_nodes_elem:num_nodes_elem * 2]
    uz = u[num_nodes_elem * 2:num_nodes_elem * 3]

    u = jnp.array([jnp.dot(ux, shape_tetra),
                     jnp.dot(uy, shape_tetra),
                     jnp.dot(uz, shape_tetra)])
    
    grad_u = jnp.vstack([gradphiXYZ @ ux,
                           gradphiXYZ @ uy,
                           gradphiXYZ @ uz])

    return u, grad_u

def compute_stress(grad_u, params):

    E = params[0]
    nu = params[1]

    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))

    strain = 1 / 2 * (grad_u + grad_u.T)

    stress = lam * jnp.trace(strain) * jnp.eye(3) + 2 * mu * strain

    return stress

def elem_residual(u, params, elem_points, num_nodes_elem,
        dof_node, gauss_weights_3D, shape_tetra, dshape_tetra):

    residual = jnp.zeros((num_nodes_elem, dof_node))

    for gaussPt3D in range(len(gauss_weights_3D)):
        w_q = gauss_weights_3D[gaussPt3D]

        dshape_tetra_q = dshape_tetra[gaussPt3D, :, :]
        shape_tetra_q = shape_tetra[gaussPt3D, :]

        dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_tetra_q)
        u_q, grad_u_q = interpolate(u, shape_tetra_q, gradphiXYZ_q, num_nodes_elem)

        stress = compute_stress(grad_u_q, params)

        residual +=  w_q * gradphiXYZ_q.T @ stress * dv_q
    
    return residual.reshape(-1, order='F')
