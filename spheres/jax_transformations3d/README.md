# jax_transformations3d
JAX compatible 3d transformations. Create and convert between 4x4 transformation matrices, quaternions and euler angles. Both the API and underlying code are heavily based on Christoph Gohlke's transformation package transformations.py, though there is some additional functionality and not all functions in transformations.py are implemented (yet). Documentation is currently lacking, please see tests.py for extensive examples. 

jax_transformations3d is under development, expect sharp edges and changes that break the current API.

Suggestions, bugs, feature requests, and contributions are welcome!




Implementation of original Functionality:

|function | basic implementation | jit-tested | grad-tested | vmap-tested |
| ------- |:--------------------:|:----------:|:-----------:|:-----------:|
`identity_matrix()` | X | X | | |
`translation_matrix(direction)` | X | X |
`translation_from_matrix(matrix)` | X | X |
`reflection_matrix(point, normal)` | X | X |
`reflection_from_matrix(matrix)` | X | |
`rotation_matrix(angle, direction, point=None)` | X | X |
`rotation_from_matrix(matrix)` | X | |
`scale_matrix(factor, origin=None, direction=None)` | X | X |
`scale_from_matrix(matrix)` | X | |
`projection_matrix(point, normal, direction=None, perspective=None, pseudo=False)` | X | X |
`projection_from_matrix(matrix, pseudo=False)` | X | |
`clip_matrix(left, right, bottom, top, near, far, perspective=False)` | | |
`shear_matrix(angle, direction, point, normal)` | X | |
`shear_from_matrix(matrix)` | X | |
`decompose_matrix(matrix)` | | |
`compose_matrix(scale=None, shear=None, angles=None, translate=None, perspective=None)` | | |
`orthogonalization_matrix(lengths, angles)` | | |
`affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True)` | | |
`superimposition_matrix(v0, v1, scale=False, usesvd=True)` | | |
`euler_matrix(ai, aj, ak, axes='sxyz')` | X | X |
`euler_from_matrix(matrix, axes='sxyz')` | X | X |
`euler_from_quaternion(quaternion, axes='sxyz')` | X | X |
`quaternion_from_euler(ai, aj, ak, axes='sxyz')` | X | X |
`quaternion_about_axis(angle, axis)` | X | X |
`quaternion_matrix(quaternion)` | X | X |
`quaternion_from_matrix(matrix, isprecise=False)` | X | X |
`quaternion_multiply(quaternion1, quaternion0)` | X | X |
`quaternion_conjugate(quaternion)` | X | X |
`quaternion_inverse(quaternion)` | X | X |
`quaternion_real(quaternion)` | X | X |
`quaternion_imag(quaternion)` | X | X |
`quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True)` | | |
`random_quaternion(rand=None, key=None)` | X | X |
`random_rotation_matrix(rand=None, key=None)` | X | X |
`class Arcball` and related methods | | |
`vector_norm(data, axis=None, out=None)` | X* | X |
`unit_vector(data, axis=None, out=None)` | X* | X |
`random_vector(size, key=None)` | full onp dependence | |
`vector_product(v0, v1, axis=0)` | X | X |
`angle_between_vectors(v0, v1, directed=True, axis=0)` | X | |
`inverse_matrix(matrix)` | X | |
`concatenate_matrices(\*matrices)` | X | |
`is_same_transform(matrix0, matrix1)` | X** | |
`is_same_quaternion(q0, q1)` | X** | |

\*Only implemented for `out=None`

\*\*Need to add tests


New Functionality:

|function | basic implementation | jit-tested | grad-tested | vmap-tested |
| ------- |:--------------------:|:----------:|:-----------:|:-----------:|
`apply_matrix(M, v)` | X | X | | |
`apply_quaternion(q, v)` | X | X | | |
