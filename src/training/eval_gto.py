from typing import Set
import numpy
import cupy
import warnings
from .cudft_utils import cugto as cugto
from pyscf.gto.moleintor import make_loc
from pyscf import gto 

def generate_basis_set(mol: gto.Mole):
    """Generate a cuexc.BasisSet object for the given molecule.

    Args:
        mol (gto.Mole): Input mol

    Returns:
        cuexc.BasisSet
    """
    shells = []
    for i in range(mol.nbas):
        loc = mol.bas_coord(i)
        angular = mol.bas_angular(i)
        exponent = mol.bas_exp(i)
        contraction = mol.bas_nctr(i)
        coefficient = mol._libcint_ctr_coeff(i).reshape(-1, contraction)
        tc = coefficient.copy()
        if angular == 0:
            tc *= 0.282094791773878143
        if angular == 1:
            tc *= 0.488602511902919921
        for c in range(contraction):
            shells.append(cugto.Shell(loc, angular, exponent, tc[:, c]))
    basis = cugto.BasisSet(shells, mol.basis, mol.cart)
    # basis_ptr = cugto.get_basis_set_ptr(basis)
    return basis

# from .utils import get_caos_with, get_paos

BLKSIZE = 104 # needs to be the same to lib/gto/grid_ao_drv.c

               
def eval_gto(mol, eval_name, coords, comp=None, non0tab=None, out=None, aux_matrix=None, libexc=None):
    r'''Evaluate AO function value on the given grids,

    Args:
        eval_name : str

            ====================  ======  =======================
            Function              comp    Expression
            ====================  ======  =======================
            "GTOval_sph"          1       |AO>
            "GTOval_ip_sph"       3       nabla |AO>
            "GTOval_ig_sph"       3       (#C(0 1) g) |AO>
            "GTOval_ipig_sph"     9       (#C(0 1) nabla g) |AO>
            "GTOval_cart"         1       |AO>
            "GTOval_ip_cart"      3       nabla |AO>
            "GTOval_ig_cart"      3       (#C(0 1) g)|AO>
            "GTOval_sph_deriv1"   4       GTO value and 1st order GTO values
            "GTOval_sph_deriv2"   10      All derivatives up to 2nd order
            "GTOval_sph_deriv3"   20      All derivatives up to 3rd order
            "GTOval_sph_deriv4"   35      All derivatives up to 4th order
            "GTOval_sp_spinor"    1       sigma dot p |AO> (spinor basis)
            "GTOval_ipsp_spinor"  3       nabla sigma dot p |AO> (spinor basis)
            ====================  ======  =======================

        atm : int32 ndarray
            libcint integral function argument
        bas : int32 ndarray
            libcint integral function argument
        env : float64 ndarray
            libcint integral function argument
        
        ** Following attributes are arguments used by ``libcint`` library **

        _atm :
            :code:`[[charge, ptr-of-coord, nuc-model, ptr-zeta, 0, 0], [...]]`
            each element reperesents one atom
        natm :
            number of atoms
        _bas :
            :code:`[[atom-id, angular-momentum, num-primitive-GTO, num-contracted-GTO, 0, ptr-of-exps, ptr-of-contract-coeff, 0], [...]]`
            each element reperesents one shell
        nbas :
            number of shells
        _env :
            list of floats to store the coordinates, GTO exponents, contract-coefficients

        coords : 2D array, shape (N,3)
            The coordinates of the grids.

    Kwargs:
        comp : int
            Number of the components of the operator
        shls_slice : 2-element list
            (shl_start, shl_end).
            If given, only part of AOs (shl_start <= shell_id < shl_end) are
            evaluated.  By default, all shells defined in mol will be evaluated.
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`dft.gen_grid.make_mask`
        out : ndarray
            If provided, results are written into this array.

    Returns:
        2D array of shape (N,nao) Or 3D array of shape (\*,N,nao) to store AO
        values on grids.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> coords = numpy.random.random((100,3))  # 100 random points
    >>> ao_value = mol.eval_gto("GTOval_sph", coords)
    >>> print(ao_value.shape)
    (100, 24)
    >>> ao_value = mol.eval_gto("GTOval_ig_sph", coords)
    >>> print(ao_value.shape)
    (3, 100, 24)
    '''

    # fast_gto_func = libexc.evaluate_grid_ao
    # lib_grid_ao = numpy.ctypeslib.load_library('grid_ao', "src/pyscf_binding/cudft/lib/grid_ao.so")
    # fast_gto_func = lib_grid_ao.evaluate_grid_ao
    # int ngrids, int natoms, double *co, double *r, double *A, double *angular, double *a, double *result
    # fast_gto_func.argtypes = [ctypes.c_void_p, POINTER(c_double), c_int, c_int, POINTER(c_double)]
    
    eval_name, comp = _get_intor_and_comp(mol, eval_name, comp)
    
    coords = cupy.asarray(coords, dtype=numpy.double, order='C')
    n_grids = len(coords)
    
    if aux_matrix is not None and aux_matrix['ao'].shape[1]== n_grids:
        ao = aux_matrix['ao']
        ao.fill(0)
        # print('LOGGER ao shape:{}'.format(ao.shape))
    elif 'spinor' in eval_name:
        ao = cupy.zeros(shape=(2, comp, n_grids, mol.nao),dtype=numpy.double, order='C')
    else:
        ao = cupy.zeros(shape=(comp, n_grids, mol.nao),dtype=numpy.double, order='C')
        # print('LOGGER LAST ao shape:{}'.format(ao.shape))
    
    cugto.evaluate_grid_ao(generate_basis_set(mol), coords.data.ptr, n_grids, comp, ao.data.ptr)
    ao = ao.reshape(comp,-1).reshape(comp,mol.nao,-1).transpose(0,2,1)
    if comp == 1:
        if 'spinor' in eval_name:
            ao = ao[:,0]
        else:
            ao = ao[0]
    
    return ao

def _get_intor_and_comp(mol, eval_name, comp=None):
    if not ('_sph' in eval_name or '_cart' in eval_name or
            '_spinor' in eval_name):
        if mol.cart:
            eval_name = eval_name + '_cart'
        else:
            eval_name = eval_name + '_sph'

    if comp is None:
        if '_spinor' in eval_name:
            fname = eval_name.replace('_spinor', '')
            comp = _GTO_EVAL_FUNCTIONS.get(fname, (None,None))[1]
        else:
            fname = eval_name.replace('_sph', '').replace('_cart', '')
            comp = _GTO_EVAL_FUNCTIONS.get(fname, (None,None))[0]
        if comp is None:
            warnings.warn('Function %s not found.  Set its comp to 1' % eval_name)
            comp = 1
    return eval_name, comp

_GTO_EVAL_FUNCTIONS = {
#   Functiona name          : (comp-for-scalar, comp-for-spinor)
    'GTOval'                : (1, 1 ),
    'GTOval_ip'             : (3, 3 ),
    'GTOval_ig'             : (3, 3 ),
    'GTOval_ipig'           : (9, 9 ),
    'GTOval_deriv0'         : (1, 1 ),
    'GTOval_deriv1'         : (4, 4 ),
    'GTOval_deriv2'         : (10,10),
    'GTOval_deriv3'         : (20,20),
    'GTOval_deriv4'         : (35,35),
    'GTOval_sp'             : (4, 1 ),
    'GTOval_ipsp'           : (12,3 ),
}


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    coords = numpy.random.random((100,3))
    ao_value = eval_gto(mol, "GTOval_sph", coords)
    print(ao_value.shape)
