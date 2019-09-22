from __future__ import generator_stop

from collections import namedtuple
from itertools import cycle

from subprocess import run, PIPE
import numpy as np

def create_var_list(q):
    p = ['p' + variable for variable in q]
    euler_variables = ['phi', 'p_phi', 'theta', 'p_theta', 'psi', 'p_psi']

    _vars = []
    for it in cycle([iter(q), iter(p)]):
        try:
            _vars.append(next(it))
        except StopIteration:
            return _vars + euler_variables
        
q = ['R', 'Theta']
_vars = create_var_list(q)
_dvars = ['d' + variable for variable in _vars]        
 
class PS_point( namedtuple('PS_point', _vars) ):
    __slots__ = ()
    def __str__(self):
        return ' '.join(str(getattr(self, _)) for _ in self._fields)
    
    @classmethod
    def from_str(cls, p_str):
        '''Call as
           d = PS_point.from_str('...')
        '''
        
        p = map(float, p_str.split(' '))
        return cls(*p)
    
class dPS_point( namedtuple('dPS_point', _dvars) ):
    __slots__ = ()
    def __str__(self):
        return ' '.join(str(getattr(self, _)) for _ in self._fields)

def strip_comments( lines ):
    return [line for line in lines if '#' not in line]

PROGRAM_PATH = "/home/artfin/Desktop/repos/generalHamDerivatives/v1.0/trajectory-python-integration/"

def compute_hamiltonian_derivatives(ps_point):
    program = PROGRAM_PATH + "./compute-derivatives"
    p = run( [program], stdout = PIPE, input = str(ps_point), encoding = 'ascii', check = True) 
    return dPS_point(*[float(line) for line in strip_comments(p.stdout.split('\n')[:-1])])

def compute_hamiltonian(ps_point):
    program = PROGRAM_PATH + "./compute-hamiltonian"
    p = run( [program], stdout = PIPE, input = str(ps_point), encoding = 'ascii', check = True )
    return [float(line) for line in strip_comments(p.stdout.split('\n')[:-1])][0]

def compute_trajectory(tmax, sampling_time, ps_point):
    program = PROGRAM_PATH + "./compute-trajectory"
    program_input = str(tmax) + ' ' + str(sampling_time) + '\n' + str(ps_point)
    p = run([program], stdout = PIPE, input = program_input, encoding = 'ascii', check = True) 
    result = [list(map(float, line.split())) for line in p.stdout.split('\n') if line]
    return [r[0] for r in result], [PS_point(*r[1:]) for r in result]

def reverse_phase_point(p):
    x = [getattr(p, field) for field in p._fields]
    for i in range(1, len(x), 2):
        x[i] *= -1
    return PS_point(*x)

def get_attribute(phase_points, attr):
    return [getattr(p, attr) for p in phase_points]

def make_energy_array(time, phase_points, step = 1):
    return np.array(
        (time[::step], [compute_hamiltonian(p) for p in phase_points[::step]]) 
                    ).T

def stack_arrays(time, phase_points, attr):
    return np.vstack((time, get_attribute(phase_points, attr))).T