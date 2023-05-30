from qiskit import *
from qiskit.circuit.library import *
import math
from math import pi
from qiskit_ionq import *
from qiskit.transpiler.passes import BasisTranslator
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#######################################################3
#Not implemented the exact formula. Not so useful in our case
def compute_number_of_shots(success_probability, confidence_interval=1.96): #1.96=95%, 2.58=99%
    return  1

####### Grover
def get_diffuser(n_var):
    qvar = QuantumRegister(n_var)
    qtarget = QuantumRegister(1)
    diffuser = QuantumCircuit(qvar, qtarget, name='diffuser')
    diffuser.h(qvar)
    diffuser.x(qvar)
    diffuser.append(MCXGate(len(qvar)), qvar[0:]+qtarget[0:])
    diffuser.x(qvar)
    diffuser.h(qvar)
    return diffuser.to_gate()


def get_oracle(n_var):
    oq_n = QuantumRegister(n_var, 'x')
    oq_checker = QuantumRegister(1, 'checker')
    oracle = QuantumCircuit(oq_n, oq_checker, name='oracle')
        
    oracle.x(oq_n[1:])
    oracle.append(MCXGate(len(oq_n)), oq_n[0:]+oq_checker[0:])
    oracle.x(oq_n[1:])
    
    return oracle

def get_grover(n):
    q_n = QuantumRegister(n, 'x')
    q_checker = QuantumRegister(1, 'checker')

    sat_solver = QuantumCircuit(q_n, q_checker)


    sat_solver.h(q_n)
    sat_solver.x(q_checker)

    iters = int(math.ceil(math.pi/4*math.sqrt(2**n)))
    for i in range(iters):
        sat_solver.append(get_oracle(len(q_n)), q_n[0:]+q_checker[0:])
        sat_solver.append(get_diffuser(len(q_n)), q_n[0:]+q_checker[0:])

    success_probability = math.sin( (2*iters +1) * math.acos(math.sqrt( (2**n-1)/2**n)) )**2
    # When the number of qubits is greater then 4, the number of shots becomes 1
    # for simplicity we can assume that the success_probability is 1
    shots = compute_number_of_shots(success_probability)
    return sat_solver, shots

####### Inverse Fourier Transform
def Inverse_QFT(n):
    q = QuantumRegister(n, 'q')
    qft_circuit = QuantumCircuit(q, name='Inverse_QFT '+str(n))
    
    for i in range(math.floor(len(q)/2)):
        qft_circuit.append(SwapGate(), [q[i], q[len(q)-i-1]])


    for i in range(len(q)):
        for j in range(i):
            controlled_phase_gate = CPhaseGate(-pi/(2**(i-j)))
            qft_circuit.append(controlled_phase_gate, [q[j], q[i]])
        qft_circuit.append(HGate(), [q[i]])
    
    success_probability = 1 # We can assume that it is 1
    shots = compute_number_of_shots(success_probability)
    return qft_circuit, shots

###############################################################################


###############################################################################
''' Notes '''
# Add Meaning of Quantum Adv in slides
# Mention photonic architecture - it works at room temperature
# Show circuit pre and post decomposition to target basis
# Mention T2 and T1 (T2 <= T1)
# The energy cost is mostly due to cooling 
    # image of dilution refrigerator 
    # Formula for computing the energy cost of dilution refrigerators (https://cds.cern.ch/record/1974048/files/arXiv:1501.07392.pdf)
        # Not woriking in our case

# The number of qubits is "essentially" independet from the power https://www.rand.org/content/dam/rand/pubs/working_papers/WRA2400/WRA2427-1/RAND_WRA2427-1.pdf

# Useful for future experiments: https://quantumcomputing.stackexchange.com/questions/17860/transpilation-into-custom-gate-set-in-qiskit
# https://arxiv.org/pdf/2203.17181.pdf

#error correction, more qubit


'''
They found that for one problem
instance, Summit consumed an average of 8.65 MW of power over 2.44 hours, for a total energy
consumption of 21.1 MWh, while Sycamore consumed an average of 15 kW over 1 minute and
41 seconds, for a total energy consumption of 420 Wh. Sycamore therefore beat Summit’s
performance by a factor of 578 for power consumption, 87 for runtime, and a factor of 578  ́ 87
» 50200 for energy consumption. 
'''
###############################################################################


gate_sizes = {
    'x':1,
    'rz':1,
    'rx':1,
    'ry':1,
    'sx':1,
    'cx':2,
    'cz':2,
    'cp':2,
    'rxx':2,
    'ryy':2
}

ibm = { #ibm_algiers
    'native_gates' : ['cx', 'id', 'rz', 'sx', 'x'],
    'hardware' : 'superconducting_ibm',
    '1_qubit_gate_speed':  80, #ns
    '2_qubit_gate_speed':  320, #ns
    'T1' :  150000, #ns
    'T2' :  108000, #ns
    'temperature': 10, #mK https://iopscience.iop.org/article/10.1088/2058-9565/acae3e/pdf
    'kwh': 25 #https://aisel.aisnet.org/cgi/viewcontent.cgi?article=1361&context=isd2014
} #ibmq_manila 5qubit

#https://qcs.rigetti.com/qpus
rigetti = {  #rigetti aspen-m-3 70 qubit 
    'native_gates' :  ['cz', 'cp', 'rxx', 'ryy', 'rx', 'rz'],
    'hardware' : 'superconducting_rigetti',
    '1_qubit_gate_speed':  40, #ns
    '2_qubit_gate_speed':  240, #ns
    'T1' :  22000, #ns
    'T2' :  24000, #ns
    'temperature': 10, #mK
    'kwh': 25 #https://aisel.aisnet.org/cgi/viewcontent.cgi?article=1361&context=isd2014
}

#target_basis_ion = ['ms', 'gpi', 'gpi2', 'rx','rz', 'cp'] #
ionq = { #ionq
    'native_gates' : ['rx', 'ry', 'rxx', 'id'],
    'hardware' : 'ion_trapped_ionq',
    '1_qubit_gate_speed':  135000, #ns
    '2_qubit_gate_speed':  600000, #ns
    'T1' :  10000000000, #ns
    'T2' :  1000000000, #ns
    'temperature': 4000, #mk A 100 W refrigerator for liquid helium temperatures (4 K, or –269°C) consumes at least 7 kW https://cds.cern.ch/record/1974048/files/arXiv:1501.07392.pdf
    'kwh': 0.0000002778, ##https://iopscience.iop.org/article/10.1088/2058-9565/acae3e/pdf
} 


#https://arxiv.org/pdf/2304.14360.pdf (below 100mK)
neutral_atoms = { #neutral atoms
    'native_gates' : ['rx', 'ry', 'rz', 'cz'],
    'hardware' : 'neutral_atoms',
    '1_qubit_gate_speed':  60, #ns
    '2_qubit_gate_speed':  120, #ns https://arxiv.org/pdf/2206.08915.pdf
    'T1' :  10000000000, #ns
    'T2' :  4000000000, #ns https://physicsworld.com/a/new-neutral-atom-qubit-offers-advantages-for-quantum-computing/
    'temperature': 100, #mk
    'kwh': 0 #
}

'''
    Decompose a circuit into the set of gate (target_basis) of the target architecture
    return:
        - depth of the transpiled circuit
        - dict of gates 
        - number of 1-qubit gate
        - number of 2-qubit gate
'''
def circuit_gates(circuit, target_basis, gate_sizes):
    circuit = transpile(circuit,
                       basis_gates=target_basis, 
                       optimization_level=3)
    dict_gates = dict(circuit.count_ops())
    n_gates = 0
    n_one_qubit_gates = 0
    n_two_qubits_gates = 0
    for key, value in dict_gates.items():
        if gate_sizes[key] == 1:
            n_one_qubit_gates+=value
        if gate_sizes[key] == 2:
            n_two_qubits_gates+=value
        n_gates = n_gates + value
    return circuit.depth(), dict_gates, n_gates, n_one_qubit_gates, n_two_qubits_gates



'''
    Compute the kWh for each architecture according to the formulas defined io:
'''
def computekWh(arch, depth, n_one_qubit_gates, n_two_qubits_gates, shots):
    if arch['hardware'] == 'neutral_atoms':
        return ((n_one_qubit_gates + n_two_qubits_gates)*shots)/1000 * 15 * 3600 #kwh
    if arch['hardware'] == 'ion_trapped_ionq':
        return shots*(arch['1_qubit_gate_speed']*n_one_qubit_gates + arch['2_qubit_gate_speed']*n_two_qubits_gates)*arch['kwh']
    else: #superconducting
        ratio_n_one_qubit_gates = n_one_qubit_gates/(n_one_qubit_gates+n_two_qubits_gates)
        ratio_n_two_qubits_gates = n_two_qubits_gates/(n_one_qubit_gates+n_two_qubits_gates)
        circuit_time_ns = shots*(arch['1_qubit_gate_speed']*depth*ratio_n_one_qubit_gates + arch['2_qubit_gate_speed']*depth*ratio_n_two_qubits_gates) 
        return arch['kwh'] * circuit_time_ns * 2.78**(-13) 


'''
    Compute an approximation of the execution time in nanosecond
'''
def compute_circuit_time_approximation(arch, depth, n_one_qubit_gates, n_two_qubits_gates):
    ratio_n_one_qubit_gates = n_one_qubit_gates/(n_one_qubit_gates+n_two_qubits_gates)
    ratio_n_two_qubits_gates = n_two_qubits_gates/(n_one_qubit_gates+n_two_qubits_gates)
    circuit_time_ns = arch['1_qubit_gate_speed']*depth*ratio_n_one_qubit_gates + arch['2_qubit_gate_speed']*depth*ratio_n_two_qubits_gates
    return circuit_time_ns

'''
'''


def plot_execute_time(df, algorithm):
    df_filtered = df[(df['algorithm'] == algorithm)]
    df_filtered['execution_time'] = df_filtered['execution_time'].apply(lambda x: x*10**(-3)) #to microsecond
    plot = sns.lineplot(data=df_filtered, x='n_qubits', y='execution_time', hue='hardware')
    plot.axhline(ibm['T2']*10**(-3), color='b', linestyle='--', label='T2 IBM')
    plot.axhline(rigetti['T2']*10**(-3), color='orange', linestyle='--', label='T2 Rigetti')
    plot.axhline(ionq['T2']*10**(-3), color='g', linestyle='--', label='T2 IonQ')
    plot.axhline(neutral_atoms['T2']*10**(-3), color='r', linestyle='--', label='T2 Neutral Atoms')
    plt.legend()
    plt.yscale('log')
    plt.show()
    plt.clf()

def plot_kwh(df, algorithm):
    df_filtered = df[(df['algorithm'] == algorithm)]
    plot = sns.lineplot(data=df_filtered, x='n_qubits', y='kwh', hue='hardware')
    plt.legend()
    plt.yscale('log')
    plt.show()
    plt.clf()

def compare_n_gates(df, algorithm):
    df_filtered = df[(df['algorithm'] == algorithm)]
    sns.barplot(data=df_filtered, x='n_qubits', y='Total_Ops', hue='type', hue_order=['Abstract_gates', 'Native_gates', 'Classical_complexity'])
    plt.yscale('log')
    plt.legend()
    plt.show()
    plt.clf()




#########################################################################

architectures = [ibm, rigetti, ionq, neutral_atoms]

algorithms = [get_grover, Inverse_QFT]

qubits_list = [3,4,5,6,7,8,9,10]

dataframe_file = './GreenQC.csv'

'''
df = pd.DataFrame()
for alg in algorithms:
    print('-----------')
    print("Algorithm "+alg.__name__)
    for arch in architectures:
        print("Architecture "+arch['hardware'])
        for n_qubits in qubits_list:
            print("N_qubits: "+str(n_qubits))
            circuit, shots = alg(n_qubits)
            depth, dict_gates, n_gates, n_one_qubit_gates, n_two_qubits_gates = circuit_gates(circuit, arch['native_gates'], gate_sizes)
            print('Depth: '+str(depth))
            print('N_gates: '+str(n_gates))
            print('N_1_qubit_gates: '+str(n_one_qubit_gates))
            print('N_2_qubit_gates: '+str(n_two_qubits_gates))
            print('-----------')
            new_row = {
                       'algorithm':alg.__name__,
                       'n_qubits': n_qubits,
                       'depth': depth,
                       '1_qubit_gates': n_one_qubit_gates,
                       '2_qubit_gates': n_two_qubits_gates,
                       'hardware': arch['hardware'],
                       'kwh': computekWh(arch, depth, n_one_qubit_gates, n_two_qubits_gates, shots),
                       'execution_time': compute_circuit_time_approximation(arch, depth, n_one_qubit_gates, n_two_qubits_gates)
                       }
            df = df.append(new_row, ignore_index=True)
df.to_csv(dataframe_file, index=False)
'''

sns.set_palette('colorblind')
print('Loading Dataframe')
df = pd.read_csv(dataframe_file)
plot_execute_time(df, 'get_grover')
plot_execute_time(df, 'Inverse_QFT')
plot_kwh(df, 'get_grover')
plot_kwh(df, 'Inverse_QFT')



'''
df_new = pd.DataFrame()
for i in range(len(df)):
    new_row_native_gates = df.loc[i].copy()
    new_row_native_gates['type'] = 'Native_gates'
    new_row_abstract_gates = df.loc[i].copy()
    new_row_abstract_gates['type'] = 'Abstract_gates'
    new_row_classical = df.loc[i].copy()
    new_row_classical['type'] = 'Classical_complexity'
    df_new = df_new.append(new_row_native_gates, ignore_index=True)
    df_new = df_new.append(new_row_abstract_gates, ignore_index=True)
    df_new = df_new.append(new_row_classical, ignore_index=True)



df_new.loc[(df_new['algorithm'] == 'Inverse_QFT') & (df_new['type'] == 'Native_gates'), 'Total_Ops'] = df_new['1_qubit_gates'] + df_new['2_qubit_gates']
df_new.loc[(df_new['algorithm'] == 'get_grover') & (df_new['type'] == 'Native_gates'), 'Total_Ops'] = df_new['1_qubit_gates'] + df_new['2_qubit_gates']

#quantum fourier O(nlogn)
df_new.loc[(df_new['algorithm'] == 'Inverse_QFT') & (df_new['type'] == 'Abstract_gates'), 'Total_Ops'] = df_new['n_qubits'] * np.log2(df_new['n_qubits'])
df_new.loc[(df_new['algorithm'] == 'get_grover') & (df_new['type'] == 'Abstract_gates'), 'Total_Ops'] = np.sqrt(2**df_new['n_qubits']) 

#classical fourier O(n2^n)
df_new.loc[(df_new['algorithm'] == 'Inverse_QFT') & (df_new['type'] == 'Classical_complexity'), 'Total_Ops'] = df_new['n_qubits'] * 2**(df_new['n_qubits'])
df_new.loc[(df_new['algorithm'] == 'get_grover') & (df_new['type'] == 'Classical_complexity'), 'Total_Ops'] = 2**df_new['n_qubits']

compare_n_gates(df_new, 'Inverse_QFT')
compare_n_gates(df_new, 'get_grover')
'''

