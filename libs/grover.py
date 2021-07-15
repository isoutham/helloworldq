"""Implement Grover algorithm"""
import binascii
from braket.circuits import circuit, Circuit, Gate, Moments
from braket.circuits.instruction import Instruction
from braket.aws import AwsQuantumTask, AwsDevice
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Grover:

    def __init__(self, qubits):
        self.qubits = qubits
        self.reps = 1
        self.simulation = None
        self.circuit = None
        self.__device = None
        self.item = None
        self.database = []

    def hadamard(self):
        """
        function to apply hadamard to all qubits
        """
        circ = Circuit()
        circ.h(np.arange(self.qubits))
        return circ

    def oracle(self, item):
        """
        function to apply oracle for given target item
        """
        circ = Circuit()
        circ.add_circuit(self.simulation[item])
        return circ

    def amplify(self):
        """
        function for amplitude amplification
        This is the starting state (see grover())
        """
        circ = Circuit()

        # Amplification
        circ.h(np.arange(self.qubits))
        circ.add_circuit(self.simulation['0' * self.qubits])
        circ.h(np.arange(self.qubits))
        return circ

    def grover(self, item):
        """
        function to put together individual modules of Grover algorithm
        """
        self.item = item
        grover_circ = self.hadamard()
        for i in range(self.reps):
            or_circ = self.oracle(item)
            grover_circ.add(or_circ)
            amplification = self.amplify()
            grover_circ.add(amplification)

        self.circuit = grover_circ

    @circuit.subroutine(register=True)
    def ccz(qubits, targets):
        """
        implementation an x-qubit Controlled-Controlled-Z Gate
        Braket does not have a built-in CCZ gate so this is the best I could
        do for now.  Need to get my head around the maths because, This
        gets massive very quickly!

        See https://threeplusone.com/pubs/on_gates.pdf
        """
        dims = 2**qubits
        gate = np.zeros(shape=(dims, dims), dtype=complex)
        pos = 0
        for row in gate:
            if pos == dims - 1:
                row[pos] = -1
            else:
                row[pos] = 1
            pos += 1
        circuit = Circuit()
        circuit.unitary(matrix=gate, targets=targets)
        return circuit

    @staticmethod
    def encode_string(string):
        """bin to ascii"""
        return ''.join(format(ord(x), 'b') for x in string)

    @staticmethod
    def decode_string(string):
        """Ascii to bin"""
        digit1 = string[0:7]
        digit2 = string[7:14]
        ret = ''
        for digit in [digit1, digit2]:
            # Coded to allow 14 bits but needs this for 7
            # not elegant but works ;)
            if digit == '':
                continue
            number = int(digit, 2)
            char = number.to_bytes((number.bit_length() + 7) // 8, 'big').decode()
            ret += char
        return ret

    def load_database(self, strings):
        """
        Put some characters/strings in to search for
        Be careful that must not be longer than the number of bits you have available
        """
        all_bits = []
        for i in range(self.qubits):
            all_bits.append(i)
        simulation = {}
        for string in strings:
            truncated = self.encode_string(string)
            zeros = []
            for i in range(len(truncated)):
                if truncated[i] == '0':
                    zeros.append(i)
            simulation[truncated] = Circuit().x(zeros).ccz(self.qubits, targets=all_bits).x(zeros)
            self.database.append(truncated)
        # This is adding the zero bits simulation for the amplification step
        simulation['0' * self.qubits] = Circuit().x(all_bits).ccz(self.qubits, targets=all_bits).x(all_bits)
        self.simulation = simulation

    def set_simulation(self):
        """
        This creates a gate for every possible bit combination in the search range
        This bit is probably not necessary but more experimentation is needed
        """
        all_bits = []
        for i in range(self.qubits):
            all_bits.append(i)
        simulation = {}
        form = f'0{self.qubits}b'
        for i in range(0, 2**self.qubits):
            bitstring = "{val:{form}}".format(val=i, form=form)
            zeros = []
            for i in range(len(bitstring)):
                if bitstring[i] == '0':
                    zeros.append(i)
            simulation[bitstring] = Circuit().x(zeros).ccz(self.qubits, targets=all_bits).x(zeros)
        self.simulation = simulation

    def get_result(self):
        """
        Just a crap routine to gather the results
        Way too long and horrible FIXME
        """
        # num_qubits = self.circuit.qubit_count
        self.circuit.probability()

        task = self.device.run(self.circuit, shots=1000)

        status_list = []
        status = task.state()
        status_list += [status]
        print('Status:', status)

        while status != 'COMPLETED':
            status = task.state()
            if status != status_list[-1]:
                print('Status:', status)
            status_list += [status]

        result = task.result()
        # metadata = result.task_metadata
        probs_values = result.values[0]
        # measurements = result.measurement_counts

        format_bitstring = '{0:0' + str(self.qubits) + 'b}'

        # Get all possible values as, probabilities can fall outside of the
        # of the requested range
        bitstring_keys = [format_bitstring.format(i) for i in range(2**self.qubits)]

        results = dict(zip(bitstring_keys, probs_values))
        odf = pd.Series(results).to_frame('probability')
        # odf = odf[odf["probability"] > 0]
        odf['char'] = odf.index.map(lambda x: self.decode_string(x))
        # odf = odf[(odf["char"] >= 'a') & (odf['char'] <= 'z')]
        print('Solution:')
        print(odf[odf.probability == odf.probability.max()])
        odf = odf[odf.index.isin(self.database)]
        plt.bar(odf['char'], odf['probability'])
        plt.title(f'Search for {self.decode_string(self.item)} ({self.item})')
        plt.xlabel('Letters')
        plt.ylabel('Probability')
        # plt.xticks(rotation=90)
        plt.savefig('grover.png')

    @property
    def device(self):
        """
        Getter for device so we can easily switch between a LocalSimulator
        and the real thing
        """
        return self.__device

    @device.setter
    def device(self, device):
        self.__device = device
