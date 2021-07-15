"""Grover algorithm in Braket - See Amazon Examples"""
import string
from braket.aws import AwsQuantumTask, AwsDevice
from braket.devices import LocalSimulator
import matplotlib.pyplot as plt
from libs.grover import Grover

grover = Grover(7)
# Load up with all letters of the alphabet
grover.load_database(list(string.ascii_uppercase))
# grover.set_simulation()
grover.device = LocalSimulator()
grover.grover(grover.encode_string('P'))
grover.get_result()
