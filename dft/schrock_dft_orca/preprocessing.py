from ase.io import read, write
from glob import glob
import sys, os
from pathlib import Path

def main():

    # Open file to edit
    with open('orca.inp','r', encoding='utf-8') as f:
        data = f.readlines()

    # Extract charge and spin
    with open('default.in', 'r') as file:
        control = file.readlines()

    charge = control[0].split('=')[-1].split('\n')[0]
    spin = control[1].split('=')[-1].split('\n')[0]

    
    # Get structure name
    struct = glob('*.xyz')[0]
    
    # Set params
    data[6] = f'* xyzfile {charge} {spin} {struct}\n'

    # Write back to file
    with open('orca.inp', 'w', encoding='utf-8') as file:
        file.writelines(data)

if __name__ == '__main__':
    main()
