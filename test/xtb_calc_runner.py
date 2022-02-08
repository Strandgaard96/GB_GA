import os
import sys
from pathlib import Path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..")))

from my_utils import auto, my_utils, my_xtb_utils

if __name__ == '__main__':

    path = Path(
        "/home/energy/magstr/nitrogenase/xtb_optimization_pureXTB_6.3.3/bases/schrock/part1")

    spin = True
    struct = ".xyz"
    paths = auto.get_paths_custom(
        source=path,
        struct=struct,
        dest="out")

    print(f'Optimizing at following locations: {paths}')
    for elem in paths:
        print(f'Optimizing {elem}')
        with my_utils.cd(elem[0:len(elem.split('/')[-1])]):

            # Get spin and charge
            with open('default.in', 'r') as f:
                data = f.readlines()

            charge = data[0].split('=')[-1].split('\n')[0]
            spin = data[1].split('=')[-1].split('\n')[0]

            my_xtb_utils.run_xtb(
                structure=elem.split('/')[-1],
                method='gfn2',
                type="ohess",
                charge=charge,
                spin=spin,
                gbsa='Benzene')

            if elem == paths[-1]:
                sys.exit(0)
