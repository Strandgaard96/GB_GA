# Import module containing commonly used building blocks
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_utils import auto, my_utils, my_xtb_utils

if __name__ == "__main__":

    path = Path(
        "/home/magstr/Documents/nitrogenase/schrock/bases/schrock_cycle_noHIPT/"
    )
    # Local path: /home/magstr/Documents/nitrogenase/schrock/cycle
    # Niflheim: /home/energy/magstr/nitrogenase/xtb_optimization_pureXTB_6.3.3/bases/schrock/part1

    struct = ".mol"
    paths = auto.get_paths_custom(source=path, struct=struct, dest="out")

    print(f"Optimizing at following locations: {paths}")
    for elem in paths:
        print(f"Processing {elem}")
        with my_utils.cd(elem.parent):

            # Get spin and charge
            with open("default.in", "r") as f:
                data = f.readlines()

            charge = data[0].split("=")[-1].split("\n")[0]
            spin = data[1].split("=")[-1].split("\n")[0]

            file = elem.name.replace(".xyz", ".mol")
            # Cut 1 HIPT group and convert to mol object
            fragname = my_xtb_utils.create_intermediates(
                file=elem.name, charge=int(charge)
            )

            # Run the xtb calculation on the cut molecule
            my_xtb_utils.run_xtb(
                structure=fragname,
                method="gfn2",
                type="ohess",
                charge=charge,
                spin=spin,
                gbsa="Benzene",
            )

            if elem == paths[-1]:
                sys.exit(0)
