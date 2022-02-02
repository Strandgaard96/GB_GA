# %%
import sys

sys.path.append("/home/julius/soft/GB-GA")

from catalyst.utils import mols_from_smi_file

# %%
with open("/home/julius/soft/GB-GA/ZINC_1000_amines.smi", "r") as f:
    o = None
    counter = 0
    for i, line in enumerate(f):
        if i % 50 == 0:
            if o:
                o.close()
            o = open(
                f"/home/julius/soft/GB-GA/GB-GA_catalyst/chunks/pop{counter:02d}.smi",
                "w",
            )
            counter += 1
        o.write(line)
    if o:
        o.close()
