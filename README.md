[ChemRxiv](linkb)  |  [Paper](linke)


# GB-GA
[Graph-based genetic algorithm](http://dx.doi.org/10.1039/C8SC05372C)
 
Repository for the paper: *Genetic algorithm for nitrogen fixation.*

Welcome to DeepStruc, a Deep Generative Model (DGM) that learns the relation between PDF and atomic structure and 
thereby solves a structure from a PDF!

1. [GA](#ga)
2. [Run](#run)


## Run

```
python GA_schrock.py
```

## Parameters
A list of possible arguments.
 
| Arg                  | Description                                                                                          |  
|----------------------|------------------------------------------------------------------------------------------------------|  
| `-h` or `--help`     | Prints help message.                                                                                 |
| `--population_size`  | Sets the size of the population pool.                                                                |
| `--mating_pool_size` | Sets the size of the mating pool.                                                                    |
| `--n_confs`          | Sets how many conformers to generate for each molecule                                               |
| `--n_tries`          | Sets how many times the GA is restarted. Can be used to run multiple GA runs in a single submission. |
| `--cpus_per_task`    | How many cores to use for each scoring job                                                           |

# Authors
__Magnus Strandgaard__<sup>1</sup>
 
<sup>1</sup> Department of Chemistryr, University of Copenhagen, 2100 Copenhagen Ø, Denmark.
Heres my email: _mastr@chem.ku.dk__.



