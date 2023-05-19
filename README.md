[ChemRxiv](linkb)  |  [Paper](linke)


# GB-GA
[Graph-based genetic algorithm](http://dx.doi.org/10.1039/C8SC05372C)

Repository for the paper: *Genetic algorithm-based re-optimization of the Schrock catalyst for dinitrogen fixation*

1. [GA](#gb-ga)
2. [Run](#how-to-run)
3. [Parameters](#parameters)


## How to run

For simple use of the GA install with conda

    conda env create --file environment.yml

To run the ga activate the relevant environment and run:
```
python GA_schrock.py
```

## Parameters
A list of possible arguments.

| Arg                  | Description                                                                                                          |
|----------------------|----------------------------------------------------------------------------------------------------------------------|
| `-h` or `--help`     | Prints help message.                                                                                                 |
| `--population_size`  | Sets the size of the population pool.                                                                                |
| `--mating_pool_size` | Sets the size of the mating pool.                                                                                    |
| `--n_confs`          | Sets how many conformers to generate for each molecule.                                                              |
| `--n_tries`          | Sets how many times the GA is restarted. Can be used to run multiple GA runs in a single submission.                 |
| `--cpus_per_task`    | How many cores to use for each scoring job.                                                                          |
| `--RMS_thresh`       | RMS cutoff value for RDKit conformer embedding.                                                                      |
| `--generations`      | How many evolution cycles of the population is performed.                                                            |
| `--mutation_rate`    | Decides the probability of performing a mutation operation instead of crossover.                                     |
| `--prune_population` | If there are duplicates within the current population these are removed.                                             |
| `--sa_screening`     | Decides if synthetic accessibility score is enabled. Highly recommended to turn this on.                             |
| `--file_name`        | Path to the database extract to create starting population.                                                          |
| `--output_dir`       | Sets output directory for all files generated during generations.                                                    |                                                                                   |
| `--timeout`          | How many minutes each slurm job is allowed to run                                                                    |
| `--debug`            | If set the starting population is a set of 4 small molecules that can run fast locally. Used for debugging.          |
| `--ga_scoring`       | If set, removes all higher energy conformers in GA.                                                                  |
| `--supress_amines`   | Supress amine heavy molecules by converting any primary amines to hydrogen in generations.                           |
| `--method`           | Which gfn method to use.                                                                                             |
| `--energy_cutoff`    | Sets energy cutoff on the conformer filtering.                                                                       |
| `--bond_opt`         | Decides if a final Mo-N bond optimization is performed during scoring.                                               |
| `--cleanup`          | If enabled, all scoring files are removed after scoring. Only the optimized structures and their energies are saved. |
| `--scoring_func`     | Which scoring function to use.                                                                                       |
| `--opt`              | Set optimization convergence criteria for xTB.                                                                       |
| `--gbsa`             | Which type of solvent to use for xTB.                                                                                |
| `--input`            | Name of input control file created for xTB                                                                           |
| `--average_size`     | Average number of atoms in molecules resulting from crossover.                                                       |
| `--size-stdev`       | STD of crossover molecule size distribution                                                                          |


# Authors
__Magnus Strandgaard__<sup>1</sup>

<sup>1</sup> Department of Chemistry, University of Copenhagen, 2100 Copenhagen Ø, Denmark.
Heres my email: _mastr@chem.ku.dk_.
