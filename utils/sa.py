### SA functionality


class SaScorer:
    def sa_prep(self):
        for mol in self.molecules:
            prim_match = Chem.MolFromSmarts("[NX3;H2]")
            # Remove the cut idx amine to prevent it hogging the SA score
            removed_mol = single_atom_remover(mol.rdkit_mol, mol.cut_idx)
            mol.rdkit_mol_sa = removed_mol
            mol.smiles_sa = Chem.MolToSmiles(removed_mol)

            _neutralize_reactions = read_neutralizers()

        neutral_molecules = []
        for ind in self.molecules:
            c_mol = ind.rdkit_mol_sa
            mol = copy.deepcopy(c_mol)
            mol.UpdatePropertyCache()
            Chem.rdmolops.FastFindRings(mol)
            assert mol is not None
            for reactant_mol, product_mol in _neutralize_reactions:
                while mol.HasSubstructMatch(reactant_mol):
                    rms = Chem.ReplaceSubstructs(mol, reactant_mol, product_mol)
                    if rms[0] is not None:
                        mol = rms[0]
            mol.UpdatePropertyCache()
            Chem.rdmolops.FastFindRings(mol)
            ind.neutral_rdkit_mol = mol

    def get_sa(self):
        """Get the SA score of the population."""

        # Neutralize and prep molecules
        self.sa_prep()

        # Get the scores
        sa_scores = [
            sa_target_score_clipped(ind.neutral_rdkit_mol) for ind in self.molecules
        ]
        # Set the scores
        self.set_sa(sa_scores)

    def set_sa(self, sa_scores):
        """Set sa score.

        If score is high, then score is not modified
        """
        for individual, sa_score in zip(self.molecules, sa_scores):
            individual.sa_score = sa_score
            # Scale the score with the sa_score (which is max 1)
            individual.score = sa_score * individual.pre_score
