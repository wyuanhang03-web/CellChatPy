#!/usr/bin/env python3
"""
Database management for CellChatPy
Handles loading and querying of ligand-receptor interaction databases
based on the new-format CellChatDB (CellChatDB.human / .mouse / .zebrafish).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Sequence, Union
import warnings
from pathlib import Path
import matplotlib.pyplot as plt


class CellChatDB:
    """
    CellChatDB - Ligand-receptor interaction database manager.

    Expects the following directory layout (new format only):
        <db_dir>/
            CellChatDB.human/
                interaction.csv   (index_col=0 = R interaction name)
                complex.csv       (index_col=0 = R shorthand, e.g. CALCRL_RAMP1)
                cofactor.csv      (index_col=0 = R cofactor shorthand)
                geneInfo.csv      (index_col=0)
            CellChatDB.mouse/   (same structure)
            CellChatDB.zebrafish/ (same structure)
    """

    def __init__(self, db_dir: Optional[str] = None):
        if db_dir is None:
            self.db_dir = Path(__file__).parent / "CellChatDB"
        else:
            self.db_dir = Path(db_dir)

        self.databases: Dict[str, Dict] = {}
        self.available_species: List[str] = []
        self._load_available_databases()

    # ------------------------------------------------------------------
    def _load_available_databases(self):
        """Discover which species have a CellChatDB.<species> directory."""
        if not self.db_dir.exists():
            warnings.warn(f"Database directory not found: {self.db_dir}")
            return

        known = {'human', 'mouse', 'zebrafish'}
        for d in sorted(self.db_dir.iterdir()):
            if not d.is_dir():
                continue
            name = d.name  # e.g. "CellChatDB.human"
            for sp in known:
                if name == f"CellChatDB.{sp}":
                    if sp not in self.available_species:
                        self.available_species.append(sp)
                    break

    # ------------------------------------------------------------------
    def load_database(self, species: str = "human") -> Dict[str, pd.DataFrame]:
        """
        Load the CellChatDB for *species* from CellChatDB.<species>/.

        Returns
        -------
        dict with keys: 'interaction', 'complex', 'cofactor', 'gene_info'
        Each value is a DataFrame whose index matches R rownames.
        """
        species = species.lower()
        if species not in self.available_species:
            raise ValueError(
                f"Species '{species}' not available. "
                f"Available: {self.available_species}"
            )

        db_path = self.db_dir / f"CellChatDB.{species}"

        tables = {
            'interaction': 'interaction.csv',
            'complex':     'complex.csv',
            'cofactor':    'cofactor.csv',
            'gene_info':   'geneInfo.csv',
        }

        database: Dict[str, pd.DataFrame] = {}
        for key, filename in tables.items():
            fpath = db_path / filename
            if fpath.exists():
                df = pd.read_csv(fpath, index_col=0)
                if key in ('complex', 'cofactor'):
                    df = df.fillna('')
                database[key] = df
            else:
                database[key] = pd.DataFrame()
                if key == 'interaction':
                    raise FileNotFoundError(
                        f"interaction.csv not found in {db_path}"
                    )

        self.databases[species] = database
        return database

    # ------------------------------------------------------------------
    def get_database(self, species: str = "human") -> Dict[str, pd.DataFrame]:
        """Return cached database, loading it first if necessary."""
        species = species.lower()
        if species not in self.databases:
            return self.load_database(species)
        return self.databases[species]

    # ------------------------------------------------------------------
    def show_database_category(
        self,
        database: Dict[str, pd.DataFrame],
        nrow: int = 1
    ) -> plt.Figure:
        """Three pie charts showing interaction database composition (matching R show_database_category)."""
        interaction = database['interaction']
        gene_info = database.get('gene_info', pd.DataFrame())

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Chart 1: Annotation categories
        if 'annotation' in interaction.columns:
            counts = interaction['annotation'].value_counts()
            # Order matching R: Secreted Signaling, ECM-Receptor, Cell-Cell Contact, Non-protein Signaling
            order = ['Secreted Signaling', 'ECM-Receptor', 'Cell-Cell Contact', 'Non-protein Signaling']
            counts = counts.reindex([c for c in order if c in counts.index])
            colors1 = ['#F8766D', '#7CAE00', '#00BFC4', '#C77CFF'][:len(counts)]
            axes[0].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                       colors=colors1, startangle=90)
            axes[0].set_title('Interaction type')

        # Chart 2: Heterodimers vs Others
        if not gene_info.empty and 'Symbol' in gene_info.columns:
            gene_symbols = set(gene_info['Symbol'].dropna())
            heterodimer_count = 0
            other_count = 0

            for _, row in interaction.iterrows():
                ligand = row.get('ligand', '')
                receptor = row.get('receptor', '')
                # Heterodimer: ligand OR receptor not in geneInfo
                if ligand not in gene_symbols or receptor not in gene_symbols:
                    heterodimer_count += 1
                else:
                    other_count += 1

            counts2 = pd.Series({'Heterodimers': heterodimer_count, 'Others': other_count})
            colors2 = ['#F8766D', '#00BFC4']
            axes[1].pie(counts2.values, labels=counts2.index, autopct='%1.1f%%',
                       colors=colors2, startangle=90)
            axes[1].set_title('Heterodimers')

        # Chart 3: KEGG vs Literature
        if 'evidence' in interaction.columns:
            kegg_count = interaction['evidence'].str.contains('KEGG', case=False, na=False).sum()
            literature_count = len(interaction) - kegg_count
            counts3 = pd.Series({'KEGG': kegg_count, 'Literature': literature_count})
            colors3 = ['#F8766D', '#00BFC4']
            axes[2].pie(counts3.values, labels=counts3.index, autopct='%1.1f%%',
                       colors=colors3, startangle=90)
            axes[2].set_title('Evidence')

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def search_pair(
        self,
        database: Dict[str, pd.DataFrame],
        signaling: Union[str, List[str]],
        key: str = "pathway_name",
        matching_exact: bool = False,
        pair_only: bool = True,
    ) -> pd.DataFrame:
        """
        Search interactions by pathway name, ligand, or receptor.

        Parameters
        ----------
        signaling : str | list
        key       : column to search ('pathway_name', 'ligand', 'receptor')
        matching_exact : exact match when True, substring match when False
        pair_only : return only [interaction_name, pathway_name, ligand, receptor]
        """
        if isinstance(signaling, str):
            signaling = [signaling]

        interaction = database['interaction']
        if key not in interaction.columns:
            raise ValueError(
                f"Column '{key}' not found. "
                f"Available: {list(interaction.columns)}"
            )

        masks = []
        for sig in signaling:
            if matching_exact:
                masks.append(interaction[key] == sig)
            else:
                masks.append(
                    interaction[key].str.contains(sig, case=False, na=False)
                )

        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m

        result = interaction[combined].copy()

        if pair_only:
            keep = [c for c in
                    ['interaction_name', 'pathway_name', 'ligand', 'receptor']
                    if c in result.columns]
            result = result[keep]

        return result.drop_duplicates().reset_index(drop=True)

    # ------------------------------------------------------------------
    def get_available_species(self) -> List[str]:
        return self.available_species.copy()


# Module-level convenience functions

search_spec = Union[str, Sequence[str]]


def _as_search_list(value: search_spec) -> List[str]:
    """Normalize one R-style search entry to a list of accepted values."""
    if isinstance(value, str):
        return [value]
    return list(value)


def _normalize_filters(search, key) -> List[tuple[str, List[str]]]:
    """Return filters for single- or multi-column database searches."""
    if isinstance(key, str):
        return [(key, _as_search_list(search))]

    if search is None:
        raise ValueError("'search' must be provided when 'key' contains multiple columns.")

    keys = list(key)
    if not isinstance(search, Sequence) or isinstance(search, str):
        raise ValueError("When 'key' is a list, 'search' must be a list with one entry per key.")
    searches = list(search)
    if len(keys) != len(searches):
        raise ValueError(
            f"Length mismatch: key has {len(keys)} entries but search has "
            f"{len(searches)} entries."
        )
    return [(col, _as_search_list(values)) for col, values in zip(keys, searches)]

def load_database(
    species: str = "human",
    db_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """Load CellChatDB for *species* from *db_dir*."""
    return CellChatDB(db_dir).load_database(species)


def set_database(
    cellchat_obj,
    species: str = "human",
    database: Optional[Dict[str, pd.DataFrame]] = None,
    db_dir: Optional[str] = None,
):
    """Attach *database* (or load it) to a CellChat object."""
    if database is None:
        database = load_database(species, db_dir)
    cellchat_obj.database = database
    return cellchat_obj


def subset_database(
    cellchat_db: Dict[str, pd.DataFrame],
    search: Optional[Union[search_spec, Sequence[search_spec]]] = None,
    key: Union[str, Sequence[str]] = "annotation",
    non_protein: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Subset CellChatDB by annotation or other interaction-table criteria.

    Values within one column are combined with OR. Filters across columns
    are combined with AND. When no annotation search is supplied, the three
    protein-signaling categories are selected by default.
    """
    database_subset = {
        'interaction': cellchat_db['interaction'].copy(),
        'complex': cellchat_db.get('complex', pd.DataFrame()).copy(),
        'cofactor': cellchat_db.get('cofactor', pd.DataFrame()).copy(),
        'gene_info': cellchat_db.get('gene_info', pd.DataFrame()).copy(),
    }
    interaction_input = database_subset['interaction']

    if search is None and key == "annotation":
        search = ["Secreted Signaling", "ECM-Receptor", "Cell-Cell Contact"]
        if non_protein:
            search.append("Non-protein Signaling")

    filters = _normalize_filters(search, key) if search is not None else []

    if any("Non-protein Signaling" in values for _, values in filters):
        non_protein = True
        print(
            "The non-protein signaling is now included for CellChat analysis, "
            "which is usually used for neuron-neuron and metabolic communication!"
        )

    if not non_protein and 'annotation' in interaction_input.columns:
        interaction_input = interaction_input[
            interaction_input['annotation'] != "Non-protein Signaling"
        ].copy()

    missing = [col for col, _ in filters if col not in interaction_input.columns]
    if missing:
        raise ValueError(
            f"Key(s) {missing} not found in interaction table. "
            f"Available columns: {list(interaction_input.columns)}"
        )

    for col, values in filters:
        interaction_input = interaction_input[
            interaction_input[col].isin(values)
        ].copy()

    if interaction_input.empty:
        raise ValueError(f"No interactions found matching search criteria: {filters}")

    database_subset['interaction'] = interaction_input
    return database_subset


def show_database_category(
    database: Dict[str, pd.DataFrame],
    nrow: int = 1
) -> plt.Figure:
    """Show interaction counts per annotation category."""
    return CellChatDB().show_database_category(database, nrow)


def search_pair(
    signaling: Union[str, List[str]],
    pair_lr_use: pd.DataFrame,
    key: str = "pathway_name",
    matching_exact: bool = False,
    pair_only: bool = True,
) -> pd.DataFrame:
    """Search a ligand-receptor DataFrame for matching signaling entries."""
    return CellChatDB().search_pair(
        {'interaction': pair_lr_use}, signaling, key, matching_exact, pair_only
    )


def get_available_databases(db_dir: Optional[str] = None) -> List[str]:
    """Return available species names in *db_dir*."""
    return CellChatDB(db_dir).get_available_species()


def extract_genes(cellchat_db: Dict[str, pd.DataFrame]) -> List[str]:
    """Return sorted list of all unique ligand and receptor gene names."""
    interaction = cellchat_db.get('interaction', pd.DataFrame())
    genes: set = set()
    for col in ('ligand', 'receptor'):
        if col in interaction.columns:
            genes.update(interaction[col].dropna().unique())
    return sorted(genes)


def extract_lr_from_genes(
    gene_set: List[str],
    cellchat_db: Dict[str, pd.DataFrame],
) -> Dict:
    """Return LR pairs (and matching complexes) that involve *gene_set*."""
    interaction = cellchat_db.get('interaction', pd.DataFrame())
    complex_df  = cellchat_db.get('complex',     pd.DataFrame())

    lr_pairs = interaction[
        interaction['ligand'].isin(gene_set) |
        interaction['receptor'].isin(gene_set)
    ].copy()

    complex_genes: set = set()
    if not complex_df.empty:
        sub_cols = [c for c in complex_df.columns if 'subunit' in c.lower()]
        for idx, row in complex_df.iterrows():
            subunits = {str(row[c]) for c in sub_cols if row[c]}
            if subunits & set(gene_set):
                complex_genes.add(idx)
        extra = interaction[
            interaction['ligand'].isin(complex_genes) |
            interaction['receptor'].isin(complex_genes)
        ]
        lr_pairs = pd.concat([lr_pairs, extra]).drop_duplicates()

    return {
        'lr_pairs': lr_pairs,
        'genes':    gene_set,
        'complexes': list(complex_genes),
    }


def filter_database_by_category(
    cellchat_db: Dict[str, pd.DataFrame],
    categories: List[str],
) -> Dict[str, pd.DataFrame]:
    """Return a copy of *cellchat_db* with interaction filtered to *categories*."""
    filtered = dict(cellchat_db)
    interaction = filtered.get('interaction', pd.DataFrame())
    if not interaction.empty:
        col = 'annotation' if 'annotation' in interaction.columns else 'category'
        if col in interaction.columns:
            filtered['interaction'] = interaction[
                interaction[col].isin(categories)
            ].copy()
    return filtered


def get_pathway_interactions(
    cellchat_db: Dict[str, pd.DataFrame],
    pathways: List[str],
) -> pd.DataFrame:
    """Return all interactions belonging to *pathways*."""
    interaction = cellchat_db.get('interaction', pd.DataFrame())
    if 'pathway_name' in interaction.columns:
        return interaction[interaction['pathway_name'].isin(pathways)].copy()
    return pd.DataFrame()


def validate_database(cellchat_db: Dict[str, pd.DataFrame]) -> Dict:
    """Basic structure/content validation."""
    result = {'is_valid': True, 'errors': [], 'warnings': [], 'stats': {}}

    if 'interaction' not in cellchat_db:
        result['errors'].append("Missing key: 'interaction'")
        result['is_valid'] = False
        return result

    interaction = cellchat_db['interaction']
    for col in ('ligand', 'receptor'):
        if col not in interaction.columns:
            result['errors'].append(f"Missing column: '{col}'")
            result['is_valid'] = False

    result['stats']['n_interactions']      = len(interaction)
    result['stats']['n_unique_ligands']    = interaction.get('ligand',    pd.Series()).nunique()
    result['stats']['n_unique_receptors']  = interaction.get('receptor',  pd.Series()).nunique()
    result['stats']['n_pathways']          = interaction.get('pathway_name', pd.Series()).nunique()

    dups = interaction.duplicated().sum()
    if dups:
        result['warnings'].append(f"{dups} duplicate interactions found")

    return result


def check_gene_symbol(
    gene_set: List[str],
    gene_info: pd.DataFrame,
) -> bool:
    """
    Check if genes in *gene_set* have official gene symbols in *gene_info*.

    Mirrors R check_gene_symbol: warns for genes not found in geneInfo$Symbol.

    Returns False (R always returns FALSE; the side effect is the warning).
    """
    gene_set = list(set([str(g) for g in gene_set if str(g) != '']))
    official_symbols = set()
    if 'Symbol' in gene_info.columns:
        official_symbols = set(gene_info['Symbol'].dropna().unique())
    else:
        official_symbols = set(gene_info.index.astype(str))

    genes_not_official = [g for g in gene_set if g not in official_symbols]
    if len(genes_not_official) > 0:
        print(
            "Issue identified!! Please check the official Gene Symbol "
            f"of the following genes: \n{genes_not_official}\n"
        )
    return False


def update_database(
    interactions: pd.DataFrame,
    gene_info: Optional[pd.DataFrame] = None,
    other_info: Optional[Dict[str, pd.DataFrame]] = None,
    gene_info_column_new: Optional[pd.DataFrame] = None,
    trim_pathway: bool = False,
    merged: bool = False,
    species_target: Optional[str] = None,
    db_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build a new CellChatDB from a user-provided L-R interaction table.

    Mirrors R update_database.

    Parameters
    ----------
    interactions : DataFrame
        Must contain at least 'ligand' and 'receptor' columns.
    gene_info : DataFrame, optional
        Must contain at least a 'Symbol' column.
    other_info : dict, optional
        {'complex': DataFrame, 'cofactor': DataFrame}.
    gene_info_column_new : DataFrame, optional
        Must contain 'Symbol' and 'AntibodyName' columns.
    trim_pathway : bool
        Remove interactions with missing pathway_name.
    merged : bool
        Merge with the built-in CellChatDB.
    species_target : str
        'human' or 'mouse'.
    db_dir : str, optional
        Path to built-in CellChatDB directory.

    Returns
    -------
    dict with keys: 'interaction', 'complex', 'cofactor', 'gene_info'
    """
    interaction_table = interactions.copy()
    for col in interaction_table.columns:
        interaction_table[col] = interaction_table[col].astype(str)

    if 'ligand' not in interaction_table.columns or 'receptor' not in interaction_table.columns:
        raise ValueError(
            "The input `interactions` must contain at least two columns named ligand and receptor."
        )

    if 'pathway_name' not in interaction_table.columns:
        warnings.warn(
            "The pathway_name associated with each L-R pair is not provided in "
            "`interactions`. "
            "We suggest to provide this information."
        )
        interaction_table['pathway_name'] = ''
    else:
        pathway_missing = interaction_table['pathway_name'] == ''
        if pathway_missing.sum() > 0:
            if trim_pathway:
                n_missing = pathway_missing.sum()
                print(f"The pathway names of {n_missing} interactions are missing "
                      "and the corresponding interactions are now deleted.")
                interaction_table = interaction_table[~pathway_missing].copy()
            else:
                n_missing = pathway_missing.sum()
                warnings.warn(
                    f"The pathway names of {n_missing} interactions are missing. "
                    "Setting `trim_pathway = True` to avoid possible errors."
                )

    if 'interaction_name' not in interaction_table.columns:
        interaction_table['interaction_name'] = (
            interaction_table['ligand'].str.upper() + '_' + interaction_table['receptor'].str.upper()
        )

    if 'interaction_name_2' not in interaction_table.columns:
        interaction_table['interaction_name_2'] = (
            interaction_table['ligand'] + ' - ' + interaction_table['receptor']
        )

    for col in ['agonist', 'antagonist', 'co_A_receptor', 'co_I_receptor']:
        if col not in interaction_table.columns:
            interaction_table[col] = ''

    dup_mask = interaction_table['interaction_name'].duplicated()
    if dup_mask.sum() > 0:
        warnings.warn(
            f"{dup_mask.sum()} duplicated interaction_names identified "
            "and the corresponding interactions are now deleted."
        )
        interaction_table = interaction_table[~dup_mask]

    interaction_input = interaction_table.copy()
    interaction_input.index = interaction_input['interaction_name']
    interaction_input.index.name = None

    cols_default = [
        'interaction_name', 'pathway_name', 'ligand', 'receptor',
        'agonist', 'antagonist', 'co_A_receptor', 'co_I_receptor',
        'annotation', 'interaction_name_2'
    ]
    cols_common = [c for c in cols_default if c in interaction_input.columns]
    cols_specific = [c for c in interaction_input.columns if c not in cols_default]
    interaction_input = interaction_input[cols_common + cols_specific]

    if other_info is not None:
        complex_input = other_info.get('complex', pd.DataFrame())
        cofactor_input = other_info.get('cofactor', pd.DataFrame())
    else:
        complex_input = pd.DataFrame()
        cofactor_input = pd.DataFrame()

    if gene_info is not None:
        if 'Symbol' not in gene_info.columns:
            raise ValueError(
                "The input `gene_info` must contain at least one column named as `Symbol`"
            )
        gene_info_input = gene_info.copy()
    else:
        if species_target is None:
            raise ValueError(
                "When setting gene_info = NULL, the input `species_target` "
                "should be provided: either `human` or `mouse`."
            )
        cdb = CellChatDB(db_dir)
        builtin = cdb.load_database(species_target)
        gene_info_input = builtin.get('gene_info', pd.DataFrame())

    if merged:
        if species_target is None:
            raise ValueError(
                "When setting merged = True, the input `species_target` "
                "should be provided: either `human` or `mouse`."
            )
        cdb = CellChatDB(db_dir)
        db_cellchat = cdb.load_database(species_target)

        interaction_cellchat = db_cellchat['interaction'].copy()
        interaction_cellchat['source_merged'] = 'CellChatDB'
        interaction_input['source_merged'] = 'User'
        cols_common_inter = [
            c for c in interaction_input.columns
            if c in interaction_cellchat.columns
        ]
        interaction_input_trim = interaction_input[cols_common_inter]
        interaction_cellchat_trim = interaction_cellchat[cols_common_inter]
        interaction_input = pd.concat(
            [interaction_cellchat_trim, interaction_input_trim],
            ignore_index=True
        )
        dup_mask = interaction_input['interaction_name'].duplicated()
        if dup_mask.sum() > 0:
            interaction_input = interaction_input[~dup_mask]

        complex_cellchat = db_cellchat.get('complex', pd.DataFrame())
        num_subunit = max(
            len(complex_input.columns) if not complex_input.empty else 0,
            len(complex_cellchat.columns) if not complex_cellchat.empty else 0
        )
        if num_subunit > 0:
            if not complex_input.empty:
                existing = len(complex_input.columns)
                for k in range(existing, num_subunit):
                    complex_input[f'subunit_{k+1}'] = ''
            if not complex_cellchat.empty:
                existing = len(complex_cellchat.columns)
                for k in range(existing, num_subunit):
                    complex_cellchat[f'subunit_{k+1}'] = ''
            complex_input = pd.concat([complex_cellchat, complex_input])
            dup_idx = complex_input.index.duplicated()
            if dup_idx.sum() > 0:
                complex_input = complex_input[~dup_idx]

        cofactor_cellchat = db_cellchat.get('cofactor', pd.DataFrame())
        num_co = max(
            len(cofactor_input.columns) if not cofactor_input.empty else 0,
            len(cofactor_cellchat.columns) if not cofactor_cellchat.empty else 0
        )
        if num_co > 0:
            if not cofactor_input.empty:
                existing = len(cofactor_input.columns)
                for k in range(existing, num_co):
                    cofactor_input[f'cofactor{k+1}'] = ''
            if not cofactor_cellchat.empty:
                existing = len(cofactor_cellchat.columns)
                for k in range(existing, num_co):
                    cofactor_cellchat[f'cofactor{k+1}'] = ''
            cofactor_input = pd.concat([cofactor_cellchat, cofactor_input])
            dup_idx = cofactor_input.index.duplicated()
            if dup_idx.sum() > 0:
                cofactor_input = cofactor_input[~dup_idx]

    if gene_info_column_new is not None:
        check_gene_symbol(
            list(gene_info_column_new['Symbol'].astype(str)),
            gene_info_input
        )
        gene_info_input['AntibodyName'] = pd.NA
        for _, row in gene_info_column_new.iterrows():
            sym = str(row['Symbol'])
            if sym in gene_info_input.index:
                gene_info_input.at[sym, 'AntibodyName'] = row.get('AntibodyName', pd.NA)
            elif 'Symbol' in gene_info_input.columns:
                mask = gene_info_input['Symbol'] == sym
                gene_info_input.loc[mask, 'AntibodyName'] = row.get('AntibodyName', pd.NA)

    db_new = {
        'interaction': interaction_input,
        'complex': complex_input if not complex_input.empty else pd.DataFrame(),
        'cofactor': cofactor_input if not cofactor_input.empty else pd.DataFrame(),
        'gene_info': gene_info_input,
    }

    return db_new


