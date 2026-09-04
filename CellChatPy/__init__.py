#!/usr/bin/env python3
"""
CellChatPy - A comprehensive toolkit for analyzing intercellular communication from single-cell transcriptomics data

This package provides functions for data preprocessing, intercellular communication
network inference, communication network analysis, and visualization.
"""

__version__ = "1.1.0"
__author__ = "CellChatPy Team"
__email__ = "cellchat@example.com"

# Import main classes and functions
from .cellchat_class import (
    CellChat, create_cellchat, merge_cellchat, subset_cellchat, lift_cellchat, set_identity,
    intersection
)

from .database import (
    CellChatDB, load_database, set_database, subset_database, show_database_category, search_pair,
    get_available_databases, extract_genes, extract_lr_from_genes,
    filter_database_by_category, get_pathway_interactions, validate_database,
    check_gene_symbol, update_database
)

# Import analysis functions
from .analysis import (
    normalize_data, preprocess_signaling_data, scale_data, scale_matrix, sketch_data, add_metadata,
    add_reduction, update_cluster_labels, subset_signaling_data, identify_overexpressed_genes,
    identify_overexpressed_ligand_receptor, identify_overexpressed_interactions,
    smooth_data, run_pca, run_umap,
    net_analysis_signaling_role_scatter, net_analysis_signaling_role_heatmap,
    net_analysis_signaling_role_network, rank_net, plot_gene_expression,
    extract_enriched_lr, identify_communication_patterns, select_k,
    compute_network_similarity, embed_network, cluster_network,
    # Comparative network analysis
    compute_pairwise_network_similarity, rank_similarity, compare_interactions,
    rank_network_pairwise, get_max_weight, merge_interactions,
    map_network_deg, compute_enrichment_score, find_enriched_signaling,
    compute_laplacian, compute_eigengap,
    net_analysis_diff_signaling_role_scatter, net_analysis_signaling_changes_scatter,
    color_ramp, extract_gene_subset_from_pair
)

# Import modeling functions
from .modeling import (
    compute_communication_probability, compute_pathway_probability, aggregate_network,
    aggregate_signaling_matrix,
    filter_communication, compute_network_centrality, subset_communication,
    identify_enriched_interactions, compute_region_distance,
    communication_dataframe_to_network,
    compute_spot_distances, compute_spot_communication_probability,
    filter_spot_probability, filter_spot_communication,
    aggregate_visium_communication, compute_spot_pathway_probability,
    compute_spot_network_centrality,
)

# Import visualization functions
from .visualization import (
    # Color helpers
    sc_palette, gg_palette,
    # Visualization functions
    plot_network_circle, plot_network_heatmap, plot_network_bubble, plot_network_barplot,
    plot_network_embedding, plot_network_aggregate, plot_network_individual,
    plot_network_chord_cell, plot_network_chord_gene, plot_analysis_contribution,
    plot_analysis_dot, plot_analysis_river,
    # Extended visualization functions
    plot_network_circle_grid,
    plot_select_k,
    plot_analysis_pattern_heatmap,
    plot_analysis_signaling_role_scatter_dual,
    plot_analysis_signaling_role_heatmap_combined,
    plot_network_embedding_by_group,
    # Ranking / network plots
    plot_network_rank, plot_network, plot_network_layout,
    cellchat_theme_options, plot_network_hierarchy_1, plot_network_hierarchy_2,
    plot_network_spatial, plot_network_diff_interaction,
    plot_network_embedding_zoom_in, plot_network_embedding_pairwise,
    plot_network_embedding_pairwise_zoom_in,
    plot_spatial_dim, plot_spatial_feature,
    plot_spatial_values, plot_spatial_proportions, plot_spatial_categories,
    plot_spot_signaling_scores, plot_spatial_gi, plot_spatial_lee,
    plot_spatial_topics, plot_topic_composition, plot_topic_signaling,
    plot_dot, plot_stacked_violin, plot_bar
)

# Import utility functions
from .utilities import (
    update_communication_score, preprocess_multiomics, validate_cellchat
)

from .utilities import (
    spatial_h5_schema, spatial_h5_schema_version,
    read_spatial_h5, write_spatial_h5, merge_spatial_inputs,
    read_visium_spatial_info, compute_cell_distance,
)
from .analysis import (
    identify_spatially_variable_genes, get_spot_signaling_scores,
    compute_getis_ord_gi, compute_spatial_gi, compute_spatial_lee,
    identify_cell_topics,
)
from .spatial import (
    compute_colocalization, spatial_visual_scoring,
    communication_distance_plot, compute_grid_size, make_grid_spatial_cellchat,
)

__all__ = [
    # Main class and core functions
    'CellChat', 'create_cellchat', 'merge_cellchat', 'lift_cellchat', 'set_identity',

    # Database management
    'CellChatDB', 'load_database', 'set_database', 'subset_database', 'show_database_category', 'search_pair',
    'get_available_databases', 'extract_genes', 'extract_lr_from_genes',
    'filter_database_by_category', 'get_pathway_interactions', 'validate_database',
    'check_gene_symbol', 'update_database',

    # Analysis functions
    'normalize_data', 'preprocess_signaling_data', 'scale_data', 'scale_matrix', 'sketch_data', 'add_metadata',
    'add_reduction', 'update_cluster_labels', 'subset_signaling_data', 'identify_overexpressed_genes',
    'identify_overexpressed_ligand_receptor', 'identify_overexpressed_interactions',
    'smooth_data', 'run_pca', 'run_umap',
    'net_analysis_signaling_role_scatter', 'net_analysis_signaling_role_heatmap',
    'net_analysis_signaling_role_network', 'rank_net', 'plot_gene_expression',
    'extract_enriched_lr', 'identify_communication_patterns', 'select_k',
    'compute_network_similarity', 'embed_network', 'cluster_network',
    # Comparative network analysis
    'compute_pairwise_network_similarity', 'rank_similarity', 'compare_interactions',
    'rank_network_pairwise', 'get_max_weight', 'merge_interactions',
    'map_network_deg', 'compute_enrichment_score', 'find_enriched_signaling',
    'compute_laplacian', 'compute_eigengap',
    'net_analysis_diff_signaling_role_scatter', 'net_analysis_signaling_changes_scatter',
    'color_ramp', 'extract_gene_subset_from_pair',

    # Modeling functions
    'compute_communication_probability', 'compute_pathway_probability', 'aggregate_network',
    'aggregate_signaling_matrix',
    'filter_communication', 'compute_network_centrality', 'subset_communication',
    'identify_enriched_interactions', 'compute_region_distance',
    'communication_dataframe_to_network',

    # Visualization functions
    'sc_palette', 'gg_palette',
    'plot_network_circle', 'plot_network_heatmap', 'plot_network_bubble', 'plot_network_barplot',
    'plot_network_embedding', 'plot_network_aggregate', 'plot_network_individual',
    'plot_network_chord_cell', 'plot_network_chord_gene', 'plot_analysis_contribution',
    'plot_analysis_dot', 'plot_analysis_river',
    'plot_network_circle_grid',
    'plot_select_k',
    'plot_analysis_pattern_heatmap',
    'plot_analysis_signaling_role_scatter_dual',
    'plot_analysis_signaling_role_heatmap_combined',
    'plot_network_embedding_by_group',
    'plot_network_rank', 'plot_network', 'plot_network_layout',
    'cellchat_theme_options', 'plot_network_hierarchy_1', 'plot_network_hierarchy_2',
    'plot_network_spatial', 'plot_network_diff_interaction',
    'plot_network_embedding_zoom_in', 'plot_network_embedding_pairwise',
    'plot_network_embedding_pairwise_zoom_in',
    'plot_spatial_dim', 'plot_spatial_feature',
    'plot_dot', 'plot_stacked_violin', 'plot_bar',

    # Utility functions
    'subset_cellchat',
    'update_communication_score', 'preprocess_multiomics', 'validate_cellchat',

    # General spatial input/output helpers
    'spatial_h5_schema', 'spatial_h5_schema_version',
    'read_spatial_h5', 'write_spatial_h5', 'merge_spatial_inputs',
    'read_visium_spatial_info',
    'compute_cell_distance',

    # Spot-level spatial modeling and analysis
    'compute_spot_distances', 'compute_spot_communication_probability',
    'filter_spot_probability', 'filter_spot_communication',
    'aggregate_visium_communication', 'compute_spot_pathway_probability',
    'compute_spot_network_centrality',
    'identify_spatially_variable_genes', 'get_spot_signaling_scores',
    'compute_getis_ord_gi', 'compute_spatial_gi', 'compute_spatial_lee',
    'identify_cell_topics',
    'compute_colocalization', 'spatial_visual_scoring',
    'communication_distance_plot', 'compute_grid_size',
    'make_grid_spatial_cellchat',

    # Spot-level spatial visualization
    'plot_spatial_values', 'plot_spatial_proportions', 'plot_spatial_categories',
    'plot_spot_signaling_scores', 'plot_spatial_gi', 'plot_spatial_lee',
    'plot_spatial_topics', 'plot_topic_composition', 'plot_topic_signaling',

    # Helper functions
    'intersection'
]


