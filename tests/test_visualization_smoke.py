"""Structural smoke tests for representative plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

import CellChatPy as cc


def test_network_circle_returns_a_nonblank_figure(network_cellchat) -> None:
    figure = cc.plot_network_circle(
        network_cellchat,
        slot_name="network",
        return_fig=True,
        label_edge=True,
    )
    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1
        assert len(figure.axes[0].patches) + len(figure.axes[0].collections) > 0
    finally:
        plt.close(figure)


def test_network_heatmap_returns_labeled_figure(network_cellchat) -> None:
    figure = cc.plot_network_heatmap(
        network_cellchat,
        slot_name="network",
        measure="weight",
        return_fig=True,
    )
    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) >= 1
        labels = {
            text.get_text()
            for axis in figure.axes
            for text in (*axis.texts, *axis.get_xticklabels(), *axis.get_yticklabels())
        }
        assert {"A", "B"}.issubset(labels)
    finally:
        plt.close(figure)


def test_network_circle_rejects_matrix_with_wrong_group_shape(toy_cellchat) -> None:
    with pytest.raises(ValueError, match="square matrix"):
        cc.plot_network_circle(
            toy_cellchat,
            net_matrix=np.ones((3, 3)),
            group_names=["A", "B"],
            return_fig=True,
        )
