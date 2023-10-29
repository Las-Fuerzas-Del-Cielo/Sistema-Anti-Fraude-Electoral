from pathlib import Path
from typing import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image
from plotly.offline import plot
import IPython.display
import tempfile

plt.rcParams['figure.figsize'] = [10, 10]

from .image import scale_image
from .mapping import scale_map, create_identity_map
from .misc import to_numpy_image, to_numpy_map


def visualize_image(image: Union[Path, np.ndarray, torch.Tensor]):
    image = to_numpy_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


def visualize_mask(mask: Union[Path, np.ndarray, torch.Tensor]):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.imshow(mask) 
    fig.show()


def visualize_matplotlib(data: np.ndarray, return_image: bool = False):
    
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    data = np.squeeze(data)

    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.colorbar(axes.imshow(data))

    if return_image:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "image.png"
            plt.savefig(temp_file , bbox_inches='tight', dpi=250)
            data = cv2.imread(str(temp_file))
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        return data
    else:
        fig.show()


def visualize_matplotlib2(data1: np.ndarray, data2: np.ndarray, return_image: bool = False):
    
    if isinstance(data1, torch.Tensor):
        data1 = data1.detach().cpu().numpy()

    if isinstance(data2, torch.Tensor):
        data2 = data2.detach().cpu().numpy()

    data1 = np.squeeze(data1)
    data2 = np.squeeze(data2)

    vmin = np.nanmin(np.stack([data1, data2]))
    vmax = np.nanmax(np.stack([data1, data2]))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(data1, vmin=vmin, vmax=vmax)
    ax2.imshow(data2, vmin=vmin, vmax=vmax)        

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cax = fig.add_axes([0.85, 0.356, 0.025, .295])
    fig.colorbar(sm, cax=cax)
    fig.subplots_adjust(right=0.8, wspace=0.3)

    if return_image:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "image.png"
            plt.savefig(temp_file , bbox_inches='tight', dpi=250)
            data = cv2.imread(str(temp_file))
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        return data
    else:
        fig.show()



def visualize_bm(image: Union[Path, np.ndarray, torch.Tensor], bm: Union[Path, np.ndarray], resolution: int = 16,
                 title: str = None, show_arrows: bool = False, interactive: bool = False):
    image = to_numpy_image(image)
    image = scale_image(image, area=600*600)
    bm = to_numpy_map(bm)
    if resolution is not None:
        bm = scale_map(bm, resolution)
    state = create_identity_map(bm.shape[0], with_margin=True)
    visualize_sequence(image=image, gt=state, state_sequence=[bm], title=title, show_gt=show_arrows, interactive=interactive)


def visualize_sequence(*, image: np.ndarray, gt: np.ndarray, state_sequence: List[np.ndarray], show_gt: bool = True,
                       export_file: Path = None, title: str = None, interactive: bool = False):
    gridcolor = "salmon"
    arrowcolor = "MediumSpringGreen"

    # prepare bm data
    bm_data = gt

    # prepare background image
    h, w, _ = image.shape
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    resolution, _, _ = gt.shape

    # create frames
    def create_frame(frame_idx, state):
        shift = bm_data - state
        new_data = state + shift
        data = state

        lines1 = [go.Scatter(x=data[i, :, 1], y=data[i, :, 0], mode='lines', name='', hoverinfo="skip",
                             marker=dict(color=gridcolor))
                  for i in range(resolution)]

        lines2 = [go.Scatter(x=data[:, i, 1], y=data[:, i, 0], mode='lines', name='', hoverinfo="skip",
                             marker=dict(color=gridcolor))
                  for i in range(resolution)]

        mesh_texts = [
            "index:   (y: {},        x: {})<br>values: (y: {:4.3f}, x: {:4.3f})".format(py, px, data[py, px, 0],
                                                                                        data[py, px, 1]) for py in
            range(resolution) for px in range(resolution)]
        mesh_points = go.Scatter(x=data[:, :, 1].reshape(-1), y=data[:, :, 0].reshape(-1), mode='markers', name='',
                                 marker=dict(color=gridcolor), text=mesh_texts, hoverinfo="text")

        if show_gt:
            arrows = [go.layout.Annotation(
                dict(x=new_data[y, x, 1], y=new_data[y, x, 0], xref="x", yref="y", text="", showarrow=True, axref="x",
                     ayref='y', ax=data[y, x, 1], ay=data[y, x, 0], arrowhead=3, arrowwidth=2, arrowcolor=arrowcolor))
                for y in range(resolution)
                for x in range(resolution)]

            arrow_texts = ["(dy: {:.3f}, dx: {:.3f})".format(dy, dx) for dy, dx in shift.reshape(-1, 2)]
            arrow_points = [
                go.Scatter(x=new_data[:, :, 1].reshape(-1), y=new_data[:, :, 0].reshape(-1), mode='markers', name='',
                           marker=dict(color=arrowcolor), opacity=0, text=arrow_texts, hoverinfo="text")]
        else:
            arrows = []
            arrow_points = []

        return go.Frame(data=lines1 + lines2 + [mesh_points] + arrow_points, layout=go.Layout(dict(annotations=arrows)),
                        name=f"frame_{frame_idx}")

    frames = [create_frame(frame_idx, state) for frame_idx, state in enumerate(state_sequence)]

    # create slider
    sliders = [dict(steps=[dict(method='animate',
                                args=[[f'frame_{k}'],
                                      dict(mode='immediate', frame=dict(duration=0, redraw=False),
                                           transition=dict(duration=0))],
                                label=f'{k + 1}')
                           for k in range(len(state_sequence))],
                    active=0,
                    transition=dict(duration=0),
                    x=0,
                    y=0,
                    currentvalue=dict(font=dict(size=12), prefix='frame: ', visible=True, xanchor='center'),
                    len=1.0)]

    # create figure
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.add_layout_image(
        dict(source=img, xref="x", yref="y", x=0, y=0, sizex=1, sizey=1, sizing="stretch", opacity=0.8, layer="below"))
    fig.update_layout(showlegend=False, width=800, height=800, xaxis=dict(range=[0, 1], autorange=False),
                      yaxis=dict(range=[0, 1], autorange=False), sliders=sliders,
                      annotations=frames[0].layout.annotations)
    fig.update_xaxes(gridcolor='rgba(255,255,255, 0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255, 0.1)', scaleanchor="x", scaleratio=h / w, autorange="reversed")

    if title is not None:
        fig.update_layout(title=title)

    # export and show
    if export_file is not None:
        assert export_file.suffix == ".html"
        plot(fig, auto_play=False, filename=str(export_file.resolve()))

    if interactive:
        fig.show()
    else:
        img_bytes = fig.to_image(format="png")
        IPython.display.display(IPython.display.Image(img_bytes))
