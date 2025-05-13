# from bash_command import bash_command as bc
import copy
import pickle
import warnings
from dataclasses import dataclass, field

import astropy.constants as const
import dash
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc

from computations import (get_bound_snaps_adjust, get_com_winf,
                          max_pairwise_dist)
##Need to make the code more self-contained??
from config.config import load_config
from movie_preamble import (comps_dict, ex_index, ex_time_max, ex_time_max_end,
                            high_df, lookup_dict, my_data, path_lookup)

##TO DO:REFACTOR PLOTLY SNAPSHOT -- REPETITIVE PLOTTING CODE (NOT NECESSARY)
##[GET STARTED ON REQUIREMENTS AND README...NEED REQUIREMENTS FOR DATA FILE...]
##[Add N-body data and do basic sanity checks...Add orbital data too]
##First priority: IS TO GET SELF CONTAINED USABLE CODE.

cfg = load_config()

pc = const.pc.cgs.value
au = const.au.cgs.value
unit = pc / au
ioffset = 10
###############################################################################
sink_cols = np.array(("t", "id", "px", "py", "pz", "vx", "vy", "vz", "h", "m"))
sink_cols = np.concatenate((sink_cols, ["sys_id", "mtot", "sma", "ecc"]))
mcol = np.where(sink_cols == "m")[0][0]
pxcol = np.where(sink_cols == "px")[0][0]
pycol = np.where(sink_cols == "py")[0][0]
pzcol = np.where(sink_cols == "pz")[0][0]
mcol = np.where(sink_cols == "m")[0][0]
mtotcol = np.where(sink_cols == "mtot")[0][0]
#################################################################################


@dataclass
class MovieData:
    ps: list = field(default_factory=list)
    time: int = field(default=0)
    comps: list = field(default_factory=list)
    comps_curr: list = field(default_factory=list)
    annotations: list = field(default_factory=list)
    max_sep: float = field(default=10000.0)
    max_sep_rel: float = field(default=5.0)
    halo_toggle: bool = field(default=True)
    ex_time_start: float = field(default=0.0)
    ex_time_max_end: float = field(default=0.0)


############Setting up the dashboard#############################################


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])  # Feel free to try others like LUX, MINTY, etc.
#
app.layout = dbc.Container([
    # --- Top Navbar ---
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Binary Snapshot Viewer", className="mx-auto fs-3 fw-bold text-white")
        ]),
        color="primary",
        dark=True,
        className="mb-4"
    ),
    # --- Main Grid ---
    dbc.Row([
        # Sidebar controls
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Controls", className="bg-light fw-bold"),
                dbc.CardBody([

                    dbc.Label("Binary Index"),
                    dcc.Input(id="bin-input", type="number", min=0, max=len(ex_index) - 1, value=0, className="form-control mb-3"),

                    dbc.Label("Time Step"),
                    dcc.Input(id="time-input", type="number", min=0, max=490,
                    value=max(int(ex_time_max[ex_index[0]]) - ioffset, my_data["fst"][ex_index[0]]), step=1, className="form-control mb-2"),
                    dbc.ButtonGroup([
                        dbc.Button("←", id="step-back", color="secondary"),
                        dbc.Button("→", id="step-forward", color="secondary"),
                    ], className="mb-4"),

                    html.Hr(),

                    dbc.Label("Absolute Plot Scale"),
                    dcc.Input(id="max-sep", type="number", value=10000, className="form-control mb-3"),

                    dbc.Label("Relative Plot Scale"),
                    dcc.Input(id="max-sep-rel", type="number", value=5, className="form-control mb-3"),

                    html.Hr(),

                    dbc.Label("Halo Particles"),
                    dbc.RadioItems(
                        id="halo_toggle",
                        options=[
                            {"label": "Include", "value": True},
                            {"label": "Exclude", "value": False}
                        ],
                        value=True,
                        inline=True,
                        className="mb-3"
                    ),

                    dbc.Label("Future Companions"),
                    dbc.RadioItems(
                        id="comp_toggle",
                        options=[
                            {"label": "Include", "value": True},
                            {"label": "Exclude", "value": False}
                        ],
                        value=True,
                        inline=True
                    )
                ])
            ])
        ], width=3),

        # Main plot area
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Absolute Scale View"),
                        dbc.CardBody([
                            dcc.Graph(id="snapshot-graph")
                        ])
                    ])
                ], md=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Relative Scale View"),
                        dbc.CardBody([
                            dcc.Graph(id="snapshot-graph2")
                        ])
                    ])
                ], md=6)
            ])
        ], width=9)
    ])
], fluid=True)


# === Initialize app ===
# app = dash.Dash(__name__)
#
# === Layout ===
# app.layout = html.Div(
#     [
#         html.H2("Binary Snapshot Viewer"),
#         html.Div(
#             [
#                 dbc.Label("Binary index"),
#                 dcc.Input(
#                     id="bin-input", type="number", min=0, max=len(ex_index) - 1, value=0
#                 ),
#             ]
#         ),  ##Have option to have to plot range scaled to the binary separation!!!
#         html.Div(
#             [
#                 dbc.Label("(Left plot) Set scale"),
#                 dcc.Input(id="max-sep", type="number", value=10000),
#             ]
#         ),
#         html.Div(
#             [
#                 dbc.Label("(Right plot) Set scale relative to binary separation:"),
#                 dcc.Input(id="max-sep-rel", type="number", value=5),
#             ]
#         ),
#         html.Div(
#             [
#                 dbc.Label("Include halo?"),
#                 dbc.RadioItems(
#                     id="halo_toggle",
#                     options=[
#                         {"label": "Include halo", "value": True},
#                         {"label": "Exclude halo", "value": False},
#                     ],
#                     value=True,
#                     inline=True,
#                 ),
#             ],
#             className="mb-1",
#         ),
#         html.Div(
#             [
#                 dbc.Label("Include FUTURE companions?"),
#                 dbc.RadioItems(
#                     id="comp_toggle",
#                     options=[
#                         {"label": "Include", "value": True},
#                         {"label": "Exclude", "value": False},
#                     ],
#                     value=True,
#                     inline=True,
#                 ),
#             ],
#             className="mb-1",
#         ),
#         html.Div(
#             [
#                 dbc.Label("Time:"),
#                 html.Button("←", id="step-back", n_clicks=0),
#                 dcc.Input(
#                     id="time-input",
#                     type="number",
#                     min=0,
#                     max=490,
#                     value=max(int(ex_time_max[ex_index[0]]) - ioffset, my_data["fst"][ex_index[0]]),
#                     step=1,
#                     style={"width": "80px"},
#                 ),
#                 html.Button("→", id="step-forward", n_clicks=0),
#             ],
#             style={"display": "flex", "alignItems": "center", "gap": "10px"},
#         ),
#         html.Div(
#             [dcc.Graph(id="snapshot-graph"), dcc.Graph(id="snapshot-graph2")],
#             style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
#         ),
#     ]
# )
#################################################################################


##Many arguments can be combined -- just pass the whole dataframe(!)
def plotly_snapshot(movie_dat):
    fig = go.Figure()
    size_col = mtotcol if movie_dat.halo_toggle else mcol
    size_scale = 0.0003

    paths = np.array([path_lookup[str(pp)] for pp in movie_dat.ps])
    paths_T = np.transpose(paths, axes=(1, 0, 2))


    ##mcol of mtotcol should also affect the center of mass calculation...
    coms = get_com_winf(paths)
    delta = np.array(coms)
    if np.isinf(delta[movie_dat.time, 0]):
        return
    ##If max_sep_rel is specified then
    pos_list = np.array(
        [
            paths[tmp, movie_dat.time, pxcol : pxcol + 2]
            for tmp in range(len(movie_dat.ps))
        ]
    )
    max_sep = movie_dat.max_sep
    max_sep_b = movie_dat.max_sep
    if (
        (movie_dat.max_sep_rel > 0)
        and ~(np.any(np.isnan(pos_list[:, 0])))
        and ~(np.any(np.isinf(pos_list[:, 0])))
    ):
        max_sep_b = movie_dat.max_sep_rel * max_pairwise_dist(pos_list) * unit

    ##Can refactor much of this code into its own function!!!
    names = [f"Star {sid}" for sid in movie_dat.ps]
    masses = paths_T[movie_dat.time, :, size_col]
    xs = (paths_T[movie_dat.time, :, pxcol] - delta[movie_dat.time, 0]) * unit
    ys = (paths_T[movie_dat.time, :, pycol] - delta[movie_dat.time, 1]) * unit
    zs = (paths_T[movie_dat.time, :, pzcol] - delta[movie_dat.time, 2]) * unit
    hover_texts = [
        f"{name}<br>Mass: {mass:.2f} M☉<br>x={xx:.2f} y={yy:.2f} z={zz:.2f}"
        for name, mass, xx, yy, zz, idx in zip(names, masses, xs, ys, zs, range(len(xs)))
    ]
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(
                symbol="square",
                size=3.0 * np.log10(paths_T[movie_dat.time, :, size_col] / size_scale),
                color="black",
                opacity=0.5,
            ),
            showlegend=False,
            text=hover_texts,
            hoverinfo="text",
        )
    )

    # Past/future companions (hollow)
    if len(movie_dat.comps) > 0:
        paths_extra = np.array([path_lookup[str(pp)] for pp in movie_dat.comps])
        paths_extra_T = np.transpose(paths_extra, axes=(1, 0, 2))

        names = [f"Star {sid}" for sid in movie_dat.comps]
        masses = paths_extra_T[movie_dat.time, :, size_col]
        xs = (paths_extra_T[movie_dat.time, :, pxcol] - delta[movie_dat.time, 0]) * unit
        ys = (paths_extra_T[movie_dat.time, :, pycol] - delta[movie_dat.time, 1]) * unit
        zs = (paths_extra_T[movie_dat.time, :, pzcol] - delta[movie_dat.time, 2]) * unit
        hover_texts = [
            f"{name}<br>Mass: {mass:.2f} M☉<br>x={xx:.2f} y={yy:.2f} z={zz:.2f}"
            for name, mass, xx, yy, zz in zip(names, masses, xs, ys, zs)
        ]
        # hover_texts = [f"{name}<br>Mass: {mass:.2f} M☉" for name, mass in zip(names, masses)]

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(
                    size=3.0
                    * np.log10(paths_extra_T[movie_dat.time, :, size_col] / size_scale),
                    color="red",
                    line=dict(width=3, color="red"),
                    symbol="circle-open",
                ),
                text=hover_texts,
                hoverinfo="text",
                showlegend=False,
            )
        )

    # Current companions (filled edge)
    if len(movie_dat.comps_curr) > 0:
        paths_curr = np.array([path_lookup[str(pp)] for pp in movie_dat.comps_curr])
        paths_curr_T = np.transpose(paths_curr, axes=(1, 0, 2))

        names = [f"Star {sid}" for sid in movie_dat.comps_curr]
        masses = paths_curr_T[movie_dat.time, :, size_col]
        xs = (paths_curr_T[movie_dat.time, :, pxcol] - delta[movie_dat.time, 0]) * unit
        ys = (paths_curr_T[movie_dat.time, :, pycol] - delta[movie_dat.time, 1]) * unit
        zs = (paths_curr_T[movie_dat.time, :, pzcol] - delta[movie_dat.time, 2]) * unit
        hover_texts = [
            f"{name}<br>Mass: {mass:.2f} M☉<br>x={xx:.2f} y={yy:.2f} z={zz:.2f}"
            for name, mass, xx, yy, zz in zip(names, masses, xs, ys, zs)
        ]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(
                    size=3.0
                    * np.log10(paths_curr_T[movie_dat.time, :, size_col] / size_scale),
                    color="red",
                    line=dict(color="red", width=1),
                ),
                text=hover_texts,
                hoverinfo="text",
                showlegend=False,
            )
        )

    xcenter = (coms[movie_dat.time, 0] - delta[movie_dat.time, 0]) * unit
    ycenter = (coms[movie_dat.time, 1] - delta[movie_dat.time, 1]) * unit
    zcenter = (coms[movie_dat.time, 2] - delta[movie_dat.time, 2]) * unit
    # Axes and annotation
    col_axis = "black"
    if (movie_dat.time >= movie_dat.ex_time_start) & (
        movie_dat.time <= movie_dat.ex_time_end
    ):
        col_axis = "red"
    fig.update_layout(
        height=600,
        width=600,
        title=movie_dat.annotations[movie_dat.time],
        scene=dict(
            camera=dict(eye=dict(x=0, y=0, z=2)),  # "eye" is the camera position
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(
                title="x [au]",
                range=[(xcenter - max_sep), (xcenter + max_sep)],
                color=col_axis,
                autorange=False,
            ),
            yaxis=dict(
                title="y [au]",
                range=[(ycenter - max_sep), (ycenter + max_sep)],
                color=col_axis,
                autorange=False,
            ),
            zaxis=dict(
                title="z [au]",
                range=[(zcenter - max_sep), (zcenter + max_sep)],
                color=col_axis,
                autorange=False,
            ),
        ),
    )
    fig2 = go.Figure(fig)
    fig2.update_layout(
        height=600,
        width=600,
        title=movie_dat.annotations[movie_dat.time],
        scene=dict(
            camera=dict(eye=dict(x=0, y=0, z=2)),  # "eye" is the camera position
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(
                title="x [au]",
                range=[(xcenter - max_sep_b), (xcenter + max_sep_b)],
                color=col_axis,
                autorange=False,
            ),
            yaxis=dict(
                title="y [au]",
                range=[(ycenter - max_sep_b), (ycenter + max_sep_b)],
                color=col_axis,
                autorange=False,
            ),
            zaxis=dict(
                title="z [au]",
                range=[(zcenter - max_sep_b), (zcenter + max_sep_b)],
                color=col_axis,
                autorange=False,
            ),
        ),
    )

    return fig, fig2


##Are n_back/n_forward really necessary here??
##Make sure we get only indices that we actually want here e.g. the excahnges(!)
@app.callback(
    Output("snapshot-graph", "figure"),
    Output("snapshot-graph2", "figure"),
    Output("time-input", "value"),
    Input("step-back", "n_clicks"),
    Input("step-forward", "n_clicks"),
    Input("bin-input", "value"),
    Input("time-input", "value"),
    Input("max-sep", "value"),
    Input("max-sep-rel", "value"),
    Input("halo_toggle", "value"),
    Input("comp_toggle", "value"),
    State("time-input", "value"),
)
def update_figure(
    n_back,
    n_forward,
    bin_input,
    input_value,
    max_sep,
    max_sep_rel,
    halo_toggle,
    comp_toggle,
    current_value,
):
    ##Only looking at subset of exchange binaries for now
    ##TO DO: Develop ability to look at all binaries.
    tmp_bin_idx = ex_index[bin_input]
    # tmp_bin_idx = bin_input
    my_bin = my_data["bin_ids"][tmp_bin_idx]
    # print(my_bin)
    ps = list(my_bin)
    ps[0] = int(ps[0])
    ps[1] = int(ps[1])
    bin_sel = get_bound_snaps_adjust(ps, high_df)

    last_star_snap = min(
        int(bin_sel["tval"].iloc[-1]) + 2, int(my_data["end_stars"][tmp_bin_idx])
    )
    first_star_snap = min(lookup_dict[ps[0]][0, 0], int(lookup_dict[ps[1]][0, 0]))
    ##Getting only persistent companions

    c1 = comps_dict["comps_a_ids_flat"][tmp_bin_idx]
    c2 = comps_dict["comps_b_ids_flat"][tmp_bin_idx]
    times1 = comps_dict["comps_a_times"][tmp_bin_idx]
    times2 = comps_dict["comps_b_times"][tmp_bin_idx]
    hiers1 = comps_dict["comps_a_ids"][tmp_bin_idx]
    hiers2 = comps_dict["comp_b_ids"][tmp_bin_idx]

    ##Get only quasi-persistent companions(!!!)
    comps = np.concatenate(
        (np.unique(np.concatenate(c1)), np.unique(np.concatenate(c2)))
    )
    comps = comps[(comps != ps[0]) & (comps != ps[1])]

    annotations1 = [""] * 490
    for ii, ttt in enumerate(times1):
        annotations1[int(ttt)] = hiers1[ii]

    annotations2 = [""] * 490
    for ii, ttt in enumerate(times2):
        annotations2[int(ttt)] = hiers2[ii]

    annotations = [
        f"Binary stars: {ps[0]} {ps[1]}"
        + "<br>"
        + f"Hier 1: {annotations1[ii]}"
        + "<br>"
        + f"Hier 2: {annotations2[ii]}"
        for ii in range(len(annotations1))
    ]

    ##Logic for click forward and click back
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "step-back":
        new_t = max(current_value - 1, first_star_snap)
    elif triggered_id == "step-forward":
        new_t = min(current_value + 1, last_star_snap)
    elif triggered_id == "bin-input":
        new_t = max(int(ex_time_max[tmp_bin_idx]) - ioffset, my_data["fst"][tmp_bin_idx])
    else:
        new_t = current_value

    comps_curr = []
    comps_prev_curr = []
    t_group = (times1, times2)
    for ii, cc in enumerate((c1, c2)):
        tmp_comps_curr = np.array(cc, dtype=object)[np.array(t_group[ii]) == new_t]
        tmp_comps_prev_curr = np.array(cc, dtype=object)[np.array(t_group[ii]) <= new_t]
        if len(tmp_comps_curr) > 0:
            comps_curr.append(tmp_comps_curr[0])
            comps_prev_curr.append(tmp_comps_prev_curr[0])

    if len(comps_curr) > 0:
        comps_curr = np.concatenate(comps_curr)
        comps_curr = np.unique(
            comps_curr[(comps_curr != ps[0]) & (comps_curr != ps[1])]
        )
        ##Current and previous companions
        comps_prev_curr = np.concatenate(comps_prev_curr)
        comps_prev_curr = np.unique(
            comps_prev_curr[(comps_prev_curr != ps[0]) & (comps_prev_curr != ps[1])]
        )

    if not comp_toggle:
        comps = comps_prev_curr

    movie_dat = MovieData()
    movie_dat.ps = ps
    movie_dat.time = new_t
    movie_dat.comps = comps
    movie_dat.comps_curr = comps_curr
    movie_dat.halo_toggle = halo_toggle
    movie_dat.max_sep = max_sep
    movie_dat.max_sep_rel = max_sep_rel
    movie_dat.ex_time_start, movie_dat.ex_time_end = (
        ex_time_max[tmp_bin_idx],
        ex_time_max_end[tmp_bin_idx],
    )
    movie_dat.annotations = annotations

    ##Need to pass the time interval over which exchange occurs...
    fig, fig2 = plotly_snapshot(movie_dat)
    return fig, fig2, new_t


if __name__ == "__main__":
    app.run(debug=True, port=8051)
