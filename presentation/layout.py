"""
Sambaza-Sim — Dashboard Layout
================================
Dark, 3-panel workspace:
  Left  : Accordion configuration panel (Scenario / Tax Policy / Spending / Advanced / Data Source)
  Center: KPI cards + tabbed results (Overview | Sector Detail | Matrix Explorer)
  Right : Tech-change builder
"""

import os
import sys
from dash import Dash, html, dcc
import dash_ag_grid as dag

# Ensure root path is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from simulators.tech_change_loader import rebuild_examples_dict_from_db
from pipeline.db.repositories.good_repo import GoodsDatabase

# Database URL
db_path = os.path.join(root_dir, 'data.db')
db_url  = f"sqlite:///{db_path}"

# ── DB helpers (called fresh on each page load) ────────────────────────────
def get_examples_config():
    try:
        return rebuild_examples_dict_from_db(db_url)
    except Exception as e:
        print(f"Error loading examples: {e}")
        return {}

def get_sector_options():
    try:
        gdb = GoodsDatabase(database_url=db_url)
        return [{'label': f"{g.name}  ({g.isic})", 'value': g.isic} for g in gdb.get_all_goods()]
    except Exception as e:
        print(f"Error loading sector options: {e}")
        return []


# ── Design tokens ──────────────────────────────────────────────────────────
BG      = '#0f1117'
PANEL   = '#13151f'
CARD    = '#1e2130'
BORDER  = '#272b3d'
ACCENT  = '#4f8ef7'
TEXT    = '#e2e8f0'
MUTED   = '#64748b'
GREEN   = '#22c55e'
RED     = '#ef4444'
WARN    = '#f59e0b'
FONT    = 'Inter, system-ui, -apple-system, sans-serif'

# ── Shared styles ──────────────────────────────────────────────────────────
_LABEL = {'fontSize': '11px', 'fontWeight': '600', 'letterSpacing': '0.04em',
          'color': MUTED, 'marginBottom': '5px', 'display': 'block'}

_DD_STYLE = {
    'backgroundColor': CARD,
    'borderColor': BORDER,
    'color': TEXT,
    'marginBottom': '10px',
}

_GRID_OPTS = {'className': 'ag-theme-alpine-dark'}


# ── Component helpers ──────────────────────────────────────────────────────

def _label(text):
    return html.Label(text, style=_LABEL)


def _slider(sid, value=0.0, min_val=0.0, max_val=1.0, step=0.01, marks=None):
    return dcc.Slider(
        id=sid, min=min_val, max=max_val, step=step, value=value,
        marks=marks or {},
        tooltip={'placement': 'bottom', 'always_visible': True},
        className='sim-slider',
    )


def _slider_row(label_text, sid, **kw):
    return html.Div(style={'marginBottom': '16px'}, children=[
        _label(label_text),
        _slider(sid, **kw),
    ])


def _accordion(title, content, open_by_default=False):
    """Native HTML <details>/<summary> accordion with dark styling."""
    return html.Details(
        open=open_by_default,
        style={'borderTop': f'1px solid {BORDER}'},
        children=[
            html.Summary(title, style={
                'padding': '11px 0',
                'cursor': 'pointer',
                'fontSize': '10px',
                'fontWeight': '700',
                'letterSpacing': '0.1em',
                'textTransform': 'uppercase',
                'color': MUTED,
                'userSelect': 'none',
                'outline': 'none',
            }),
            html.Div(content, style={'paddingBottom': '12px'}),
        ]
    )


def _kpi_card(card_id, label):
    return html.Div(
        style={
            'flex': '1 1 150px',
            'backgroundColor': CARD,
            'border': f'1px solid {BORDER}',
            'borderRadius': '10px',
            'padding': '16px 18px',
        },
        children=[
            html.P(label, style={
                'margin': '0 0 8px', 'fontSize': '10px', 'fontWeight': '700',
                'letterSpacing': '0.08em', 'textTransform': 'uppercase', 'color': MUTED,
            }),
            html.P('—', id=card_id, style={
                'margin': 0, 'fontSize': '24px', 'fontWeight': '700', 'color': TEXT,
                'lineHeight': '1',
            }),
        ]
    )


def _graph_card(graph_id, height=300):
    return html.Div(
        style={
            'flex': '1 1 340px',
            'backgroundColor': CARD,
            'border': f'1px solid {BORDER}',
            'borderRadius': '10px',
            'padding': '4px 4px 0',
            'overflow': 'hidden',
        },
        children=[
            dcc.Graph(
                id=graph_id,
                style={'height': f'{height}px'},
            )
        ]
    )


# ── App initialization ─────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap',
    ],
    suppress_callback_exceptions=True,
)
app.title = 'Sambaza-Sim'

# Global CSS injected into the page
app.index_string = r'''
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; overflow: hidden; }
  body { background: #0f1117; color: #e2e8f0; font-family: Inter, system-ui, sans-serif; }

  /* Dash 4.0 Theme Tokens */
  :root {
      --Dash-Fill-Disabled: #1e293b;            
      --Dash-Fill-Interactive-Strong: #4f8ef7; 
      --Dash-Fill-Interactive-Weak: rgba(79, 142, 247, 0.12);  
      --Dash-Fill-Inverse-Strong: #1e2130;      
      --Dash-Fill-Primary-Active: #252836;      
      --Dash-Fill-Primary-Hover: #2d3a5a;       
      --Dash-Shading-Strong: rgba(0, 0, 0, 0.4);
      --Dash-Shading-Weak: rgba(0, 0, 0, 0.2);

      --Dash-Stroke-Strong: #272b3d;
      --Dash-Stroke-Weak: rgba(255, 255, 255, 0.06);

      --Dash-Text-Disabled: #475569;
      --Dash-Text-Primary: #e2e8f0;
      --Dash-Text-Strong: #ffffff;
      --Dash-Text-Weak: #64748b;

      --Dash-Tooltip-Background-Color: #4f8ef7;
      --Dash-Tooltip-Border-Color: rgba(255, 255, 255, 0.1);
  }

  /* Catch portaled dropdown content for Dash 4.0 */
  [data-radix-popper-content-wrapper],
  [role="listbox"],
  [role="option"],
  [role="combobox"],
  [data-state],
  [data-highlighted] {
      background-color: #1e2130 !important;
      color: #e2e8f0 !important;
      border-color: #272b3d !important;
  }

  div[style*="background-color: white"],
  div[style*="background: white"],
  div[style*="background-color: rgb(255"] {
      background-color: #1e2130 !important;
  }

  /* AG Grid Dark Mode Overrides */
  .ag-theme-alpine-dark {
      --ag-background-color: #1e2130 !important;
      --ag-header-background-color: #13151f !important;
      --ag-odd-row-background-color: rgba(255, 255, 255, 0.02) !important;
      --ag-row-hover-color: rgba(79, 142, 247, 0.08) !important;
      --ag-border-color: #272b3d !important;
      --ag-header-foreground-color: #64748b !important;
      --ag-foreground-color: #e2e8f0 !important;
      --ag-font-family: Inter, system-ui, sans-serif !important;
  }

  /* Plotly Dark Mode Overrides */
  .js-plotly-plot .plotly > .main-svg:first-child {
      border-radius: 10px;
  }
  .js-plotly-plot .plotly .bg,
  .js-plotly-plot .plotly rect.bg {
      fill: #1e2130 !important;
  }
  .js-plotly-plot .plotly .xtick text,
  .js-plotly-plot .plotly .ytick text { fill: #64748b !important; }
  .js-plotly-plot .plotly .g-xtitle text,
  .js-plotly-plot .plotly .g-ytitle text,
  .js-plotly-plot .plotly .gtitle text { fill: #e2e8f0 !important; }
  .js-plotly-plot .plotly .legend text { fill: #64748b !important; }
  .js-plotly-plot .plotly .legend rect { fill: #1e2130 !important; stroke: #272b3d !important; }
  .js-plotly-plot .plotly .cbtitle text,
  .js-plotly-plot .plotly .cbtick text { fill: #64748b !important; }
  .js-plotly-plot .plotly .hoverlayer .hovertext rect { fill: #0f1117 !important; }
  .js-plotly-plot .plotly .hoverlayer .hovertext text { fill: #e2e8f0 !important; }
  .js-plotly-plot .plotly .gridlayer line { stroke: #272b3d !important; }
  .js-plotly-plot .plotly .modebar-btn { fill: #64748b !important; color: #64748b !important; }

  /* Scrollbars */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2d3142; border-radius: 3px; }

  /* Accordion marker */
  details > summary { list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::after { content: '▸'; float: right; transition: transform 0.18s; }
  details[open] > summary::after { transform: rotate(90deg); }

  /* Inputs */
  input[type=number] {
    background: #1e2130; border: 1px solid #272b3d; color: #e2e8f0;
    border-radius: 6px; padding: 7px 10px; outline: none; font-family: inherit; font-size: 13px;
  }
  input[type=number]:focus { border-color: #4f8ef7; box-shadow: 0 0 0 2px rgba(79,142,247,0.2); }

  /* Dash Tabs */
  .dash-tabs .tab { background: transparent !important; color: #64748b !important; border: none !important; border-bottom: 2px solid transparent !important; padding: 10px 18px !important; font-size: 13px !important; font-weight: 500 !important; }
  .dash-tabs .tab--selected { color: #4f8ef7 !important; border-bottom-color: #4f8ef7 !important; }
  .dash-tabs .tab-container { background: transparent !important; border-bottom: 1px solid #272b3d !important; }

  /* AgGrid dark tweaks */
  .ag-theme-alpine-dark {
    --ag-background-color: #1e2130;
    --ag-header-background-color: #13151f;
    --ag-odd-row-background-color: #1a1d2e;
    --ag-border-color: #272b3d;
    --ag-row-hover-color: #252938;
    --ag-header-foreground-color: #94a3b8;
    --ag-foreground-color: #cbd5e1;
    --ag-row-border-color: #272b3d;
    --ag-font-family: Inter, system-ui, sans-serif;
    --ag-font-size: 12px;
  }

  /* Buttons */
  button:hover { filter: brightness(1.12); }
  button:active { filter: brightness(0.95); }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
'''


# ── Layout function ────────────────────────────────────────────────────────
def serve_layout():
    examples_config = get_examples_config()
    example_options = [{'label': v['title'], 'value': k} for k, v in examples_config.items()]
    default_example = list(examples_config.keys())[0] if examples_config else None
    sector_options  = get_sector_options()

    # Discover valid data folders
    data_dir = os.path.join(root_dir, 'data')
    data_sources = []
    if os.path.exists(data_dir):
        for entry in os.listdir(data_dir):
            ep = os.path.join(data_dir, entry)
            if (os.path.isdir(ep)
                    and os.path.exists(os.path.join(ep, 'goods.csv'))
                    and os.path.exists(os.path.join(ep, 'productions.csv'))):
                data_sources.append({'label': f"data/{entry}", 'value': f"data/{entry}"})

    # ── LEFT PANEL ─────────────────────────────────────────────────────────
    left_panel = html.Div(
        style={
            'width': '290px', 'minWidth': '290px',
            'backgroundColor': PANEL,
            'borderRight': f'1px solid {BORDER}',
            'padding': '14px 14px 14px',
            'overflowY': 'auto',
            'display': 'flex', 'flexDirection': 'column',
        },
        children=[
            html.P('Configure', style={
                'fontSize': '10px', 'fontWeight': '700', 'letterSpacing': '0.12em',
                'textTransform': 'uppercase', 'color': MUTED, 'marginBottom': '6px',
            }),

            # ── Data Source accordion ───────────────────────────────────
            _accordion('Data Source', open_by_default=True, content=[
                _label('Source Folder'),
                dcc.Dropdown(
                    id='data-source-selector',
                    options=data_sources,
                    placeholder='Select folder in data/…',
                    style=_DD_STYLE,
                ),
                html.Button(
                    'Load Data Source', id='load-source-btn', n_clicks=0,
                    style={
                        'width': '100%', 'padding': '8px',
                        'backgroundColor': WARN, 'color': 'white',
                        'border': 'none', 'borderRadius': '6px',
                        'cursor': 'pointer', 'fontSize': '12px', 'fontWeight': '600',
                    }
                ),
                html.Div(id='load-source-status',
                         style={'fontSize': '11px', 'color': RED, 'marginTop': '6px'}),
            ]),

            # ── Scenario accordion ──────────────────────────────────────
            _accordion('Scenario', content=[
                _label('Dataset / Example'),
                dcc.Dropdown(
                    id='example-selector',
                    options=example_options, value=default_example,
                    clearable=False, style=_DD_STYLE,
                ),
                html.Div(
                    id='example-description',
                    style={'fontSize': '11px', 'color': MUTED, 'marginBottom': '10px', 'lineHeight': '1.5'},
                ),
                _label('Total Final Demand ($)'),
                dcc.Input(
                    id='input-total-fd', type='number', min=0, step=1.0,
                    placeholder='Override…',
                    style={'width': '100%', 'marginBottom': '4px'},
                ),
                html.Div(id='input-total-fd-hint',
                         style={'fontSize': '10px', 'color': MUTED, 'marginBottom': '6px'}),
            ]),

            # ── Tax Policy accordion ────────────────────────────────────
            _accordion('Tax Policy', content=[
                html.Div(
                    style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '14px'},
                    children=[
                        # BEFORE column
                        html.Div([
                            html.P('BEFORE', style={
                                'fontSize': '9px', 'fontWeight': '700', 'letterSpacing': '0.1em',
                                'color': MUTED, 'marginBottom': '10px',
                            }),
                            _label('Income Tax'),
                            _slider('input-income-tax-before', value=0.0),
                            html.Div(style={'height': '14px'}),
                            _label('Corp Tax'),
                            _slider('input-corp-tax-before', value=0.0),
                        ]),
                        # AFTER column
                        html.Div([
                            html.P('AFTER', style={
                                'fontSize': '9px', 'fontWeight': '700', 'letterSpacing': '0.1em',
                                'color': ACCENT, 'marginBottom': '10px',
                            }),
                            _label('Income Tax'),
                            _slider('input-income-tax-after', value=0.0),
                            html.Div(style={'height': '14px'}),
                            _label('Corp Tax'),
                            _slider('input-corp-tax-after', value=0.0),
                        ]),
                    ]
                ),
            ]),

            # ── Spending Rates accordion ────────────────────────────────
            _accordion('Spending Rates', content=[
                _slider_row('Wage Spend Rate',       'input-wage-spend',    value=1.0),
                _slider_row('Surplus Spend Rate',    'input-surplus-spend', value=1.0),
                _slider_row('Government Spend Rate', 'input-tax-spend',     value=1.0),
            ]),

            # ── Advanced accordion ──────────────────────────────────────
            _accordion('Advanced', content=[
                _label('Iterations'),
                dcc.Slider(
                    id='input-iterations', min=1, max=20, step=1, value=5,
                    marks={i: {'label': str(i), 'style': {'color': MUTED, 'fontSize': '10px'}}
                           for i in [1, 5, 10, 15, 20]},
                    tooltip={'placement': 'bottom', 'always_visible': False},
                ),
                html.Div(style={'height': '14px'}),
                _label('Solver Type'),
                dcc.Dropdown(
                    id='input-solver',
                    options=[
                        {'label': 'Supply Curves', 'value': 'supply_curves'},
                        {'label': 'Leontief Inverse', 'value': 'leontief'},
                    ],
                    value='supply_curves', clearable=False, style=_DD_STYLE,
                ),
                _label('Economy Type'),
                dcc.Dropdown(
                    id='input-economy-type',
                    options=[
                        {'label': 'Open  (Imports Leak)', 'value': 'open'},
                        {'label': 'Closed  (Imports Recycled)', 'value': 'closed'},
                    ],
                    value='open', clearable=False, style=_DD_STYLE,
                ),
            ]),


        ]
    )

    # ── CENTER PANEL ───────────────────────────────────────────────────────
    kpi_row = html.Div(
        style={'display': 'flex', 'gap': '10px', 'marginBottom': '16px', 'flexWrap': 'wrap'},
        children=[
            _kpi_card('summary-output', 'Δ Total Output'),
            _kpi_card('summary-va',     'Δ Value Added'),
            _kpi_card('summary-fd',     'Δ Final Demand'),
            _kpi_card('summary-tax',    'Δ Tax Revenue'),
        ]
    )

    center_panel = html.Div(
        style={
            'flex': '1 1 0', 'minWidth': 0,
            'padding': '16px 18px',
            'overflowY': 'auto',
        },
        children=[
            kpi_row,
            dcc.Tabs(
                id='main-tabs', value='tab-overview',
                children=[
                    # ── Overview ─────────────────────────────────────────
                    dcc.Tab(label='Overview', value='tab-overview', children=[
                        html.Div(
                            style={'display': 'flex', 'gap': '12px', 'marginTop': '14px', 'flexWrap': 'wrap'},
                            children=[
                                _graph_card('graph-waterfall', height=310),
                                _graph_card('graph-delta-bar',  height=310),
                            ]
                        ),
                        html.Div(style={'marginTop': '14px'}, children=[
                            dag.AgGrid(
                                id='table-results',
                                columnDefs=[
                                    {'field': 'Sector', 'pinned': 'left', 'width': 190, 'headerName': 'Sector'},
                                    {'field': 'Output_Before', 'headerName': 'Output (Before)',
                                     'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}, 'flex': 1},
                                    {'field': 'Output_After',  'headerName': 'Output (After)',
                                     'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}, 'flex': 1},
                                    {'field': 'Output_Delta',  'headerName': 'Δ Output',
                                     'valueFormatter': {"function": "d3.format(',.2f')(params.value)"},
                                     'cellStyle': {"function": "params.value < 0 ? {'color':'#ef4444','fontWeight':'600'} : params.value > 0 ? {'color':'#22c55e','fontWeight':'600'} : {}"}, 'flex': 1},
                                    {'field': 'VA_Before', 'headerName': 'VA (Before)',
                                     'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}, 'flex': 1},
                                    {'field': 'VA_After',  'headerName': 'VA (After)',
                                     'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}, 'flex': 1},
                                    {'field': 'VA_Delta',  'headerName': 'Δ VA',
                                     'valueFormatter': {"function": "d3.format(',.2f')(params.value)"},
                                     'cellStyle': {"function": "params.value < 0 ? {'color':'#ef4444','fontWeight':'600'} : params.value > 0 ? {'color':'#22c55e','fontWeight':'600'} : {}"}, 'flex': 1},
                                    {'field': 'FD_Before', 'headerName': 'FD (Before)',
                                     'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}, 'flex': 1},
                                    {'field': 'FD_After',  'headerName': 'FD (After)',
                                     'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}, 'flex': 1},
                                    {'field': 'FD_Delta',  'headerName': 'Δ FD',
                                     'valueFormatter': {"function": "d3.format(',.2f')(params.value)"},
                                     'cellStyle': {"function": "params.value < 0 ? {'color':'#ef4444','fontWeight':'600'} : params.value > 0 ? {'color':'#22c55e','fontWeight':'600'} : {}"}, 'flex': 1},
                                ],
                                rowData=[],
                                defaultColDef={'sortable': True, 'filter': True, 'resizable': True},
                                dashGridOptions={'pagination': True, 'paginationPageSize': 20},
                                **_GRID_OPTS,
                                style={'height': 480, 'width': '100%'},
                            )
                        ])
                    ]),

                    # ── Matrix Explorer ───────────────────────────────────
                    dcc.Tab(label='Matrix Explorer', value='tab-matrices', children=[
                        html.Div(style={'marginTop': '14px', 'display': 'flex', 'flexDirection': 'column', 'gap': '12px'}, children=[

                            # Demand vector + VA coefficients row
                            html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap'}, children=[
                                html.Div(
                                    style={'flex': '1 1 250px', 'backgroundColor': CARD,
                                           'borderRadius': '10px', 'border': f'1px solid {BORDER}', 'padding': '14px'},
                                    children=[
                                        html.P('Final Demand Vector',
                                               style={'fontWeight': '600', 'marginBottom': '10px', 'fontSize': '13px', 'color': TEXT}),
                                        dag.AgGrid(
                                            id='table-demand-vector',
                                            columnDefs=[
                                                {'field': 'Sector', 'width': 170},
                                                {'field': 'Demand', 'headerName': 'Final Demand ($)', 'flex': 1,
                                                 'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}},
                                            ],
                                            rowData=[],
                                            defaultColDef={'resizable': True},
                                            **_GRID_OPTS,
                                            style={'height': 280, 'width': '100%'},
                                        )
                                    ]
                                ),
                                html.Div(
                                    style={'flex': '2 1 380px', 'backgroundColor': CARD,
                                           'borderRadius': '10px', 'border': f'1px solid {BORDER}', 'padding': '14px'},
                                    children=[
                                        html.P('Value Added Coefficients',
                                               style={'fontWeight': '600', 'marginBottom': '10px', 'fontSize': '13px', 'color': TEXT}),
                                        dag.AgGrid(
                                            id='table-va-coefficients',
                                            columnDefs=[
                                                {'field': 'Sector', 'width': 170},
                                                {'field': 'VA_Before', 'headerName': 'VA Coeff (Before)', 'flex': 1,
                                                 'valueFormatter': {"function": "d3.format(',.4f')(params.value)"}},
                                                {'field': 'VA_After',  'headerName': 'VA Coeff (After)',  'flex': 1,
                                                 'valueFormatter': {"function": "d3.format(',.4f')(params.value)"}},
                                                {'field': 'VA_Delta',  'headerName': 'Δ VA Coeff', 'flex': 1,
                                                 'valueFormatter': {"function": "d3.format(',.4f')(params.value)"},
                                                 'cellStyle': {"function": "params.value < 0 ? {'color':'#ef4444'} : params.value > 0 ? {'color':'#22c55e'} : {}"}},
                                            ],
                                            rowData=[],
                                            defaultColDef={'resizable': True},
                                            **_GRID_OPTS,
                                            style={'height': 280, 'width': '100%'},
                                        )
                                    ]
                                ),
                            ]),

                            # Matrix heatmap card
                            html.Div(
                                style={'backgroundColor': CARD, 'borderRadius': '10px',
                                       'border': f'1px solid {BORDER}', 'padding': '14px'},
                                children=[
                                    html.Div(
                                        style={'display': 'flex', 'alignItems': 'center', 'gap': '14px', 'marginBottom': '12px', 'flexWrap': 'wrap'},
                                        children=[
                                            dcc.Tabs(
                                                id='matrix-type-selector',
                                                value='delta_A',
                                                style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '0px', 'border': 'none', 'padding': '10px 0'},
                                                children=[
                                                    dcc.Tab(label='B Tech Coeffs', value='A_before', 
                                                            style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': PANEL, 'border': f'1px solid {BORDER}', 'borderRadius': '16px', 'color': MUTED, 'fontSize': '11px', 'fontWeight': '500', 'margin': '4px', 'whiteSpace': 'nowrap'}, 
                                                            selected_style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': 'rgba(79, 142, 247, 0.12)', 'border': f'1px solid {ACCENT}', 'borderRadius': '16px', 'color': ACCENT, 'fontSize': '11px', 'fontWeight': '600', 'margin': '4px', 'whiteSpace': 'nowrap'}),
                                                    dcc.Tab(label='A Tech Coeffs',  value='A_after', 
                                                            style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': PANEL, 'border': f'1px solid {BORDER}', 'borderRadius': '16px', 'color': MUTED, 'fontSize': '11px', 'fontWeight': '500', 'margin': '4px', 'whiteSpace': 'nowrap'}, 
                                                            selected_style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': 'rgba(79, 142, 247, 0.12)', 'border': f'1px solid {ACCENT}', 'borderRadius': '16px', 'color': ACCENT, 'fontSize': '11px', 'fontWeight': '600', 'margin': '4px', 'whiteSpace': 'nowrap'}),
                                                    dcc.Tab(label='Δ Tech Coeffs',         value='delta_A', 
                                                            style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': PANEL, 'border': f'1px solid {BORDER}', 'borderRadius': '16px', 'color': MUTED, 'fontSize': '11px', 'fontWeight': '500', 'margin': '4px', 'whiteSpace': 'nowrap'}, 
                                                            selected_style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': 'rgba(79, 142, 247, 0.12)', 'border': f'1px solid {ACCENT}', 'borderRadius': '16px', 'color': ACCENT, 'fontSize': '11px', 'fontWeight': '600', 'margin': '4px', 'whiteSpace': 'nowrap'}),
                                                    dcc.Tab(label='B Leontief', value='L_before', 
                                                            style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': PANEL, 'border': f'1px solid {BORDER}', 'borderRadius': '16px', 'color': MUTED, 'fontSize': '11px', 'fontWeight': '500', 'margin': '4px', 'whiteSpace': 'nowrap'}, 
                                                            selected_style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': 'rgba(79, 142, 247, 0.12)', 'border': f'1px solid {ACCENT}', 'borderRadius': '16px', 'color': ACCENT, 'fontSize': '11px', 'fontWeight': '600', 'margin': '4px', 'whiteSpace': 'nowrap'}),
                                                    dcc.Tab(label='A Leontief',  value='L_after', 
                                                            style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': PANEL, 'border': f'1px solid {BORDER}', 'borderRadius': '16px', 'color': MUTED, 'fontSize': '11px', 'fontWeight': '500', 'margin': '4px', 'whiteSpace': 'nowrap'}, 
                                                            selected_style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': 'rgba(79, 142, 247, 0.12)', 'border': f'1px solid {ACCENT}', 'borderRadius': '16px', 'color': ACCENT, 'fontSize': '11px', 'fontWeight': '600', 'margin': '4px', 'whiteSpace': 'nowrap'}),
                                                    dcc.Tab(label='Δ Leontief',         value='delta_L', 
                                                            style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': PANEL, 'border': f'1px solid {BORDER}', 'borderRadius': '16px', 'color': MUTED, 'fontSize': '11px', 'fontWeight': '500', 'margin': '4px', 'whiteSpace': 'nowrap'}, 
                                                            selected_style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': 'rgba(79, 142, 247, 0.12)', 'border': f'1px solid {ACCENT}', 'borderRadius': '16px', 'color': ACCENT, 'fontSize': '11px', 'fontWeight': '600', 'margin': '4px', 'whiteSpace': 'nowrap'}),
                                                    dcc.Tab(label='B Flow', value='Z_before', 
                                                            style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': PANEL, 'border': f'1px solid {BORDER}', 'borderRadius': '16px', 'color': MUTED, 'fontSize': '11px', 'fontWeight': '500', 'margin': '4px', 'whiteSpace': 'nowrap'}, 
                                                            selected_style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': 'rgba(79, 142, 247, 0.12)', 'border': f'1px solid {ACCENT}', 'borderRadius': '16px', 'color': ACCENT, 'fontSize': '11px', 'fontWeight': '600', 'margin': '4px', 'whiteSpace': 'nowrap'}),
                                                    dcc.Tab(label='A Flow',  value='Z_after', 
                                                            style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': PANEL, 'border': f'1px solid {BORDER}', 'borderRadius': '16px', 'color': MUTED, 'fontSize': '11px', 'fontWeight': '500', 'margin': '4px', 'whiteSpace': 'nowrap'}, 
                                                            selected_style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': 'rgba(79, 142, 247, 0.12)', 'border': f'1px solid {ACCENT}', 'borderRadius': '16px', 'color': ACCENT, 'fontSize': '11px', 'fontWeight': '600', 'margin': '4px', 'whiteSpace': 'nowrap'}),
                                                    dcc.Tab(label='Δ Flow',         value='delta_Z', 
                                                            style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': PANEL, 'border': f'1px solid {BORDER}', 'borderRadius': '16px', 'color': MUTED, 'fontSize': '11px', 'fontWeight': '500', 'margin': '4px', 'whiteSpace': 'nowrap'}, 
                                                            selected_style={'minWidth': 'max-content', 'width': 'auto', 'padding': '6px 12px', 'backgroundColor': 'rgba(79, 142, 247, 0.12)', 'border': f'1px solid {ACCENT}', 'borderRadius': '16px', 'color': ACCENT, 'fontSize': '11px', 'fontWeight': '600', 'margin': '4px', 'whiteSpace': 'nowrap'}),
                                                ]
                                            ),
                                        ]
                                    ),
                                    dcc.Graph(id='graph-matrix-heatmap',
                                              style={'marginBottom': '12px'}),
                                    html.P(id='matrix-table-title',
                                           style={'fontWeight': '600', 'color': TEXT, 'marginBottom': '8px', 'fontSize': '13px'}),
                                    dag.AgGrid(
                                        id='table-matrix',
                                        columnDefs=[], rowData=[],
                                        defaultColDef={'sortable': False, 'filter': False, 'resizable': True, 'width': 100},
                                        **_GRID_OPTS,
                                        style={'height': 380, 'width': '100%'},
                                    ),
                                ]
                            ),
                        ])
                    ]),
                ]
            ),
        ]
    )

    # ── RIGHT PANEL (Tech Changes) ─────────────────────────────────────────
    right_panel = html.Div(
        style={
            'width': '280px', 'minWidth': '280px',
            'backgroundColor': PANEL,
            'borderLeft': f'1px solid {BORDER}',
            'padding': '14px',
            'overflowY': 'auto',
            'display': 'flex', 'flexDirection': 'column', 'gap': '10px',
        },
        children=[
            html.P('Tech Changes', style={
                'fontSize': '10px', 'fontWeight': '700', 'letterSpacing': '0.12em',
                'textTransform': 'uppercase', 'color': MUTED,
            }),
            html.P(
                'Modify the supply structure. Changes are applied to the "After" simulation only.',
                style={'fontSize': '11px', 'color': MUTED, 'lineHeight': '1.5'},
            ),

            # Change type
            html.Div([
                _label('Change Type'),
                dcc.Dropdown(
                    id='tc-change-type',
                    options=[
                        {'label': 'All inputs of a sector',             'value': 'add_sector_change'},
                        {'label': 'Usage of an input (all sectors)',    'value': 'add_input_change'},
                        {'label': 'Specific coefficient [row → col]',  'value': 'add_coefficient_change'},
                    ],
                    value='add_input_change',
                    clearable=False, style=_DD_STYLE,
                ),
            ]),

            # Sector selector (col)
            html.Div(id='tc-sector-container', children=[
                _label('Sector (column / producer)'),
                dcc.Dropdown(
                    id='tc-sector',
                    options=sector_options,
                    placeholder='Select sector…',
                    style=_DD_STYLE,
                ),
            ]),

            # Input sector selector (row)
            html.Div(id='tc-input-sector-container', children=[
                _label('Input Sector (row)'),
                dcc.Dropdown(
                    id='tc-input-sector',
                    options=sector_options,
                    placeholder='Select input sector…',
                    style=_DD_STYLE,
                ),
            ]),

            # Operation
            html.Div([
                _label('Operation'),
                dcc.Dropdown(
                    id='tc-operation',
                    options=[
                        {'label': 'Multiply  (0.8 = −20%)', 'value': 'multiply'},
                        {'label': 'Add  (−0.05)',            'value': 'add'},
                        {'label': 'Set  (absolute value)',   'value': 'set'},
                    ],
                    value='multiply',
                    clearable=False, style=_DD_STYLE,
                ),
            ]),

            # Value input
            html.Div([
                _label('Value'),
                dcc.Input(
                    id='tc-value', type='number', value=0.8, step=0.01,
                    style={'width': '100%'},
                ),
            ]),

            # Action buttons
            html.Div(style={'display': 'flex', 'gap': '8px', 'marginTop': '4px'}, children=[
                html.Button(
                    '+ Add Change', id='tc-add-button', n_clicks=0,
                    style={
                        'flex': 1, 'padding': '9px 0',
                        'backgroundColor': ACCENT, 'color': 'white',
                        'border': 'none', 'borderRadius': '6px',
                        'fontWeight': '700', 'cursor': 'pointer', 'fontSize': '12px',
                    }
                ),
                html.Button(
                    'Clear', id='tc-clear-button', n_clicks=0,
                    style={
                        'flex': '0 0 auto', 'padding': '9px 14px',
                        'backgroundColor': '#272b3d', 'color': MUTED,
                        'border': f'1px solid {BORDER}', 'borderRadius': '6px',
                        'cursor': 'pointer', 'fontSize': '12px',
                    }
                ),
            ]),

            # Pending changes list
            html.Div(
                style={'borderTop': f'1px solid {BORDER}', 'paddingTop': '12px', 'marginTop': '4px'},
                children=[
                    html.P('Pending Changes', style={
                        'fontSize': '10px', 'fontWeight': '700', 'letterSpacing': '0.08em',
                        'textTransform': 'uppercase', 'color': MUTED, 'marginBottom': '8px',
                    }),
                    html.Div(id='tc-changes-display'),
                ]
            ),
        ]
    )

    # ── HEADER ────────────────────────────────────────────────────────────
    header = html.Div(
        style={
            'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
            'padding': '0 18px',
            'height': '52px',
            'backgroundColor': PANEL,
            'borderBottom': f'1px solid {BORDER}',
            'flexShrink': 0,
        },
        children=[
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[
                html.Span('⬡', style={'fontSize': '20px', 'color': ACCENT}),
                html.Span('Sambaza-Sim', style={'fontWeight': '700', 'fontSize': '15px', 'color': TEXT}),
                html.Span('·', style={'color': BORDER, 'fontSize': '16px'}),
                html.Span('Input-Output Model', style={'fontSize': '12px', 'color': MUTED}),
            ]),
            html.Button(
                '▶  Run Simulation', id='run-button', n_clicks=0,
                style={
                    'backgroundColor': ACCENT, 'color': 'white',
                    'border': 'none', 'borderRadius': '7px',
                    'padding': '8px 20px',
                    'fontWeight': '700', 'fontSize': '13px',
                    'cursor': 'pointer', 'letterSpacing': '0.02em',
                }
            ),
        ]
    )

    # ── ROOT ──────────────────────────────────────────────────────────────
    return html.Div(
        style={
            'display': 'flex', 'flexDirection': 'column',
            'height': '100vh', 'overflow': 'hidden',
            'backgroundColor': BG, 'fontFamily': FONT,
        },
        children=[
            dcc.Location(id='url', refresh=True),
            dcc.Store(id='store-matrices'),
            dcc.Store(id='store-tech-changes', data=[]),
            header,
            html.Div(
                style={'display': 'flex', 'flex': '1 1 0', 'overflow': 'hidden'},
                children=[left_panel, center_panel, right_panel],
            ),
        ]
    )


app.layout = serve_layout
