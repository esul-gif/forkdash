import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load the data
df = pd.read_csv('accel_data.csv')  # Adjust the path as necessary

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m-%d %H:%M:%S')

# Calculate magnitude of acceleration
df['magnitude'] = np.sqrt(df['accX']**2 + df['accY']**2 + df['accZ']**2)

# Initialize the Dash app
app = dash.Dash(__name__)

app.css.config.serve_locally = True

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Forklift Accelerometer Data"),

    dcc.Graph(id='acceleration-time-series'),

    html.Div([
        dcc.Graph(id='velocity-time-series'),
        dcc.Graph(id='displacement-time-series')
    ]),

    dcc.Graph(id='magnitude-time-series'),

    html.Label('Select Axis:'),
    dcc.Dropdown(
        id='axis-selector',
        options=[
            {'label': 'X Axis', 'value': 'accX'},
            {'label': 'Y Axis', 'value': 'accY'},
            {'label': 'Z Axis', 'value': 'accZ'}
        ],
        value='accX'
    ),

    dcc.Graph(id='acceleration-histogram'),

    html.Div([
        html.Label('Select Axes for Scatter Plot:'),
        dcc.Dropdown(
            id='scatter-axes-selector',
            options=[
                {'label': 'X vs Y', 'value': 'accX-accY'},
                {'label': 'X vs Z', 'value': 'accX-accZ'},
                {'label': 'Y vs Z', 'value': 'accY-accZ'}
            ],
            value='accX-accY'
        )
    ]),

    dcc.Graph(id='acceleration-scatter'),

    dcc.Graph(id='3d-scatter-plot'),

    dcc.Graph(id='heatmap'),

    dcc.Graph(id='temperature-line-chart'),  # New line chart for temperature
    dcc.Graph(id='brightness-line-chart'),  # New line chart for brightness

    html.Label('Select Time Range:'),
    dcc.DatePickerRange(
        id='date-picker',
        start_date=df['timestamp'].min(),
        end_date=df['timestamp'].max()
    ),

    html.H2("Summary Statistics"),
    html.Div(id='summary-stats')
])

# Callback to update the combined time series plot
@app.callback(
    Output('acceleration-time-series', 'figure'),
    Output('velocity-time-series', 'figure'),
    Output('displacement-time-series', 'figure'),
    Output('magnitude-time-series', 'figure'),
    Output('heatmap', 'figure'),
    Output('temperature-line-chart', 'figure'),  # Added output for temperature line chart
    Output('brightness-line-chart', 'figure'),   # Added output for brightness line chart
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
)
def update_time_series(start_date, end_date):
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    
    # Time series plot for acceleration
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['accX'], mode='lines', name='AccX'))
    fig_acc.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['accY'], mode='lines', name='AccY'))
    fig_acc.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['accZ'], mode='lines', name='AccZ'))
    fig_acc.update_layout(title='Acceleration over Time', xaxis_title='Time', yaxis_title='Acceleration')
    
    # Integrate acceleration to get velocity
    filtered_df['velocityX'] = np.cumsum(filtered_df['accX']) * 0.1
    filtered_df['velocityY'] = np.cumsum(filtered_df['accY']) * 0.1
    filtered_df['velocityZ'] = np.cumsum(filtered_df['accZ']) * 0.1
    
    fig_vel = go.Figure()
    fig_vel.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['velocityX'], mode='lines', name='VelX'))
    fig_vel.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['velocityY'], mode='lines', name='VelY'))
    fig_vel.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['velocityZ'], mode='lines', name='VelZ'))
    fig_vel.update_layout(title='Velocity over Time', xaxis_title='Time', yaxis_title='Velocity')
    
    # Integrate velocity to get displacement
    filtered_df['displacementX'] = np.cumsum(filtered_df['velocityX']) * 0.1
    filtered_df['displacementY'] = np.cumsum(filtered_df['velocityY']) * 0.1
    filtered_df['displacementZ'] = np.cumsum(filtered_df['velocityZ']) * 0.1
    
    fig_disp = go.Figure()
    fig_disp.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['displacementX'], mode='lines', name='DispX'))
    fig_disp.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['displacementY'], mode='lines', name='DispY'))
    fig_disp.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['displacementZ'], mode='lines', name='DispZ'))
    fig_disp.update_layout(title='Displacement over Time', xaxis_title='Time', yaxis_title='Displacement')
    
    # Magnitude of acceleration
    fig_mag = px.line(filtered_df, x='timestamp', y='magnitude', title='Magnitude of Acceleration over Time')

    # Heatmap of forklift usage
    filtered_df['date'] = filtered_df['timestamp'].dt.date
    filtered_df['hour'] = filtered_df['timestamp'].dt.hour
    heatmap_data = filtered_df.pivot_table(index='date', columns='hour', values='magnitude', aggfunc='mean').fillna(0)
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis'
    ))
    fig_heatmap.update_layout(
        title="Forklift Usage Heatmap",
        xaxis_title="Hour of Day",
        yaxis_title="Date"
    )

    # Line chart for temperature variations
    fig_temp = px.line(filtered_df, x='timestamp', y='temperature', title='Temperature Variation over Time')

    # Line chart for brightness variations
    fig_brightness = px.line(filtered_df, x='timestamp', y='lightLevel', title='Brightness Variation over Time')

    return fig_acc, fig_vel, fig_disp, fig_mag, fig_heatmap, fig_temp, fig_brightness

# Callback to update histogram and scatter plot
@app.callback(
    Output('acceleration-histogram', 'figure'),
    Output('acceleration-scatter', 'figure'),
    Input('axis-selector', 'value'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
)
def update_histogram_and_scatter(selected_axis, start_date, end_date):
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    hist_fig = px.histogram(filtered_df, x=selected_axis, nbins=50, title=f'{selected_axis} Distribution')
    scatter_fig = px.scatter(filtered_df, x='timestamp', y=selected_axis, title=f'{selected_axis} Scatter Plot')
    return hist_fig, scatter_fig

# Callback to update 3D scatter plot
@app.callback(
    Output('3d-scatter-plot', 'figure'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
)
def update_3d_scatter(start_date, end_date):
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    fig = px.scatter_3d(filtered_df, x='accX', y='accY', z='accZ', title='3D Scatter Plot of Acceleration')
    return fig

# Callback to update summary statistics
@app.callback(
    Output('summary-stats', 'children'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
)
def update_summary_stats(start_date, end_date):
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    summary_stats = filtered_df[['accX', 'accY', 'accZ']].describe().to_dict()
    stats = []
    for axis in summary_stats:
        stats.append(html.H4(f'Summary for {axis}'))
        for stat, value in summary_stats[axis].items():
            stats.append(html.P(f'{stat}: {value:.2f}'))
    return stats

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
