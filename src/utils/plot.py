import numpy as np
import plotly.graph_objects as go
from plotly.colors import n_colors
from plotly.subplots import make_subplots


def plot_label_proportion(labels1, labels2, title, labels):
    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(go.Pie(labels=labels, values=labels1, name="Train"), 1, 1)
    fig.add_trace(go.Pie(labels=labels, values=labels2, name="Test"), 1, 2)
    
    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    
    fig.update_layout(
        title_text=title,
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Train', x=0.18, y=0.5, font_size=20, showarrow=False),
                     dict(text='Test', x=0.82, y=0.5, font_size=20, showarrow=False)])
    fig.show()


def ridgeline_plot(
    train: np.ndarray,
    test: np.ndarray,
    title="Distributions of DR features in train and test",
):
    fig = go.Figure()
    print(train)
    k = train.shape[1]
    colors1 = n_colors("rgb(520, 251, 250)", "rgb(178, 172, 192)", k, colortype="rgb"),
    colors2 = n_colors("rgb(520, 251, 250)", "rgb(112, 147, 115)", k, colortype="rgb"),
    
    for col, color1, color2 in zip(range(k), colors1, colors2):
        x0 = train[:, col]
        x1 = test[:, col]
        fig.add_trace(go.Violin(x=x0, side="positive", name=col, line_color=color1))
        fig.add_trace(go.Violin(x=x1, side="positive", name=col, line_color=color2))
    
    fig.update_traces(orientation="h", side="positive", points=False, width=3)
    
    fig.update_layout(
        xaxis_showgrid=False,
        xaxis_zeroline=False,
        height=1000,
        width=2000,
        title=title,
        showlegend=False,
    )
    
    fig.show()
