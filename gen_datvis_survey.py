import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# pip install kaleido
# pip install plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import openpyxl

def plot_survey():
    wb = openpyxl.load_workbook('data/keeratan.xlsx')
    sheets = wb.sheetnames
    print(sheets)
    print(len(sheets))
    for k in sheets:
        keeratan = pd.read_excel('data/keeratan.xlsx', sheet_name=k)
        fig = go.Figure()
        fig.update_layout(
            polar=dict(
            radialaxis=dict(
                visible=True,
                range=[1, 5]
            )),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly",
        )
        # fig.update_polars(angularaxis_tickformatstops=[0,1,2,3,4,5])
        fig.add_trace(go.Scatterpolar(
                r=keeratan['tracer'],
                theta=keeratan['keeratan'],
                fill='toself',
                name='Tracer Studi, mean value ={}'.format(round(keeratan['tracer'].mean(),1))
        ))
        fig.add_trace(go.Scatterpolar(
                r=keeratan['survey'],
                theta=keeratan['keeratan'],
                fill='toself',
                name='Survey Pengguna, mean value={}'.format(round(keeratan['survey'].mean(),1))
        ))
        # fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        pio.write_image(fig, "fig/keeratan_{}.png".format(k), scale=1, width=1200, height=800)