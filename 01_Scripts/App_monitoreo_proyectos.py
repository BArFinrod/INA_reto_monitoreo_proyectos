#%%
import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
# from  PIL import Image
import io 
import pdb
import copy
import matplotlib.pyplot as plt
import seaborn as sns

from openai import OpenAI
import json
import re
#%%


st.title("App de seguimiento de proyectos utilizando simulaciones Monte Carlo y Chatgpt (Procesamiento de Lenguaje Natural)")

# Simulación Montecarlo
dfinput = pd.DataFrame([['Actividad 1',4,5,6,None,'Trabajador 1',None,None],
                      ['Actividad 2',2,3,4,['Actividad 1'],'Trabajador 2',None,None],
                      ['Actividad 3',4,8,12,['Actividad 1'],'Trabajador 3',None,None],
                      ['Actividad 4',9,11,13,['Actividad 2','Actividad 3'],'Trabajador 4',None,None],
                      ['Actividad 5',9,11,13,['Actividad 2','Actividad 3'],'Trabajador 5',None,None],
                      ['Actividad 6',9,11,13,['Actividad 2','Actividad 3'],'Trabajador 6',None,None],
                      ['Actividad 7',4,5,6,['Actividad 4','Actividad 5','Actividad 6'],'Trabajador 7',None,None],
                      ['Actividad 8',4,5,6,['Actividad 7'],'Trabajador 8',None,None],
                      ], columns = ['Actividad','Duración_mínima','Duración_media','Duración_máxima','Precedente','Recurso','start','end'])

# una actividad no puede iniciar antes de que finalice su precedente
# Calcular fechas de inicio
# al margen de los precedentes, puede calcular los valores aleatorios de forma individual e independiente

# start = 0
# calcular los días que pasaron desde el inicio de la actividad
#%%
def _get_times_mc(dfinput):
    for index, row in dfinput.iterrows():
        dfinput.loc[row.name,'Duración_mc'] = round(np.random.normal(loc=row['Duración_media'], scale=(row['Duración_media']-row['Duración_mínima'])/2),0) # 95% ic
    return dfinput

def _get_times_mc_updated(dfinput, activity, days_passed, advance):
    # days_passed = today #today - dfinput.loc[dfinput['Actividad']==activity,'start']
    duration_adjusteed = (1-advance)*dfinput.loc[dfinput['Actividad']==activity,'Duración_mc'] # o media??
    duration_adjusteed_sd = (1-advance)*((dfinput.loc[dfinput['Actividad']==activity,'Duración_media']-dfinput.loc[dfinput['Actividad']==activity,'Duración_mínima'])/2)
    durationd_updated_mc = round(np.random.normal(loc=duration_adjusteed, scale=duration_adjusteed_sd)[0], 0) # 95% ic
    # pdb.set_trace()
    dfinput.loc[dfinput['Actividad']==activity, 'Duración_mc'] = days_passed + durationd_updated_mc
    return dfinput

def _get_gantt(dfinput): # from la tabla resumen
    for index, row in dfinput.iterrows():
        if row['Precedente']==None:
            dfinput.loc[row.name, 'start'] = 0
            dfinput.loc[row.name, 'end'] = dfinput.loc[row.name, 'Duración_mc']
        else:
            start = dfinput.loc[dfinput['Actividad'].apply(lambda x: x in row['Precedente']),'end'].max()
            dfinput.loc[row.name, 'start'] = start
            dfinput.loc[row.name, 'end'] = start + dfinput.loc[row.name, 'Duración_mc'] 
    return dfinput

#%%
simulaciones = []
for simul in range(100):
    simulaciones.append(_get_times_mc(copy.copy(dfinput)))

simulaciones_gantted = []
for dfsimul in simulaciones:
    simulaciones_gantted.append(_get_gantt(copy.copy(dfsimul)))
    
dfsimulaciones = pd.concat(simulaciones, keys=range(len(simulaciones)))
dfsimulaciones_gantted = pd.concat(simulaciones_gantted, keys=range(len(simulaciones)))

serie_resumen = dfsimulaciones_gantted.groupby('Actividad').agg({'Duración_mc':'mean'})#.rename({'Duración_mc':'Duración_mean'}, axis=1)

dfresumen = _get_gantt(dfinput.merge(serie_resumen, left_on='Actividad', right_index=True, how='left'))
# dfinput_pre

dfresumen['Escenario'] = 'Inicial'
# dfresumen['start_date'] = pd.to_timedelta(dfresumen['start'], unit='D') + pd.to_datetime("27/11/2023")
# dfresumen['end_date'] = pd.to_timedelta(dfresumen['end'], unit='D') + pd.to_datetime("27/11/2023")

###################################
# NLP input

actividad_selected = st.selectbox("Seleccione la actividad",['Actividad 1',
                                                                 'Actividad 2',
                                                                 'Actividad 3',
                                                                 'Actividad 4',
                                                                 'Actividad 5',
                                                                 'Actividad 6',
                                                                 'Actividad 7',
                                                                 'Actividad 8',])
bitacora_txt = st.text_input("Coloque la bitácora. Por ejemplo, puede colocar: 'Han pasado 4 días desde el inicio y el avacen es solo del 5%. Por otro lado, se corre el riesgo de que la actividad sea paralizada en 1 semana.'")

dfsimulaciones_updated_gantted = None

if bitacora_txt!='':

    #%%
    # bitacora_txt = "Han pasado 4 días desde el inicio y el avacen es solo del 5%. Por otro lado, se corre el riesgo de que la actividad sea paralizada en 1 semana."

    key_ = st.secrets["akey"]

    def _get_description(texto):
        client = OpenAI(api_key = key_)
        # pdb.set_trace()
        texto = str(texto)
        response = client.chat.completions.create(
            # model="gpt-4",
            model = "gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": "Eres un administrador de proyectos"},
            {"role": "user", "content": texto},
            {"role": "user", "content": "completa la siguiente tabla con la información brindada: tabla:{'Días que pasaron desde el inicio de la actividad':[],'Avance (puntos porcentuales)':[]}"}]
        )
        return response.dict()['choices'][0]['message']['content']

    def _get_table(text):
        return "{" + re.split('{|}',text)[1] + "}"

    answer = _get_description(bitacora_txt)

    #%%

    exec("tabla = " + _get_table(answer))
    days_passed = float(tabla['Días que pasaron desde el inicio de la actividad'][0])
    advance = float(tabla['Avance (puntos porcentuales)'][0])/100

    ########################

    # if st.button("simular"):
    #%%
    simulaciones_updated = []
    for dfsimul in simulaciones_gantted:
        dfsimul_temp = _get_times_mc_updated(copy.copy(dfsimul), activity=actividad_selected, days_passed=days_passed, advance=advance)
        dfsimul_temp['start'] = None
        simulaciones_updated.append(_get_gantt(dfsimul_temp))

    dfsimulaciones_updated_gantted = pd.concat(simulaciones_updated, keys=range(len(simulaciones_updated)))

    serie_resumen_updated = dfsimulaciones_updated_gantted.groupby('Actividad').agg({'Duración_mc':'mean'})#.rename({'Duración_mc':'Duración_mean'}, axis=1)

    dfinput_updated = _get_gantt(dfinput.merge(serie_resumen_updated, left_on='Actividad', right_index=True, how='left'))
    dfinput_updated['Escenario'] = "Actualizado"
    dfresumen = pd.concat([dfresumen, dfinput_updated])#, keys=['Inicial','Actualizado']).reset_index().rename({'level_0':'Escenario'}, axis=1)

dfresumen['start_date'] = pd.to_timedelta(dfresumen['start'], unit='D') + pd.to_datetime("27/11/2023")
dfresumen['end_date'] = pd.to_timedelta(dfresumen['end'], unit='D') + pd.to_datetime("27/11/2023")

# guardar cada base como simulación
# aplicar el update a cada base simulada
# para la gráfica utilizar el resumen de las simulaciones
# para el histograma utilizar las simulaciones
discrete_map_resource = { 'Actualizado': 'red', 'Inicial': 'blue'}
fig = px.timeline(
                    dfresumen, 
                    x_start="start_date", 
                    x_end="end_date", 
                    y="Actividad",
                    color='Escenario',
                    color_discrete_map=discrete_map_resource,
                    hover_name="Duración_mc",
                    opacity=0.9,
                    )

fig.update_yaxes(autorange="reversed")          #if not specified as 'reversed', the tasks will be listed from bottom up       
fig.update_layout(barmode='group')
fig.update_layout(
                title='Project Plan Gantt Chart',
                hoverlabel_bgcolor='#DAEEED',   #Change the hover tooltip background color to a universal light blue color. If not specified, the background color will vary by team or completion pct, depending on what view the user chooses
                bargap=0.2,
                height=600,              
                xaxis_title="", 
                yaxis_title="",                   
                title_x=0.5,                    #Make title centered                     
                xaxis=dict(
                        tickfont_size=15,
                        tickangle = 270,
                        rangeslider_visible=True,
                        side ="top",            #Place the tick labels on the top of the chart
                        showgrid = True,
                        zeroline = True,
                        showline = True,
                        showticklabels = True,
                        tickformat="%x\n",      #Display the tick labels in certain format. To learn more about different formats, visit: https://github.com/d3/d3-format/blob/main/README.md#locale_format
                        )
            )

fig.update_xaxes(tickangle=0, tickfont=dict(family='Rockwell', color='blue', size=15))
st.plotly_chart(fig, use_container_width=True)  #Display the plotly chart in Streamlit

dfgraph = dfsimulaciones_gantted.loc[dfsimulaciones_gantted['Actividad']=='Actividad 8']
dfgraph['Escenario'] = 'Inicial'
print_text = f"Duración inicial del proyecto: {dfgraph['end'].mean()} días"

if dfsimulaciones_updated_gantted is not None:
    dfgraph2 = dfsimulaciones_updated_gantted.loc[dfsimulaciones_updated_gantted['Actividad']=='Actividad 8']
    dfgraph2['Escenario'] = 'Actualizado'
    dfgraph = pd.concat([dfgraph, dfgraph2])
    print_text = print_text + f"\nDuración actualizada del proyecto: {dfgraph2['end'].mean()} días"

plotg = sns.histplot(data = dfgraph, bins=10, x='end', hue='Escenario')
#dfsimulaciones_updated_gantted
st.header("Gráfico 1. Distribución de la duración total del proyecto en días")
st.pyplot(plotg.get_figure())
st.text(print_text)

#%%
#Main interface section 2
# st.subheader('Step 2: Upload your project plan file')
# Tasks = pd.DataFrame([['Task1','Task Description','24/11/2023','27/11/2023', 0.1,'Original'],
#                       ['Task1','Task Description','24/11/2023','28/11/2023', 0.2,'Adjusted'],
#                       ['Task2','Task Description','27/11/2023','30/11/2023', 0.2,'Original']], columns=['Task','Task Description','Start','Finish','Completion Pct','Team'])
# Tasks['Start'] = Tasks['Start'].astype('datetime64[ns]')
# Tasks['Finish'] = Tasks['Finish'].astype('datetime64[ns]')

# # grid_response = AgGrid(
# #     Tasks,
# #     editable=True,
# #     height=300,
# #     width='100%',
# #     )

# # updated = grid_response['data']
# # df = pd.DtaFrame(updated) 
# # st.table(Tasks)

# df = Tasks

# #Main interface - section 3
# st.subheader('Step 3: Generate the Gantt chart')
# discrete_map_resource = { 'Adjusted': 'red', 'Original': 'blue'}

# Options = st.selectbox("View Gantt Chart by:", ['Team','Completion Pct'],index=0)
# if st.button('Generate Gantt Chart'): 
#     fig = px.timeline(
#                     df, 
#                     x_start="Start", 
#                     x_end="Finish", 
#                     y="Task",
#                     color=Options,
#                     color_discrete_map=discrete_map_resource,
#                     hover_name="Task Description",
#                     opacity=0.5,
#                     )

#     fig.update_yaxes(autorange="reversed")          #if not specified as 'reversed', the tasks will be listed from bottom up       
#     fig.update_layout(barmode='group')
#     fig.update_layout(
#                     title='Project Plan Gantt Chart',
#                     hoverlabel_bgcolor='#DAEEED',   #Change the hover tooltip background color to a universal light blue color. If not specified, the background color will vary by team or completion pct, depending on what view the user chooses
#                     bargap=0.2,
#                     height=600,              
#                     xaxis_title="", 
#                     yaxis_title="",                   
#                     title_x=0.5,                    #Make title centered                     
#                     xaxis=dict(
#                             tickfont_size=15,
#                             tickangle = 270,
#                             rangeslider_visible=True,
#                             side ="top",            #Place the tick labels on the top of the chart
#                             showgrid = True,
#                             zeroline = True,
#                             showline = True,
#                             showticklabels = True,
#                             tickformat="%x\n",      #Display the tick labels in certain format. To learn more about different formats, visit: https://github.com/d3/d3-format/blob/main/README.md#locale_format
#                             )
#                 )
    
#     fig.update_xaxes(tickangle=0, tickfont=dict(family='Rockwell', color='blue', size=15))

#     st.plotly_chart(fig, use_container_width=True)  #Display the plotly chart in Streamlit

#     st.subheader('Bonus: Export the interactive Gantt chart to HTML and share with others!') #Allow users to export the Plotly chart to HTML
#     buffer = io.StringIO()
#     fig.write_html(buffer, include_plotlyjs='cdn')
#     html_bytes = buffer.getvalue().encode()
#     st.download_button(
#         label='Export to HTML',
#         data=html_bytes,
#         file_name='Gantt.html',
#         mime='text/html'
#     ) 
# else:
#     st.write('---') 