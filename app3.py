import streamlit as st
import pandas as pd
import joblib

st.title("app di penguins")
model_pipe = joblib.load('penguinspipe.pkl')
print('modello caricato')
#input front end
island = st.selectbox('inserire isola',['Torgensen','Dream','Biscoe'])
bill_length_mm = st.number_input('inserire lungh becco',20.0,60.0,20.33)
bill_depth_mm = st.number_input('inserire larghezza becco',8.0,15.0,10.5)
flipper_length_mm = st.number_input('inserire lungh pinna',20.0,60.0,40.25)
body_mass_g =  st.number_input('inserire massa',2000.0,6000.0,3590.50)
sex = st.selectbox('inserire sesso',['male','female'])

#island = 'Torgersen'
# bill_length_mm = 20.33
# # # #bill_depth_mm = 10.50
# # # #flipper_length_mm = 40.25
# # # #body_mass_g = 3590.50
# # # #sex = 'female'

data = {
        "island": [island],
        "bill_length_mm": [bill_length_mm],
        "bill_depth_mm": [bill_depth_mm],
        "flipper_length_mm": [flipper_length_mm],
        "body_mass_g": [body_mass_g],
        "sex": [sex]
        }

input_df = pd.DataFrame(data)
res = model_pipe.predict(input_df).astype(int)[0]
print(res)

classes = {0:'Adelie',
           1:'Gentoo',
           2:'Chinstrap'
           }

y_pred = classes[res]


if st.button ('Prediction'):
    st.success(f'la specie predetta è  {y_pred}')