import os
from supabase import Client, create_client
import numpy as np
import pandas as pd


class DataLayer:

    value = 20

    def __init__(self): #Supabase connections
        url = "https://ifwbciqfpxccapdmqhmy.supabase.co"
        key = "sb_publishable_V_Fn1Ra9C8BqWNrI-HHHjA_gxiu_IXb"
        self.supabase: Client = create_client(url, key)


    def get_training_data_sp(self):
        # Consulta a la tabla 'training_samples' (ajusta el nombre)
        response = self.supabase.table("diabetes_sb").select("*").execute()
        return pd.DataFrame(response.data)
        
    def insertValue(self, value):
        response = self.supabase.table("diabetes_sb").insert(value).execute()
        return response
    
    def set_data_sp(self, value):
        response = self.supabase.table("diabetes.csv").insert({value}).execute()        
        

    def get_training_data(self):
        # Datos dummy para ejemplo: 2 características, 2 clases
        X = np.array([[0,0], [1,1], [1,0], [0,1]])
        y = np.array([0, 1, 1, 0])
        return X, y

    def get_dataFromCsv():
        return pd.read_csv('diabetes.csv')

