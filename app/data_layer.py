import os
from supabase import Client, create_client
import numpy as np
import pandas as pd
import env


class DataLayer:

    value = 20

    def __init__(self): #Supabase connections
        config = env.Config()
        self.supabase: Client = create_client(config.url, config.key)


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

