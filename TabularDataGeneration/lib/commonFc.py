import pandas as pd
# import torch
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.compose import ColumnTransformer
import json


def check_constraints_noTNM(current,print_flag=False):
    in_situ=["carcinome in situ","carcinome intracanalaire non infiltrant","carcinome lobulaire in situ",
             "adenocarcinome in situ","carcinome intracanalaire et carcinome lobulaire in situ"]

    # remove entries  with no diagnostic or no echantillon
    if current["ref_morpho_name_tumor_0"] in ["unknown"] or  current["type_diagnostique"] in ["unknown"]:
        return False
    
    # in situ 
    if current["embols_vasculaires_tumor_0"] not in ["unknown"] and current["ref_morpho_name_tumor_0"] in in_situ:
        return False

    if current["ganglions_preleves"] not in ["unknown"] and current["ref_morpho_name_tumor_0"] in in_situ:
        return False

    if current["ref_grade_tumor_0"] not in ["unknown"] and current["ref_morpho_name_tumor_0"] in in_situ:
        return False


    
    # size and margins for biopsie
    if current["taille_tumor_0"]!="unknown" and current["type_diagnostique"] in ["biopsie","curage ganglionnaire"]: 
        if print_flag:
                print("biopsie with taille", current["taille_tumor_0"])
        return False

    if current["type_diagnostique"] in ["biopsie","curage ganglionnaire"] and  current["marges_saines_tumor_0"]!= "unknown": 
        return False

    # size required for piece operatoire??
    if current["type_diagnostique"] in ["tumorectomie","mammectomie", "exérèse chirurgicale"] and  current["taille_tumor_0"]== "unknown": 
        return False

    # ganglions constraints
    if current["ganglions_atteints"] =="unknown" and current["ganglions_preleves"] not in ["unknown"]:
        if print_flag:
                print("gang att prel", current["ganglions_atteints"],current["ganglions_preleves"])
        return False

    if current["ganglions_atteints"] =="0" and current["ganglions_preleves"] not in ["1-3","4-9", "+10"]:
        if print_flag:
                print("gang att prel", current["ganglions_atteints"],current["ganglions_preleves"])
        return False
    if current["ganglions_atteints"] =="1-3" and current["ganglions_preleves"] not in ["1-3","4-9", "+10"]:
        if print_flag:
            print("gang att prel", current["ganglions_atteints"],current["ganglions_preleves"])
        return False
    if current["ganglions_atteints"] =="4-9" and current["ganglions_preleves"] not in ["4-9","+10"]:
        if print_flag:
            print("gang att prel", current["ganglions_atteints"],current["ganglions_preleves"])
        return False
    if current["ganglions_atteints"] =="+10" and current["ganglions_preleves"] not in ["+10"]:
        if print_flag:
            print("gang att prel", current["ganglions_atteints"],current["ganglions_preleves"])
        return False

    if current["ganglions_atteints"] in ["unknown","0"] and current["rupture_capsulaire"] not in ["unknown"]:        
        return False
    # curage ganglionaire: no grade, has ganglions, 
    if (current["type_diagnostique"] =="curage ganglionnaire") and  current["ref_grade_tumor_0"]not in ["unknown"]:
        return False
    if (current["type_diagnostique"] =="curage ganglionnaire") and  current["ganglions_preleves"] in ["unknown","0"]:
        return False

    return True



def check_constraints(current,print_flag=False,use_new=True):
    in_situ=["carcinome in situ","carcinome intracanalaire non infiltrant","carcinome lobulaire in situ",
             "adenocarcinome in situ","carcinome intracanalaire et carcinome lobulaire in situ"]
    if use_new and current["ganglions_atteints"] =="unknown" and current["ganglions_preleves"] not in ["unknown"]:
        if print_flag:
                print("gang att prel", current["ganglions_atteints"],current["ganglions_preleves"])
        return False

    if current["ganglions_atteints"] =="0" and current["ganglions_preleves"] not in ["1-3","4-9", "+10"]:
        if print_flag:
                print("gang att prel", current["ganglions_atteints"],current["ganglions_preleves"])
        return False
    if current["ganglions_atteints"] =="1-3" and current["ganglions_preleves"] not in ["1-3","4-9", "+10"]:
        if print_flag:
            print("gang att prel", current["ganglions_atteints"],current["ganglions_preleves"])
        return False
    if current["ganglions_atteints"] =="4-9" and current["ganglions_preleves"] not in ["4-9","+10"]:
        if print_flag:
            print("gang att prel", current["ganglions_atteints"],current["ganglions_preleves"])
        return False

    # ganglions and n
    if use_new and current["ganglions_atteints"] =="unknown" and current["n"] not in ["NX"]:
        if print_flag:
            print("gang att n", current["ganglions_atteints"],current["n"])
        return False
    if current["ganglions_atteints"] =="0" and current["n"] not in ["N0","NX"]:
        if print_flag:
            print("gang att n", current["ganglions_atteints"],current["n"])
        return False
    if current["ganglions_atteints"] =="1-3" and current["n"] not in ["N1","NX"]:
        if print_flag:
            print("gang att n", current["ganglions_atteints"],current["n"])
        return False
    if current["ganglions_atteints"] =="4-9" and current["n"] not in ["N2","NX"]:
        if print_flag:
            print("gang att n", current["ganglions_atteints"],current["n"])
        return False
    if current["ganglions_atteints"] =="+10" and current["n"] not in ["N3","NX"]:
        if print_flag:
            print("gang att n", current["ganglions_atteints"],current["n"])
        return False
    # t and tumor size 

    # 11-20, 6-10, 21-50, unknown, 1-5, 50+,0
    if current["taille_tumor_0"] == 'unknown' and current["t"] not in ["Tx","Tis","T4"]:
        if print_flag:
            print("taille, t", current["taille_tumor_0"],current["t"])
        return False
    if current["taille_tumor_0"] == '0' and current["t"] not in ["T0","Tis","T4"]:
        if print_flag:
            print("taille, t", current["taille_tumor_0"],current["t"])
        return False
    if current["taille_tumor_0"] in ['1-5','6-10','11-20'] and current["t"] not in ["T1","Tis", "T4"]:
        if print_flag:
            print("taille, t", current["taille_tumor_0"],current["t"])
        return False
    if current["taille_tumor_0"]== '21-50' and current["t"] not in ["T2","Tis", "T4"]:
        if print_flag:
            print("taille, t", current["taille_tumor_0"],current["t"])
        return False
    if current["taille_tumor_0"]== '50+' and current["t"] not in ["Tis","T3", "T4"]:
        if print_flag:
            print("taille, t", current["taille_tumor_0"],current["t"])
        return False
    if current["t"]=="Tis" and current["ref_morpho_name_tumor_0"] not in in_situ:
        return False
    if current["t"]!="Tis" and current["ref_morpho_name_tumor_0"] in in_situ:
        return False
    if current["embols_vasculaires_tumor_0"] not in ["unknown"] and current["ref_morpho_name_tumor_0"] in in_situ:
        return False

    if current["rupture_capsulaire"] not in ["unknown"] and current["ref_morpho_name_tumor_0"] in in_situ:
        return False
    
    if current["ganglions_preleves"] not in ["unknown"] and current["ref_morpho_name_tumor_0"] in in_situ:
        return False
    return True



