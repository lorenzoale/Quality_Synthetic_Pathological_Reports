import numpy as np
import unicodedata

# import sample_values

# 'carcinome intracanalaire non infiltrant',
# 'adenocarcinome canalaire in situ',
_rng = np.random.default_rng(42)



def normalize_text(value):
    if value is None:
        return ""
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("utf-8")
    return value

def convert_cerb_to_her2_score_random(value, p_zero=0.7, rng=None):
    v = normalize_text(value)
    rng = rng or _rng

    if v == "positif":
        return "3+"
    elif v == "douteux":
        return "2+"
    elif v == "negatif":
        return rng.choice(["0", "1+"], p=[p_zero, 1 - p_zero])
    elif v in {"unknown", ""}:
        return "unknown"
    else:
        return "unknown"


def normalize_specimen(entry_data):
    type_diag = entry_data.get("type_diagnostique", "unknown")
    size = entry_data.get("taille_tumor_0", "unknown")
    nodes = entry_data.get("ganglions_preleves", "unknown")

    if type_diag != "unknown":
        type_diag = str(type_diag).strip().lower()
        # mapping = {
        #     "exérèse chirurgicale": "tumorectomie",
        #     "tumorectomie": "tumorectomie",
        #     "biopsie": "biopsie",
        #     "curage ganglionnaire": "curage ganglionnaire",
        #     "mastectomie": "mastectomie",
        # }
        # return mapping.get(type_diag, type_diag)
        return type_diag

    if size == "unknown" or (nodes == "unknown" and str(size) == "0"):
        return "biopsie"
    return "tumorectomie"


def format_sbr(sbr: str) -> str:
    mapping = {
        "SBR1": "Grade I de SBR (SBR 1 ou GRADE 1)",
        "SBR2": "Grade II de SBR (SBR 2 ou GRADE 2)",
        "SBR3": "Grade III de SBR (SBR 3 ou GRADE 3)",
    }
    key = sbr.strip().upper()
    return mapping.get(key, sbr)


def _safe_int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None

def create_donnees_cliniques_rag(entry_data):
    
    in_situ=["carcinome in situ","carcinome intracanalaire non infiltrant","carcinome lobulaire in situ",
             "carcinome canalaire in situ","adenocarcinome papillaire intracanalaire non infiltrant",
         "adenocarcinome in situ","carcinome intracanalaire et carcinome lobulaire in situ"]
    
    donnees_cliniques=""
    donnees_rag=""
    

    # 1) type diagnostique
    type_diag = entry_data.get("type_diagnostique", "unknown")
    if type_diag=="curage ganglionnaire": 
        donnees_rag+= "GANGLION LYMPHATIQUE - "
    else:
        donnees_rag+= "SEIN - "
    if type_diag != "unknown":
        donnees_rag=donnees_rag+f"{type_diag}\n"
        donnees_cliniques += f"- Échantillon : {type_diag}\n"
    else:
        type_diag = "tumorectomie"
        if (
            entry_data.get("taille_tumor_0", "unknown") == "unknown"
            or (
                entry_data.get("ganglions_preleves", "unknown") == "unknown"
                and entry_data.get("taille_tumor_0", "unknown") == "0"
            )
        ):
            type_diag = "biopsie"
        donnees_cliniques += f"- Échantillon : {type_diag}\n"    

    # 2) morphologie
    morpho_raw = entry_data.get("ref_morpho_name_tumor_0", "unknown")
    if morpho_raw != "unknown":
        morpho = (
            "carcinome canalaire in situ"
            if morpho_raw == "carcinome intracanalaire non infiltrant"
            else morpho_raw
        )
        donnees_rag=donnees_rag+f"{morpho}\n"
        donnees_cliniques += f"- Diagnostic (morphologie) : {morpho}\n"
    

    # 3) grade SBR
    if morpho_raw in in_situ:
        donnees_cliniques += "- Grade SBR : non applicable (lésion non infiltrante)\n"
    elif entry_data.get("ref_grade_tumor_0", "unknown") == "unknown":
        donnees_cliniques += "- Grade SBR : non évalué\n"
    else:
        donnees_cliniques += f"- Grade SBR : {entry_data.get('ref_grade_tumor_0', '')}\n"    
        donnees_rag=donnees_rag+f"{format_sbr(entry_data.get("ref_grade_tumor_0","unknown"))}\n"


    # 4) taille tumorale
    taille = entry_data.get("taille_tumor_0", "unknown")
    if taille != "unknown" and type_diag != "biopsie":
        donnees_cliniques += f"- Taille tumorale : {taille} mm\n"
        donnees_rag=donnees_rag+f"- Taille tumorale : {entry_data.get('taille_tumor_0', 'non spécifié')} mm\n"
    elif taille == "unknown":
        if type_diag in {"biopsie", "curage ganglionnaire"}:
            donnees_cliniques += f"- Taille tumorale : non applicable ({type_diag})\n"
        else:
            donnees_cliniques += "- Taille tumorale : non évaluée\n"
            
        
    ########### ganglions
    # 5) ganglions examinés
    ganglions_preleves = entry_data.get("ganglions_preleves", "unknown")
    if ganglions_preleves != "unknown":
        donnees_cliniques += f"- Nombre de ganglions examinés : {ganglions_preleves}\n"
        donnees_rag+= f"{ganglions_preleves} ganglions examinés."
    else:
        donnees_cliniques += "- Nombre de ganglions examinés : non évalués\n"

    # 6) ganglions atteints
    ganglions_atteints = entry_data.get("ganglions_atteints", "unknown")
    if ganglions_preleves != "unknown" and ganglions_atteints != "unknown":
        donnees_cliniques += (
            f"- Nombre de ganglions atteints : {ganglions_atteints} "
            f"({ganglions_atteints}/{ganglions_preleves})\n"
        )
        # rag
        if ganglions_atteints== 0 or ganglions_atteints== "0": 
            rag_att="Absence de métastase ganglionnaire (0/N).\n"
        elif ganglions_atteints== 1 or ganglions_atteints== "1":
            rag_att=f"{ganglions_atteints} ganglion métastatique ({ganglions_atteints}/{ganglions_preleves}).\n"
        else:
            rag_att=f"{ganglions_atteints} ganglions métastatiques ({ganglions_atteints}/{ganglions_preleves}).\n"
        donnees_rag+=rag_att

        
    elif ganglions_preleves != "unknown" and ganglions_atteints == "unknown":
        donnees_cliniques += "- Nombre de ganglions atteints : non spécifié\n"

    # 7) emboles vasculaires
    emb = entry_data.get("embols_vasculaires_tumor_0", "unknown")
    if emb != "unknown":
        if emb == "1":
            donnees_cliniques += "- Présence d’emboles vasculaires\n"
            donnees_rag=donnees_rag+f"- Présence d’emboles vasculaires\n"
        else:
            donnees_cliniques += "- Absence d’emboles vasculaires\n"
            donnees_rag=donnees_rag+f"- Absence d'emboles vasculaires.\n"
    else:
        donnees_cliniques += "- Emboles vasculaires : non évalués\n"
    
        

    # 8) rupture capsulaire
    # no entries with word rupture in the templates, avoid for RAG
    rupture_caps = str(entry_data.get("rupture_capsulaire", "unknown")).strip().lower()
    gang_atteints_int = _safe_int(ganglions_atteints)
    has_positive_nodes = gang_atteints_int is not None and gang_atteints_int > 0

    if rupture_caps == "non":
        donnees_cliniques += "- Présence de rupture capsulaire : non\n"
    elif rupture_caps == "oui" and has_positive_nodes:
        donnees_cliniques += "- Présence de rupture capsulaire : oui\n"
    else:
        donnees_cliniques += "- Présence de rupture capsulaire : non évaluée\n"


    # 9) récepteurs hormonaux
    re = entry_data.get("re_tumor_0", "unknown")
    if re == "unknown":
        donnees_cliniques += "- RE ou RO (récepteurs aux œstrogènes) : non évalués\n"
    elif re == "positifs":
        donnees_cliniques += (
            f"- RE ou RO (récepteurs aux œstrogènes) : positifs ; "
            f"pourcentage : {entry_data['re_perc']}%\n"
        )
        donnees_rag += f"- RE ou RO (récepteurs aux œstrogènes) : positifs.\n"
    else:
        donnees_cliniques += "- RE ou RO (récepteurs aux œstrogènes) : négatifs\n"

    rp = entry_data.get("rp_tumor_0", "unknown")
    if rp == "unknown":
        donnees_cliniques += "- RP (récepteurs à la progestérone) : non évalués\n"
    elif rp == "positifs":
        donnees_cliniques += (
            f"- RP (récepteurs à la progestérone) : positifs ; "
            f"pourcentage : {entry_data['rp_perc']}%\n"
        )
        donnees_rag += f"- RP (récepteurs à la progestérone) : positifs.\n"
    else:
        donnees_cliniques += "- RP (récepteurs à la progestérone) : négatifs\n"
    
    # 10) HER2 / CerbB2
    cerb2 = convert_cerb_to_her2_score_random(
        entry_data.get("cerb_tumor_0", "unknown"),
        p_zero=0.7,
    )
    if cerb2 != "unknown":
        donnees_cliniques += f"- HER2 (CerbB2) : {cerb2}\n"
        donnees_rag += f"HER2 (Her2Neu, CerbB2 ou CERB2): {entry_data.get('cerb_tumor_0', 'unknown')} (Score {cerb2}). \n"
    else:
        donnees_cliniques += "- HER2 (CerbB2) : non évalué\n"        

    
    # 11) Ki67
    ki67 = entry_data.get("ki67_tumor_0", "unknown")
    if ki67 != "unknown":
        donnees_cliniques += f"- Prolifération cellulaire (Ki67) : {ki67}%\n"
        donnees_rag+= f"- Prolifération cellulaire (Ki67) : {ki67}%\n"
    else:
        donnees_cliniques += "- Prolifération cellulaire (Ki67) : non évaluée\n"
    
    # 12) marges
    marges = entry_data.get("marges_saines_tumor_0", "unknown")
    if marges != "unknown":
        donnees_cliniques += f"- Marges chirurgicales saines : {marges}\n"
        # oui/non
        if marges=="oui": 
            donnees_rag+= "Berges d'exérèse saines"
        else:
            donnees_rag+= "Berges d'exérèse au contact."        
    else:
        if type_diag == "biopsie":
            donnees_cliniques += "- Marges chirurgicales saines : non applicable (biopsie)\n"
        else:
            donnees_cliniques += "- Marges chirurgicales saines : non évaluées\n"

    return donnees_cliniques,donnees_rag






def generate_report_prompt(donnees_cliniques, RAG_ex, system_prompt=False):
    user_prompt=f"""TÂCHE: 
Rédige un **compte rendu anatomopathologique complet en français médical** à partir des données cliniques délimitées ### DONNÉES CLINIQUES ###  ci-dessous. 

Les exemples fournis servent uniquement de modèle de style, de ton et de structure.
Ils doivent être respectés sur la forme, tout en étant strictement adaptés aux données cliniques fournies.
N’utilise que les données cliniques fournies ci-dessous. Tu peux seulement ajouter un détail rédactionnel générique ou plausible lorsqu’il est strictement nécessaire pour la fluidité du compte rendu, à condition qu’il ne modifie, ne complète abusivement, ne contredise ni n’interprète les données cliniques fournies.
Toute donnée inconnue, non évaluée, absente ou non documentée doit rester inconnue, non évaluée, absente ou non documentée dans le texte final. Ne la transforme jamais en donnée négative, positive ou normale.
Exemple : si les ganglions sont indiqués comme non évalués, supprime toute mention de ganglions examinés, de pN0 ou de 0/0.

    
### DONNÉES CLINIQUES ###
{donnees_cliniques}
### FIN DONNÉES CLINIQUES ###


Tu dois produire exactement les sections suivantes :
- Titre  (indiquer une latéralité et un quadrant choisis de façon plausible)
- Examen macroscopique 
- Examen microscopique
- Étude immunohistochimique
- Conclusion

────────────────────────────────────────
RÈGLES OBLIGATOIRES
────────────────────────────────────────
1) Cohérence générale
- Toute valeur fixe fournie doit être reprise exactement, sans modification.
- Ne jamais faire varier une même donnée entre les sections.

2) Données non évaluées
- Toute donnée indiquée comme "non évaluée", "non réalisée" ou "non disponible" dans les DONNÉES CLINIQUES doit être reprise strictement telle quelle.
- Ne jamais la compléter, estimer ou interpréter.
- Ne jamais la remplacer par une valeur normale ou pathologique (ex : "marges saines", "absence d’emboles", etc.).
- Cette règle s’applique même si les exemples contiennent une formulation correspondante.

3) Compléments descriptifs autorisés
- Tu peux enrichir la description macroscopique et microscopique avec des éléments plausibles, purement descriptifs et rédactionnels.
- Tu ne dois jamais modifier, compléter, interpréter ni contredire les données cliniques fournies.
- Les variables cliniques (taille tumorale, ganglions, marges, emboles, récepteurs, etc.) doivent être reprises strictement à l’identique.

4) Taille tumorale et pièce opératoire
- La taille tumorale ne doit jamais être inventée ou estimée si elle n’est pas fournie.
- Si elle est fournie, elle doit être identique dans la macroscopie, la microscopie et la conclusion.
- La taille tumorale doit toujours être inférieure ou égale à la taille de la pièce opératoire.
- Si poids et dimensions de la pièce sont ajoutés, ils doivent être plausibles et cohérents entre eux.

5) Hormonorécepteurs et Ki67
- Si RE ou RP sont positifs, utiliser le pourcentage fourni dans les données cliniques.
- Si RE ou RP sont négatifs, les rapporter comme négatifs sans inventer d’intensité absente.
- Une seule valeur par marqueur, sans contradiction.

6) Ganglions lymphatiques
- Le nombre de ganglions atteints doit toujours être inférieur ou égal au nombre de ganglions examinés.
- Le nombre décrit en macroscopie, microscopie et conclusion doit être identique.
- Si des ganglions sont examinés et qu’il n’y a aucune métastase ganglionnaire, indiquer 0/N dans la conclusion (N = nombre de ganglions examinés).
- Si le statut ganglionnaire est non évalué, ne pas mentionner de nombre de ganglions ni de statut (pas de 0/0, pN0, etc.).
- Ne pas mentionner de rupture capsulaire si les ganglions sont négatifs.

7) Grade
- Le grade SBR ne s’applique qu’aux lésions infiltrantes.
- Si un score chiffré est présent (ex. 3 + 2 + 1), calculer le total et en déduire le grade :
  - 3 à 5 : grade I
  - 6 à 7 : grade II
  - 8 à 9 : grade III
- Corriger toute incohérence entre score et grade.

8) Exemples
- Supprimer tout doublon ou variante contradictoire issus des exemples.
- Ne retenir que ce qui est compatible avec les données du patient.
- Les exemples servent uniquement à reproduire le style rédactionnel, jamais les données.
- Si une variable est définie comme absente ou non évaluée dans les DONNÉES CLINIQUES, ne jamais la réintroduire à partir des exemples.

────────────────────────────────────────
RÈGLES DE STYLE
────────────────────────────────────────

- Français médical clair, structuré, homogène, sans redondance.
- Fournir uniquement le rapport final.
- N’ajouter aucune explication, justification ou note hors compte rendu.

### EXEMPLES ###
{RAG_ex}
### FIN EXEMPLES ###
"""
    
    if system_prompt:
        return [
            {
                "role": "system",
                "content": (
                    "Tu es un médecin anatomopathologiste hospitalier expérimenté. "
                    "Tu rédiges des comptes rendus anatomopathologiques complets, cohérents et conformes aux standards hospitaliers français. "
                    "Tu corriges automatiquement toute incohérence médicale, stylistique ou logique."
                )
            },
            {
                "role": "user",
                "content": f"""{user_prompt}"""
    
            }
        ]
    else: 
        system_prompt = """RÔLE :
        Tu es un médecin anatomopathologiste hospitalier expérimenté.
        Tu rédiges des comptes rendus anatomopathologiques complets, cohérents et conformes aux standards hospitaliers français.
        Tu corriges automatiquement toute incohérence médicale, stylistique ou logique.
        
        """
        return [
            {
                "role": "user",
                "content":  f"{system_prompt}{user_prompt}"
    
            }
        ]



