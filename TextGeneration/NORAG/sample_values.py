import numpy as np


# Taille tumorale : get 1 val for each bin
def sample_size_value(bin_name):
    if bin_name == "0":
        return 0

    elif bin_name == "1-5":
        # plus probable: mode
        # value= np.random.triangular(left=1, mode=5, right=5)
        value = np.random.choice([1,2,3,4,5], p=[0.05,0.1,0.15,0.2,0.5])
        

    elif bin_name == "6-10":
        # value= np.random.triangular(left=6, mode=10, right=10)
        value = np.random.choice([6,7,8,9,10], p=[0.15,0.15,0.2,0.1,0.4])

    elif bin_name == "11-20":
        # value= np.random.triangular(left=11, mode=15, right=20)
        value = np.random.choice([11,12,13,14,15,16,17,18,20], p=[0.05,0.15,0.05,0.05,0.325,0.05,0.05,0.05,0.225])

    elif bin_name == "21-50":
        value= np.random.triangular(left=21, mode=30, right=50)

    elif bin_name == "50+":
        # 50 + loi exponentielle (longue queue réaliste)
        value = 50 + np.random.exponential(scale=30)
        value= min(value, 200)  # tronquer à 200 mm maximum
    else:
        raise ValueError("bin inconnu")
    return int(round(value))  # convert float → int    

# Ganglions atteints : à déterminer, sans dépasser le nombre de ganglions examinés.
def sample_gang_value(bin_name, other=None):
    """
    Sample number of nodes : examinated lymph nodes (ganglions prélevés) or 
    affected lymph nodes (ganglions atteints) given a range.
    If 'other' is provided, it represents the number of nodes sampled previously (ganglions prélevés)
    and the result will be capped at 'other'.
    """
    if bin_name == "0":
        value = 0

    elif bin_name == '1-3':
        value = np.random.choice([1,2,3], p=[0.55,0.3,0.15])

    elif bin_name == '4-9':
        # Triangular-like discrete: mode at 6
        value = np.random.choice([4,5,6,7,8,9], p=[0.25,0.15,0.10,0.10,0.25,0.15])

    elif bin_name == '+10':
        # Linearly decreasing from 10 to 25
        values = np.arange(10, 26)  # 10 to 25
        weights = np.arange(16, 0, -1)  # 16,15,...,1
        weights = weights / weights.sum()
        value = np.random.choice(values, p=weights)

    else:       
        raise ValueError("bin inconnu")

    # Cap the sampled value by 'other' if provided
    if other is not None:
        value = min(value, other)

    return value


# Ki67 : 
def sample_ki67(bin_name=None):
    """
    Sample Ki-67 as DISCRETE multiples of 5, depending on the bin.
    Values are integers: 0,5,10,15,...,100
    """

    # Choose bin if not provided
    if bin_name is None:
        bin_name = np.random.choice(ki67_bins, p=ki67_probs)

    if bin_name == '0-5':
        values = [0,1,2,3,4, 5]
        weights = [0.05,0.05,0.05,0.1,0.1, 0.65]    # 0 slightly more common
        # normalize weights just in case

    elif bin_name == '6-10':
        values = [6,7,8,9, 10]
        weights = [0.05,0.05,0.2, 0.05,0.65]    # many pathologists round 8 → 10

    elif bin_name == '11-20':
        values = [12, 15,18, 20]
        weights = [0.05, 0.4, 0.15, 0.4]  # most 15 but also some for 20 

    elif bin_name == '21-30':
        values = [22, 25, 30]
        weights = [0.1, 0.45, 0.45]

    elif bin_name == '31-40':
        values = [ 35, 40]
        weights = [ 0.3, 0.7]

    elif bin_name == '41-100':
        values = [  50,  60, 70, 80, 90]
        weights = [ 0.4, 0.25,0.2, 0.1, 0.05]

    else:
        raise ValueError("Unknown bin")

    # normalize weights just in case
    weights = np.array(weights)
    weights = weights / weights.sum()

    return int(np.random.choice(values, p=weights))

    
def sample_receptor_value(receptor: str):
    # Distributions as given (approx values)
    distributions = {
        "RE": {
            100: 0.30,
            90: 0.10,
            80: 0.05,
            70: 0.02,
            95: 0.01,
            30: 0.01,
            50: 0.01,
            40: 0.005,
            60: 0.003,
            20: 0.003
        },
        "RP": {
            100: 0.15,
            90: 0.10,
            80: 0.06,
            50: 0.03,
            70: 0.03,
            30: 0.02,
            60: 0.02,
            20: 0.02,
            40: 0.01,
            10: 0.01
        }
    }

    if receptor not in distributions:
        raise ValueError(f"Unknown receptor type '{receptor}'")

    values = list(distributions[receptor].keys())
    weights = list(distributions[receptor].values())

    # Normalize to sum to 1
    total = sum(weights)
    norm_weights = [w / total for w in weights]

    # Sample a single value
    return np.random.choice(values, p=norm_weights)


    
def sample_value(col_name,ran,other=None):
    # print(ran,other)
    if col_name=="ki67_tumor_0":
        return sample_ki67(ran)
    elif col_name=="taille_tumor_0":
        return sample_size_value(ran)
    elif col_name=="ganglions_preleves":
        return sample_gang_value(ran)
    elif col_name=="ganglions_atteints":
        return sample_gang_value(ran,other)
    else: 
        return None
    
    
