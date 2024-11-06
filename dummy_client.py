#!/usr/bin/env python3

import requests

url = "http://localhost:8000/predict"
data = {
    "data": {
        "0": "1.0",
        "1": "1.0",
        "2": "0.0",
        "3": "0.0",
        "4": "1.0",
        "5": "0.0",
        "6": "0.0",
        "7": "0.0",
        "8": "1.0",
        "9": "1.0",
        "10": "0.0",
        "11": "0.0",
        "12": "0.0",
        "13": "1.0",
        "14": "0.0",
        "15": "0.0",
        "16": "0.0",
        "17": "0.0",
        "18": "0.0",
        "19": "0.0",
        "20": "0.0",
        "21": "1.0",
        "22": "0.0",
        "23": "0.0",
        "24": "0.0",
        "25": "0.0",
        "26": "0.0",
        "27": "0.0",
        "28": "1.0",
        "29": "0.0",
        "30": "0.0",
        "31": "0.0",
        "32": "0.0",
        "33": "1.0",
        "34": "0.0",
        "35": "0.0",
        "36": "0.0",
        "37": "0.0",
        "38": "0.0",
        "39": "0.0",
        "40": "0.0",
        "41": "0.0",
        "42": "0.0",
        "43": "0.0",
        "44": "0.0",
        "45": "1.0",
        "46": "0.0",
        "47": "1.0",
        "48": "1.0",
        "49": "0.0",
        "50": "0.0",
        "51": "1.0",
        "52": "2.0",
        "53": "3.0",
        "54": "0.0",
        "55": "1.0",
        "56": "1.0",
        "57": "2.0",
        "58": "3.0",
        "59": "2.0",
        "60": "2.0",
        "61": "3.0",
        # "62": "3.0",
    },
}

try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print("Predicción recibida:", response.json())
    else:
        print(f"Error en la predicción: {response.status_code}", response.json())
except requests.exceptions.RequestException as e:
    print("Error al conectar con el servicio:", e)
