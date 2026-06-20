from flask import request, jsonify
from PIL import Image, ImageEnhance, ImageFilter
from app import app
from app.model import SimpleSVCModel
from app.data_layer import DataLayer
import numpy as np
import easyocr
import io



model = SimpleSVCModel()
data_layer = DataLayer()

# Entrenar modelo al iniciar la app
model.train()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()    
    features_list = [   
            data['Pregnancies'],
            data['Glucose'],
            data['BloodPressure'],
            data['SkinThickness'],
            data['Insulin'],
            data['BMI'],
            data['DiabetesPedigreeFunction'],
            data['Age']
        ]

    features = np.array(features_list).reshape(1, -1)
    
    try:
        accuracy, prediction = model.predict_fun(features)
        data.update({"Outcome": prediction[0]})
        data_layer.insertValue(data)
        return jsonify({'prediction': prediction, 'confidence': accuracy})
    except Exception as e:
        print("estoy aca")
        print(str(e))
        return jsonify({'error': str(e)}), 500
    
@app.route('/train', methods=['GET'])
def is_trained():    
    if (model.trained()):
       message = 'The model is trained' 
    else: 
        message ='The model is not trained'

    return jsonify({'message': message}), 400


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No se recibió imagen'}), 400

    file = request.files['image']

    # Leer imagen en memoria
    img_bytes = file.read()   

    # Convertir imagen a formato que EasyOCR pueda procesar (numpy array)
    img_np = preprocess_image(img_bytes)
    # Inicializar lector EasyOCR (idioma español, por ejemplo)
    reader = easyocr.Reader(['es'], gpu=False)

    # Extraer texto
    result = reader.readtext(img_np)

    # Procesar resultado para extraer datos importantes    
    extracted_data = {}
    extracted_data['nombre'] = get_value_below_label(result, ['nombre'])
    extracted_data['fecha_nacimiento'] = get_value_below_label(result, ['fecha de naciemiento'])
    extracted_data['apellido'] = get_value_below_label(result, ['apellido'])

    return jsonify(extracted_data)

def get_value_below_label(results, label_keywords):
    
    label_bbox = None
    label_text = None

    # Buscar la etiqueta
    for bbox, text, conf in results:
        if any(kw in text.lower() for kw in label_keywords):
            label_bbox = bbox
            label_text = text
            break

    if not label_bbox:
        return None  # No encontró la etiqueta

    # Coordenadas del bbox de la etiqueta
    # bbox es lista de 4 puntos: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    # Tomamos el punto inferior izquierdo o derecho para referencia vertical
    label_bottom_y = max([point[1] for point in label_bbox])
    label_center_x = sum([point[0] for point in label_bbox]) / 4

    # Buscar texto debajo de la etiqueta
    candidates = []
    for bbox, text, conf in results:
        text_top_y = min([point[1] for point in bbox])
        text_center_x = sum([point[0] for point in bbox]) / 4

        # Condición: texto debajo y horizontalmente cercano
        if text_top_y > label_bottom_y and abs(text_center_x - label_center_x) < 100:  # 100 pixeles de tolerancia horizontal
            candidates.append((text_top_y - label_bottom_y, text))

    if not candidates:
        return None

    # Ordenar por distancia vertical y tomar el más cercano
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def preprocess_image(image_bytes):
    # Abrir imagen con Pillow desde bytes
    img = Image.open(io.BytesIO(image_bytes))

    # Convertir a escala de grises
    img = img.convert('L')

    # Mejorar contraste
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)

    # Aplicar filtro de nitidez
    img = img.filter(ImageFilter.SHARPEN)
    #img = img.resize([400,300])
    # Convertir a numpy array para EasyOCR
    img_np = np.array(img)

    return img_np



@app.route('/hello', methods=['GET'])
def hello():
    with open('index.html', 'r') as file:
        index = file.read()
    return index

