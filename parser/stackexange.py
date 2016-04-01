# -*-coding:utf-8-*-
import numpy as np
import re
import xml.etree.ElementTree as ET


def run(name='photo'):
    #Se define el glosario de términos de fotografía
    glossary = ["darkroom", "safelight", "photosensitive", "exposure",
                "enlarger", "lamphouse", "negative holder", "easel",
                "contact proofer", "contact", "enlargement", "gray scale",
                "emulsion", "resin", "glossy", "matte", "latitude",
                "over developed", "under developed", "reticulation", "stain",
                "graininess", "burning", "dodging", "toning",
                "variable contrast", "highlights", "developer", "stop",
                "fixer", "wetting", "agitation", "fog", "thin", "dense",
                "simple camera", "lens", "resolving power", "focal length",
                "normal", "macro", "telephoto", "wide-angle", "zoom",
                "shutter", "aperture", "apertures", "iris", "depth of field",
                "exposure meter", "f/number", "bellows", "focal plane",
                "ground glass", "lens hood", "lens paper", "cassette",
                "cable release", "iso", "latent image", "load", "negative",
                "focus", "infinity", "flash", "background", "foreground",
                "contrast", "camera", "crop", "parallax", "perspective",
                "time exposure", "high key", "low key", "front lighting",
                "candid", "composition", "double exposure", "filter",
                "filter factor", "panchromatic", "panning", "distortion",
                "flat", "opaque", "refraction", "over exposure",
                "under exposure", "halation", "art work", "dry mounting",
                "graduate", "fahrenheit", "film", "slide", "print", "halftone",
                "tripod", "nikon", "canon", "minolta", "exif", "bulb",
                "exposures", "second", "seconds", "light", "lights"]
    
    # Se define el glosario de palabras negativas
    neg_glossary = ["no", "never", "can't", "don't", "shouldn't"
                    "careful"]

    # Se cargan los XML de posts y usuarios
    posts = ET.parse('../data/{0}/Posts.xml'.format(name))
    users = ET.parse('../data/{0}/Users.xml'.format(name))
    
    # Se toman todas las entradas que sean respuestas
    answers = posts.findall(".//row[@PostTypeId='2']")
    
    # Se extraen todas las puntuaciones y se colocan ordenadas
    scores = np.sort(
        np.array([x.attrib['Score'] for x in answers],
                 dtype='i8'))

    # Se sacan los valores únicos de puntuaciones y las veces que aparece
    # cada uno
    unique, counts = np.unique(scores, return_counts=True)

    # Se toma como puntuación umbral la más frecuente
    threshold_score = unique[np.argmax(counts)]

    characteristics = []
    classification = []

    i = 0
    # Por cada respuesta
    for answer in answers:
        print("{0}/{1}".format(i, len(answers)), end='\r')
        i += 1

        occurrences = 0
        neg_occurences = 0
        words = re.findall(r"\w+", answer.attrib['Body'])
        # Se mira cuántas palabras de fotografía y negativas posee
        for w in words:
            if w in glossary:
                occurrences += 1
            if w in neg_glossary:
                neg_occurences += 1
                
        # Se almacena si es una respuesta bien valorada o no
        classification.append(
            int(int(answer.attrib['Score']) > threshold_score))

        user = ""
        attribute = ""
        # Se obtiene el usuario que ha respondido
        if 'OwnerUserId' in answer.attrib:
            user = answer.attrib['OwnerUserId']
            attribute = 'Id'
        elif 'OwnerDisplayName' in answer.attrib:
            user = answer.attrib['OwnerDisplayName']
            attribute = 'DisplayName'
        else:
            user = -2
            attribute = 'Id'
            
        # Se saca la resputación de dicho usuario en caso de que exista
        body = answer.attrib['Body'].lower()
        user = users.find(".//row[@{0}='{1}']".format(attribute, user))
        reputation = 0
        if user is not None:
            reputation = int(user.attrib['Reputation'])
        
        # Se saca el número de frases de la respuesta
        sentences = body.split('. ')
        sentences_len = np.fromiter(map(len, sentences), dtype='i8')
        
        # Se saca el número de palabras de la respuesta
        words_len = np.fromiter(map(len, words), dtype='i8')

        # Se construye el vector de características de la respuesta actual
        ch_vector = [reputation, # Reputación
                     len(body), # Longitud de la respuesta
                     body.count('img src'), # Número de imágenes
                     body.count('a href'), # Número de enlaces
                     occurrences, # Número de palabras de fotografía
                     neg_occurences, # Número de palabras negativas
                     len(sentences), #Número de frases
                     np.mean(sentences_len), # Longitud media de las frases
                     len(words), # Número de palabras
                     np.mean(words_len) # Longitud media de las palabras
                    ]

        characteristics.append(ch_vector)

    np.save('../files/characteristics.npy',
            np.array(characteristics, dtype='f8'))
    np.save('../files/classification.npy',
            np.array(classification, dtype='f8'))

if __name__ == "__main__":
    run()
