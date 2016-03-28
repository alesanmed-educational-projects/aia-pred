# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import numpy as np
import re
import xml.etree.ElementTree as ET


def run(name='photo'):
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

    posts = ET.parse('../data/{0}/Posts.xml'.format(name))
    users = ET.parse('../data/{0}/Users.xml'.format(name))

    answers = posts.findall(".//row[@PostTypeId='2']")

    scores = np.sort(
        np.array([x.attrib['Score'] for x in answers],
                 dtype='i8'))

    unique, counts = np.unique(scores, return_counts=True)

    threshold_score = unique[np.argmax(counts)]

    characteristics = []
    classification = []

    i = 0
    for answer in answers:
        print("{0}/{1}".format(i, len(answers)), end='\r')
        i += 1

        # print(answer.attrib['Id'])
        occurrences = 0
        for w in re.findall(r"\w+", answer.attrib['Body']):
            if w in glossary:
                occurrences += 1

        classification.append(
            int(int(answer.attrib['Score']) > threshold_score))

        user = ""
        attribute = ""
        if 'OwnerUserId' in answer.attrib:
            user = answer.attrib['OwnerUserId']
            attribute = 'Id'
        elif 'OwnerDisplayName' in answer.attrib:
            user = answer.attrib['OwnerDisplayName']
            attribute = 'DisplayName'
        else:
            user = -2
            attribute = 'Id'

        body = answer.attrib['Body'].lower()
        user = users.find(".//row[@{0}='{1}']".format(attribute, user))
        reputation = 0
        if user is not None:
            reputation = int(user.attrib['Reputation'])

        ch_vector = [int(answer.attrib['CommentCount']),
                     reputation,
                     len(body),
                     body.count('a href'),
                     body.count('img src'),
                     body.count('&'),
                     occurrences]

        characteristics.append(ch_vector)

    np.save('../files/characteristics.npy',
            np.array(characteristics, dtype='f8'))
    np.save('../files/classification.npy',
            np.array(classification, dtype='f8'))

    # plt.bar(np.arange(scores_norm.size), scores_norm)
    # plt.show()

if __name__ == "__main__":
    run()
