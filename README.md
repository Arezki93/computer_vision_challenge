# Détection d'Objets en Temps Réel avec SSD MobileNetV1 et TensorFlow Lite

## Introduction

Ce projet illustre l'implémentation d'un algorithme de détection d'objets en temps réel en utilisant le modèle SSD MobileNetV1, optimisé avec TensorFlow Lite (TFLite) pour une utilisation sur des dispositifs portables. L'objectif est de développer une routine d'inférence légère, rapide et précise, capable de traiter des flux vidéo en temps réel.

## Choix du Modèle

### Pourquoi SSD MobileNetV1 ?

Le modèle SSD MobileNetV1 a été sélectionné en raison de son équilibre entre performance, précision et efficacité des ressources. Voici les raisons principales pour ce choix :

- **Portabilité** : Le modèle est compact, avec une taille d'environ 3,99 Mo, ce qui le rend adapté au déploiement sur des dispositifs ayant une mémoire et une capacité de stockage limitées.
- **Vitesse** : Le modèle offre des temps d'inférence rapides, essentiels pour les applications en temps réel. Il atteint une vitesse d'inférence de 20 ms sur un appareil Pixel 4 utilisant le GPU, et 29 ms sur le CPU. Cela garantit que le modèle peut traiter les images vidéo assez rapidement pour fournir un retour en temps réel.
- **Précision** : Bien que SSD MobileNetV1 soit optimisé pour la vitesse, il offre toujours un niveau de précision raisonnable, avec un COCO mAP de 21. Cela le rend adapté aux applications où le traitement en temps réel est plus critique que d'obtenir la précision maximale possible.

Références :
- [TensorFlow Lite Object Detection Overview](https://www.tensorflow.org/lite/examples/object_detection/overview?hl=fr)
- [TensorFlow Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

### Pourquoi TensorFlow Lite ?

TensorFlow Lite (TFLite) est un framework qui permet le déploiement de modèles sur des dispositifs mobiles et embarqués. Il offre plusieurs avantages :

- **Optimisé pour Mobile** : Les modèles TFLite sont optimisés pour fonctionner efficacement sur des dispositifs mobiles et edge, en utilisant l'accélération matérielle lorsque disponible.
- **Multi-Plateforme** : TFLite prend en charge plusieurs plateformes, y compris Android, iOS et Linux embarqué, ce qui le rend polyvalent pour divers scénarios de déploiement.
- **Petite Empreinte** : Les modèles TFLite sont plus petits en taille par rapport à leurs homologues TensorFlow, ce qui aide à conserver les ressources des dispositifs.

## Routine d'Inférence

Voici les étapes principales du processus d'inférence :

1. **Chargement du Modèle** : Charger le modèle SSD MobileNetV1 au format TFLite.
2. **Entrée Vidéo** : Capturer la vidéo depuis une caméra ou utiliser un fichier vidéo préenregistré.
3. **Prétraitement** : Convertir les images vidéo au format d'entrée requis par le modèle.
4. **Inférence** : Exécuter le modèle sur chaque image pour détecter les objets.
5. **Post-traitement** : Décoder la sortie du modèle pour obtenir les boîtes de délimitation et les étiquettes de classe.
6. **Affichage** : Rendre les objets détectés sur les images vidéo et afficher la vidéo en temps réel.

## Test avec des Données Disponibles

Pour démontrer les capacités du modèle, la routine d'inférence a été testée sur des vidéos récupérées sur Kaggle et sur un PC local avec 8 Go de RAM. Les métriques suivantes ont été enregistrées :

- **Taux de Frame** : Le système a pu traiter les images à un taux moyen de 30 FPS.
- **Temps d'inférence moyen par image** : 33 ms.
- **Utilisation de la Mémoire** : L'empreinte mémoire était faible, avec moins de 2 Mo de RAM utilisés pendant l'inférence.

Référence vidéo : [Kaggle Video](https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v1/tfLite/metadata/1?lite-format=tflite&tfhub-redirect=true)

## Performances et Métriques

### Vitesse et Efficacité

Le modèle SSD MobileNetV1, lorsqu'il est déployé avec TensorFlow Lite, est très efficace en termes de vitesse et d'utilisation des ressources :

- **Latence** : La latence d'inférence sur un appareil Pixel 4 est d'environ 20 ms sur le GPU et 29 ms sur le CPU.
- **Taille du Modèle** : Le modèle TFLite est compact, à 3,99 Mo, ce qui le rend idéal pour les dispositifs ayant un espace de stockage limité.
- **Frugalité** : L'efficacité des ressources du modèle garantit qu'il peut fonctionner sur des dispositifs avec une puissance limitée, le rendant adapté aux applications en temps réel sur le terrain.

Références :
- [TensorFlow Lite Object Detection Overview](https://www.tensorflow.org/lite/examples/object_detection/overview?hl=fr)
- [TensorFlow Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

## Structure de Projet

```bash
├── data
│   └── ...
├── evaluation
│   ├── evaluate_ssd_mobilenet_performance.ipynb
├── inference
│   ├── __init__.py
│   ├── object_detector.py
│   └── ssd_mobilenet_tflite_inference.py
├── models
│   └── ssd_mobilenet_tflite
│       ├── label_map.txt
│       └── ...
├── main.py
├── object_detection_manager.py
├── README.md
├── requirements.txt
```

- **`evaluation/`**  
  - **`evaluate_ssd_mobilenet_performance.ipynb`**  
    - Notebook pour évaluer les performances du modèle SSD MobileNetV1.

- **`inference/`**
  - **`ssd_mobilenet_tflite_inference.py`**  
    - **`SSDMobileNetTFLiteDetector`**  
      - Rôle : Exécute l'inférence en utilisant le modèle SSD MobileNetV1 au format TFLite. Réalise les prédictions sur une image.

  - **`object_detector.py`**  
    - **`ObjectDetector`**  
      - Rôle : Utilise `SSDMobileNetTFLiteDetector` pour détecter les objets, prétraite l'image, puis dessine les bounding boxes et les classes sur l'image.
      
- **`object_detection_manager.py`**  
  - **`ObjectDetectionManager`**  
    - Rôle : Coordonne le processus de détection d'objets. Gère les fichiers d'entrée/sortie et utilise `ObjectDetector` pour traiter les images et les flux vidéo.
    
- **`main.py`**  
  - Rôle : Point d'entrée principal du programme. Configure et exécute les détections d'objets selon les arguments fournis (image ou flux vidéo).

##  Utilisation du Programme

### Installation

1. Cloner le dépôt :
    ```bash
    git clone https://github.com/Arezki93/computer_vision_challenge/
    ```
2. Se rendre dans le répertoire du projet :
    ```bash
    cd computer_vision_challenge
    ```
3. Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

### Utilisation

Le script `main.py` permet de réaliser des détections d'objets à partir d'images ou de flux vidéo en utilisant le modèle SSD MobileNetV1 en tflite. Il accepte plusieurs arguments pour configurer son comportement.

#### Arguments

- `--mode` : Mode de fonctionnement, soit `image` pour les images, soit `stream` pour les vidéos.
- `--model` : Chemin vers le fichier du modèle TFLite.
- `--labels` : Chemin vers le fichier des labels.
- `--confidence` : Seuil de confiance pour la détection.
- `--input` : Chemin vers le fichier d'entrée (image ou vidéo) ou index de la caméra.
- `--output` : Chemin pour sauvegarder le résultat traité (image ou vidéo).

## Exemples d'Exécution

**Mode stream**  
Pour traiter un flux vidéo et sauvegarder le résultat :
```bash
python main.py --mode stream --model ./models/ssd_mobilenet_tflite/ssd_mobilenet.tflite --labels ./models/label_map.txt --confidence 0.6 --input ./data/026c7465-309f6d33.mp4 --output ./data/output/026c7465-309f6d33.mp4
```




