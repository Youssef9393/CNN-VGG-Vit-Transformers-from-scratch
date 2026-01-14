# Image Mining & Deep Learning
## ÉTAPE 2 - Prédiction avec Modèles Entraînés

---

##  Description

Ce projet contient les modèles de deep learning entraînés pour la classification binaire des lésions cutanées et un script de prédiction permettant d'utiliser ces modèles sur de nouvelles images.

**Classes** :
- **Classe 0** : Malade (mel - mélanome)
- **Classe 1** : Saine (autres lésions bénignes)

### Modèles Disponibles

1. **CNN Personnalisé** - Architecture CNN à 5 blocs convolutionnels
2. **VGG16** - Architecture VGG16 implémentée from scratch
3. **ViT-1** - Vision Transformer avec 1 bloc encodeur
4. **ViT-2** - Vision Transformer avec 2 blocs encodeurs

Tous les modèles utilisent une **tête de classification VGG16 commune** pour assurer une comparaison équitable.


##  Utilisation

### 1. Prédiction avec un modèle spécifique

```bash
python predict.py --image HAM5000\HAM500_images\ISIC_0024314.jpg --model cnn
```

Modèles disponibles : `cnn`, `vgg16`, `vit-1`, `vit-2`

### 2. Prédiction avec tous les modèles

```bash
python predict.py --image HAM5000\HAM500_images\ISIC_0024314.jpg --all
```

### 3. Exemples d'utilisation

**Exemple 1 : Utiliser le modèle CNN**
```bash
python predict.py --image HAM5000\HAM500_images\ISIC_0024314.jpg --model cnn
```

**Sortie attendue :**
```
======================================================================
Analyse de l'image: HAM5000\HAM500_images\ISIC_0024314.jpg
======================================================================

Chargement du modèle CNN...
✓ CNN: Malade (confiance:  94.88%)
```


**Sortie attendue :**
```
======================================================================
Analyse de l'image: HAM5000\HAM500_images\ISIC_0024314.jpg
======================================================================

Chargement du modèle CNN...
✓ CNN: Saine (confiance:  94.88%)
Chargement du modèle VGG16...
✓ VGG16: Saine (confiance: 87.35%)
Chargement du modèle ViT-1...
✓ VIT-1: Saine (confiance: 96.03%)
Chargement du modèle ViT-2...
✓ VIT-2: Malade (confiance: 93.35%)

======================================================================
RÉSUMÉ DES PRÉDICTIONS
======================================================================
Modèle          Prédiction   Confiance    P(Malade)    P(Saine)    
----------------------------------------------------------------------
CNN             Saine            94.88%       6.22%       94.88%
VGG16           Saine            87.35%       12.65%      87.35%
VIT-1           Saine            96.03%       3.97%       96.03%
VIT-2           Malade           93.35%       6.65%       93.35%
======================================================================

```

---

##  Spécifications Techniques

### Configuration des Modèles

- **Taille d'entrée** : 64×64 pixels RGB
- **Normalisation** : ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Classes** : 
  - Classe 0 : Malade (mel - mélanome)
  - Classe 1 : Saine (autres lésions bénignes)

### Architecture CNN Personnalisé
- Bloc 1: Conv(3→32) × 2 + MaxPool
- Bloc 2: Conv(32→64) × 2 + MaxPool
- Bloc 3: Conv(64→128) × 2 + MaxPool
- Bloc 4: Conv(128→256) × 2 + MaxPool
- Bloc 5: Conv(256→512) + GlobalAvgPool
- Tête VGG16: FC(512→4096) → FC(4096→4096) → FC(4096→2)

### Architecture VGG16
- 5 blocs convolutionnels standards VGG
- Tête VGG16 commune

### Architecture ViT
- **Patches** : 8×8 (64 patches total)
- **d_model** : 64
- **Attention heads** : 4
- **MLP hidden** : 128
- **ViT-1** : 1 bloc Transformer
- **ViT-2** : 2 blocs Transformer
- Projection vers 512 → Tête VGG16

---

##  Performances

Les modèles ont été entraînés sur 8 époques avec un split 80/20 (train/validation).

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| CNN    | 0.91     | 0.0       | 0.0    | 0.0      |
| VGG16  | 0.91     | 0.0       | 0.0    | 0.0      |
| ViT-1  | 0.91     | 0.0       | 0.0    | 0.0      |
| ViT-2  | 0.91     | 0.0       | 0.0    | 0.0      |

---

##  Fonctionnalités du Script

### Arguments Disponibles

| Argument | Description | Requis | Valeurs |
|----------|-------------|--------|---------|
| `--image` | Chemin vers l'image à analyser | Oui | Chemin de fichier |
| `--model` | Modèle spécifique à utiliser | Non | cnn, vgg16, vit-1, vit-2 |
| `--all` | Utiliser tous les modèles | Non | Flag |

### Sorties du Script

1. **Prédiction individuelle** : Classe prédite + confiance
2. **Probabilités** : P(Saine) et P(Malade) pour chaque modèle
3. **Consensus** : Vote majoritaire si plusieurs modèles sont utilisés

---

### Objectifs Pédagogiques
- Implémentation de modèles de deep learning from scratch
- Comparaison d'architectures (CNN vs Transformer)
- Mise en production de modèles entraînés
- Application au domaine médical (classification de lésions)


---
