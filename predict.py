"""
================================================================================
CONTRÔLE PRATIQUE - IMAGE MINING & DEEP LEARNING
ÉTAPE 2 - À DOMICILE
Script de Prédiction (predict.py)
================================================================================
Ce script charge les 4 modèles entraînés et prédit la classe d'une image.

Usage:
    python predict.py --image chemin/vers/image.jpg
    python predict.py --image chemin/vers/image.jpg --model cnn
    python predict.py --image chemin/vers/image.jpg --all
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
import sys

## ========== CONFIGURATION ==========
IMG_SIZE = 64
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## ========== DÉFINITION DES MODÈLES (identique à l'entraînement) ==========

class VGG16Head(nn.Module):
    """Tête de classification VGG16"""
    def __init__(self, input_dim=512, num_classes=2):
        super(VGG16Head, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class CustomCNN(nn.Module):
    """CNN personnalisé"""
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.head = VGG16Head(512, num_classes)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.head(x)
        return x

class VGG16(nn.Module):
    """VGG16 from scratch"""
    def __init__(self, num_classes=2):
        super(VGG16, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = VGG16Head(512, num_classes)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.adaptive_pool(x)
        x = self.head(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, d_model=64):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, d_model, 
                                    kernel_size=patch_size, 
                                    stride=patch_size)
        
    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x

class MLP(nn.Module):
    def __init__(self, d_model=64, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=4, mlp_hidden=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_hidden, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT_1(nn.Module):
    """Vision Transformer avec 1 bloc encodeur"""
    def __init__(self, img_size=64, patch_size=8, in_channels=3, 
                 d_model=64, num_heads=4, mlp_hidden=128, 
                 dropout=0.1, num_classes=2):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = TransformerEncoderBlock(d_model, num_heads, mlp_hidden, dropout)
        
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, 512)
        self.head = VGG16Head(512, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        x = self.encoder(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.projection(x)
        x = self.head(x)
        
        return x

class ViT_2(nn.Module):
    """Vision Transformer avec 2 blocs encodeurs"""
    def __init__(self, img_size=64, patch_size=8, in_channels=3, 
                 d_model=64, num_heads=4, mlp_hidden=128, 
                 dropout=0.1, num_classes=2):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.dropout = nn.Dropout(dropout)
        
        self.encoder1 = TransformerEncoderBlock(d_model, num_heads, mlp_hidden, dropout)
        self.encoder2 = TransformerEncoderBlock(d_model, num_heads, mlp_hidden, dropout)
        
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, 512)
        self.head = VGG16Head(512, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        x = self.encoder1(x)
        x = self.encoder2(x)
        
        x = self.norm(x)
        x = x[:, 0]
        x = self.projection(x)
        x = self.head(x)
        
        return x

## ========== FONCTIONS DE CHARGEMENT ==========

def load_model(model_name, model_path):
    """Charge un modèle entraîné"""
    
    # Initialiser le modèle
    if model_name == 'cnn':
        model = CustomCNN(num_classes=NUM_CLASSES)
    elif model_name == 'vgg16':
        model = VGG16(num_classes=NUM_CLASSES)
    elif model_name == 'vit-1':
        model = ViT_1(img_size=IMG_SIZE, patch_size=8, d_model=64,
                      num_heads=4, mlp_hidden=128, dropout=0.1, 
                      num_classes=NUM_CLASSES)
    elif model_name == 'vit-2':
        model = ViT_2(img_size=IMG_SIZE, patch_size=8, d_model=64,
                      num_heads=4, mlp_hidden=128, dropout=0.1, 
                      num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Modèle inconnu: {model_name}")
    
    # Charger les poids (avec weights_only=False pour compatibilité PyTorch 2.6+)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(DEVICE)
    model.eval()
    
    return model

def preprocess_image(image_path):
    """Prétraiter l'image pour la prédiction"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Ajouter dimension batch
    
    return image

def predict(model, image_tensor):
    """Faire une prédiction"""
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    # Classe 0 = Malade, Classe 1 = Saine
    class_names = ['Malade', 'Saine']
    predicted_class = class_names[predicted.item()]
    confidence_value = confidence.item() * 100
    
    return predicted_class, confidence_value, probabilities[0].cpu().numpy()

## ========== FONCTION PRINCIPALE ==========

def main():
    parser = argparse.ArgumentParser(description='Prédiction de lésions cutanées')
    parser.add_argument('--image', type=str, required=True, 
                       help='Chemin vers l\'image à prédire')
    parser.add_argument('--model', type=str, choices=['cnn', 'vgg16', 'vit-1', 'vit-2'], 
                       help='Modèle à utiliser (défaut: tous)')
    parser.add_argument('--all', action='store_true', 
                       help='Utiliser tous les modèles')
    
    args = parser.parse_args()
    
    # Vérifier que l'image existe
    if not os.path.exists(args.image):
        print(f"❌ Erreur: L'image {args.image} n'existe pas")
        sys.exit(1)
    
    # Prétraiter l'image
    print(f"\n{'='*70}")
    print(f"Analyse de l'image: {args.image}")
    print(f"{'='*70}\n")
    
    image_tensor = preprocess_image(args.image)
    
    # Déterminer quels modèles utiliser
    if args.all or args.model is None:
        models_to_use = ['cnn', 'vgg16', 'vit-1', 'vit-2']
    else:
        models_to_use = [args.model]
    
    # Faire les prédictions
    results = []
    
    for model_name in models_to_use:
        model_path = f'D:/AIT_IDAR_Abdelaali_MLAIM/{model_name}_final.pth'
        
        if not os.path.exists(model_path):
            print(f"  Modèle {model_name} non trouvé: {model_path}")
            continue
        
        print(f"Chargement du modèle {model_name.upper()}...")
        model = load_model(model_name, model_path)
        
        predicted_class, confidence, probs = predict(model, image_tensor)
        
        results.append({
            'model': model_name.upper(),
            'prediction': predicted_class,
            'confidence': confidence,
            'prob_malade': probs[0] * 100,
            'prob_saine': probs[1] * 100
        })
        
        print(f" {model_name.upper()}: {predicted_class} (confiance: {confidence:.2f}%)")
    
    # Afficher un résumé
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("RÉSUMÉ DES PRÉDICTIONS")
        print(f"{'='*70}")
        print(f"{'Modèle':<15} {'Prédiction':<12} {'Confiance':<12} {'P(Saine)':<12} {'P(Malade)':<12}")
        print(f"{'-'*70}")
        
        for r in results:
            print(f"{r['model']:<15} {r['prediction']:<12} {r['confidence']:>10.2f}% "
                  f"{r['prob_saine']:>10.2f}% {r['prob_malade']:>10.2f}%")
        
        print(f"{'='*70}\n")
        
        # Consensus
        predictions = [r['prediction'] for r in results]
        if predictions.count('Malade') > len(predictions) / 2:
            consensus = 'Malade'
        else:
            consensus = 'Saine'
        
        print(f" CONSENSUS: {consensus}")
        print(f"   ({predictions.count(consensus)}/{len(predictions)} modèles d'accord)\n")

if __name__ == '__main__':
    main()