# Projet final — Réduction de dimension & Évaluation

## Membres du groupe
- Wissal Zghrata  
- Rim Zeidan  
- Manel Lakrouz  

---

## Objectif

Comparer plusieurs méthodes de réduction de dimension (**PCA**, **t-SNE**, **UMAP**) appliquées au dataset `city_lifestyle_dataset.csv`.

L’évaluation repose sur :

- **Trustworthiness** : mesure la préservation des voisinages locaux après réduction de dimension.
- **Bonus** :
  - Accuracy d’un **kNN** sur l’embedding 2D (validation croisée stratifiée 5-fold).
  - Choix dynamique des méthodes à évaluer via CLI (`--methods`).
  - Paramétrage du nombre de voisins (`--neighbors`).
  - Activation/désactivation du kNN via `--label`.

---

## Structure du projet

```

.
├── data/
│   └── city_lifestyle_dataset.csv      # Dataset original (monté en volume)
│
├── notebooks/
│   ├── pca.ipynb
│   ├── tsne.ipynb
│   └── umap.ipynb
│
├── outputs/
│   ├── pca_emb_2d.csv
│   ├── tsne_emb_2d.csv
│   └── umap_emb_2d.csv
│
├── evaluate.py                         # Script principal d’évaluation
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md

````

⚠ Les dossiers `data/` et `outputs/` ne sont **pas inclus dans l’image Docker**.  
Ils sont montés en **volume** au moment de l’exécution.

---

## Exécution en local

Activer l’environnement virtuel :

### PowerShell
```bash
.\.venv\Scripts\Activate
````

### Git Bash

```bash
source .venv/Scripts/activate
```

Lancer l’évaluation :

```bash
python evaluate.py
```

### Options disponibles

```bash
python evaluate.py --neighbors 15
python evaluate.py --methods PCA TSNE
python evaluate.py --label country
python evaluate.py --label ""   # désactive le kNN
```

---

## Docker (bonus : volumes et mise à jour dynamique)

### Build de l’image

L’image contient uniquement :

* `evaluate.py`
* `requirements.txt`
* les dépendances Python

Elle **n’embarque pas les données ni les outputs**.

---

### Exécution (Windows + Git Bash)

```bash
WINPATH="$(pwd -W)"

docker run --rm \
  -v "${WINPATH}\\data:/app/data" \
  -v "${WINPATH}\\outputs:/app/outputs" \
  project-final-evaluate:v1
```

---

### Exemple avec options

```bash
docker run --rm \
  -v "${WINPATH}\\data:/app/data" \
  -v "${WINPATH}\\outputs:/app/outputs" \
  project-final-evaluate:v1 --neighbors 15 --methods PCA UMAP
```

---

### Bonus : modification du code sans rebuild

Il est possible de modifier `evaluate.py` sans reconstruire l’image :

```bash
WINPATH="$(pwd -W)"

docker run --rm \
  -v "${WINPATH}\\evaluate.py:/app/evaluate.py" \
  -v "${WINPATH}\\data:/app/data" \
  -v "${WINPATH}\\outputs:/app/outputs" \
  project-final-evaluate:v1
```

Cela permet :

* de modifier le code
* de changer les données
* de relancer immédiatement
  sans refaire `docker build`.
---

## Remarque sur les résultats
Les méthodes non linéaires (t-SNE, UMAP) obtiennent généralement un score supérieur à PCA, ce qui est cohérent avec leur capacité à préserver les structures locales complexes.
