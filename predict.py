"""Predict a leaf image with any of the six classical models or the separate CNN."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
from parts._pipeline import read_rgb, handcrafted, FEATURE_VERSION

ROOT = Path(__file__).resolve().parent
MODELS = ['logistic_regression','svm','knn','decision_tree','random_forest','gradient_boosting']

def predict(path, model='selected'):
    image = read_rgb(path)
    if model == 'cnn':
        import torch
        from CNN.train_cnn import BasilLeafCNN
        torch.set_num_threads(2)
        bundle = torch.load(ROOT/'CNN'/'outputs'/'cnn_model.pt', map_location='cpu', weights_only=True)
        network = BasilLeafCNN(len(bundle['classes']))
        network.load_state_dict(bundle['model_state_dict']); network.eval()
        image = image.resize((bundle['img_size'],bundle['img_size']), Image.Resampling.BILINEAR)
        arr = np.asarray(image,dtype=np.float32)/127.5-1
        with torch.no_grad():
            index = network(torch.from_numpy(arr).permute(2,0,1)[None]).argmax(1).item()
        return {'model':'BasilLeafCNN','label':bundle['classes'][index]}
    if model not in MODELS + ['selected']: raise ValueError('Unknown model')
    artifact = ROOT/'outputs'/'model.joblib' if model == 'selected' else ROOT/'parts'/model/'outputs'/f'{model}_model.joblib'
    bundle = joblib.load(artifact)
    if bundle['feature_version'] != FEATURE_VERSION: raise ValueError('Incompatible feature version')
    return {'model':bundle['name'],'label':str(bundle['estimator'].predict(handcrafted(image)[None])[0])}

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('image',type=Path)
    parser.add_argument('--model',choices=['selected',*MODELS,'cnn'],default='selected')
    args=parser.parse_args()
    print(json.dumps(predict(args.image,args.model),indent=2))
