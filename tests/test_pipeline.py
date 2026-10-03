"""Check split isolation, artifact consistency and deployable inference."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, accuracy_score
from parts._pipeline import make_splits, handcrafted, read_rgb
from parts.training import NAMES, specification

ROOT=Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    def test_group_isolation(self):
        frame=pd.read_csv(ROOT/'outputs'/'dataset_manifest.csv')
        dev,test,cv=make_splits(frame)
        for a,b in [(dev,test)]+[(dev[a],dev[b]) for a,b in cv]:
            self.assertFalse(set(frame.iloc[a].split_group)&set(frame.iloc[b].split_group))
            self.assertFalse(set(frame.iloc[a].pixel_sha256)&set(frame.iloc[b].pixel_sha256))
            self.assertEqual(set(frame.iloc[a].label),set(frame.iloc[b].label))

    def test_six_saved_models_and_predictions(self):
        summary=json.loads((ROOT/'outputs'/'results.json').read_text())
        comparison=pd.read_csv(ROOT/'outputs'/'six_model_comparison.csv')
        self.assertEqual(set(comparison.model),set(NAMES.values()))
        self.assertEqual(summary['selected_model'],comparison.iloc[0].model)
        test_paths=None
        for key in NAMES:
            out=ROOT/'parts'/key/'outputs'
            bundle=joblib.load(out/f'{key}_model.joblib')
            m=json.loads((out/f'{key}_metrics.json').read_text())
            predictions=pd.read_csv(out/'test_predictions.csv')
            trials=pd.read_csv(out/'tuning_results.csv')
            self.assertEqual(bundle['dataset_fingerprint'],summary['dataset_fingerprint'])
            self.assertEqual(set(trials.param_select__percentile),{25,100})
            self.assertAlmostEqual(trials.mean_test_f1.max(),m['cv_macro_f1'])
            self.assertAlmostEqual(f1_score(predictions.label,predictions.prediction,average='macro'),m['macro_f1'])
            self.assertAlmostEqual(accuracy_score(predictions.label,predictions.prediction),m['accuracy'])
            paths=predictions.path.tolist()
            if test_paths is not None: self.assertEqual(paths,test_paths)
            test_paths=paths
            features=np.array([handcrafted(read_rgb(ROOT/'data'/'raw'/p)) for p in paths[:3]])
            self.assertEqual(bundle['estimator'].predict(features).tolist(),predictions.prediction[:3].tolist())
            pipe,grid=specification(key)
            self.assertIn('select',pipe.named_steps)

    def test_web_app_and_upload(self):
        from app import app
        client=app.test_client()
        self.assertEqual(client.get('/').status_code,200)
        self.assertEqual(client.get('/api/report').status_code,200)
        self.assertEqual(client.post('/api/predict').status_code,400)
        self.assertEqual(client.post('/api/example/0').status_code,200)
        with client.get('/download/notebook') as response:
            self.assertEqual(response.status_code,200)

    def test_cnn_holdout_and_reload(self):
        from predict import predict
        cnn=pd.read_csv(ROOT/'CNN'/'results'/'test_predictions.csv')
        shared=pd.read_csv(ROOT/'outputs'/'test_predictions.csv')
        self.assertEqual(cnn.path.tolist(),shared.path.tolist())
        self.assertEqual(cnn.y_true.tolist(),shared.label.tolist())
        m=json.loads((ROOT/'CNN'/'results'/'cnn_metrics.json').read_text())
        self.assertAlmostEqual(f1_score(cnn.y_true,cnn.y_pred,average='macro'),m['macro_f1'])
        first=cnn.iloc[0]
        self.assertEqual(predict(ROOT/'data'/'raw'/first.path,'cnn')['label'],first.y_pred)

if __name__=='__main__': unittest.main()
