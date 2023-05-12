import os
import json
import argparse
import tarfile
import xgboost as xgb

class evalauator():
    
    def __init__(self, args):
                
        self.args = args
        self.strInOutPrefix = '/opt/ml/processing'
        self.strRegionName = self.args.region # boto3.Session().region_name
        
    def _load_model(self, trained_model_dir):
                
        
        model_file = "xgboost-model"
        
        t = tarfile.open(os.path.join(trained_model_dir, "model.tar.gz"), "r:gz")
        t.extractall(trained_model_dir)
        print (os.listdir(trained_model_dir))

        model = xgb.Booster()
        model.load_model(os.path.join(trained_model_dir, model_file))    
        print ("model path", trained_model_dir)
        print ("model", model)
        #pred = model.predict(dtest)
        
        return model
    
    def _evaluation_logic(self, ):
        
        ## your logic here
        ## load model from trained model artifact (os.path.join(self.args.prefix_eval, "model"))
        ## predict results using evaluation dataset (os.path.join(self.args.prefix_eval, "test"))
        ## save evaluation results (report_dict) to (os.path.join(self.args.prefix_eval, "evaluation", "evaluation-" + self.args.model_name + ".json"))
        
        ## Examples for evaluation metric
        fMSE, fRMSE, fMAE = 0.7, 0.1, 0.2  # Mean Squared Error (MSE). Root Mean Squared Error (RMSE). Mean Absolute Error (MAE)

        report_dict = {
            "performance_metrics": {
                "mse": {"value": fMSE, "standard_deviation": "NaN",},
                "rmse": {"value": fRMSE, "standard_deviation": "NaN"},
                "mae": {"value": fMAE, "standard_deviation": "NaN"},
            },
        }
        
        return report_dict
        
    def execution(self, ):
        
        ## information detail 
        print (f"Model URI: {self.args.s3_model_path}")
        print (f"Model Name: {self.args.model_name}")
        print (f"Prefix : {self.args.model_name}")
        trained_model_dir = os.path.join(self.args.prefix_eval, "model")
        print (f"trained model: {trained_model_dir}: {os.listdir(trained_model_dir)}")
        evaluation_data_dir = os.path.join(self.args.prefix_eval, "test")
        print (f"trained model: {evaluation_data_dir}: {os.listdir(evaluation_data_dir)}")
        
        ## load model
        self._load_model(trained_model_dir)
        
        ## evaluation
        report_dict = self._evaluation_logic()
        print("Evaluation report:\n{}".format(report_dict))
        
        ## save results
        evaluation_output_path = os.path.join(self.args.prefix_eval, "evaluation", "evaluation-" + self.args.model_name + ".json")
        print("Saving classification report to {}".format(evaluation_output_path))
        with open(evaluation_output_path, "w") as f:
            f.write(json.dumps(report_dict))
            
        print ("complete")


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--s3_model_path', type=str, default= "s3://")
    parser.add_argument("--region", type=str, default="ap-northeast-2")
    parser.add_argument("--model_name", type=str, default="model-default")
    parser.add_argument("--prefix_eval", type=str, default="/opt/ml/processing/")
    
    args, _ = parser.parse_known_args()
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.region
    
    
    evaluation = evalauator(args)
    evaluation.execution() 