import os
import argparse
from os import path
from pprint import pprint
from utils.s3 import s3_handler
from utils.ssm import parameter_store
from config.config import config_handler

from sagemaker.workflow.functions import Join
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.xgboost import XGBoostProcessor
from sagemaker.xgboost.model import XGBoostModel
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model_metrics import MetricsSource, ModelMetrics 
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession, LocalPipelineSession
from sagemaker.processing import ProcessingInput, ProcessingOutput, FrameworkProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo, ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.retry import StepRetryPolicy, StepExceptionTypeEnum, SageMakerJobExceptionTypeEnum, SageMakerJobStepRetryPolicy


class pipeline_tr():
    
    def __init__(self, args):
        
        self.args = args
        
        self.strRegionName = self.args.config.get_value("COMMON", "region")
        self.pm = parameter_store(self.strRegionName)
        self._env_setting()        
        
    def _env_setting(self, ):
        
        self.strPrefix = self.args.config.get_value("COMMON", "prefix")
        self.strExcutionRole = self.args.config.get_value("COMMON", "role")
        self.strBucketName = self.args.config.get_value("COMMON", "bucket")
        self.strModelName = self.args.config.get_value("COMMON", "model_name")
        self.strImageUri = self.args.config.get_value("COMMON", "image_uri")
        self.strPrepImageUri = self.args.config.get_value("COMMON", "image_uri_prep")
        self.strPipelineName = "-".join([self.strPrefix, self.strModelName])
            
        self.cache_config = CacheConfig(
            enable_caching=self.args.config.get_value("PIPELINE", "enable_caching", dtype="boolean"),
            expire_after=self.args.config.get_value("PIPELINE", "expire_after")
        )
        
        self.retry_policies=[                
            # retry when resource limit quota gets exceeded
            SageMakerJobStepRetryPolicy(
                exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
                expire_after_mins=180,
                interval_seconds=60,
                backoff_rate=1.0
            ),
        ]
        
        self.git_config = {
            'repo': f'https://{self.pm.get_params(key="-".join([self.strPrefix, "CODE-REPO"]))}',
            'branch': 'main',
            'username': self.pm.get_params(key="-".join([self.strPrefix, "CODECOMMIT-USERNAME"]), enc=True),
            'password': self.pm.get_params(key="-".join([self.strPrefix, "CODECOMMIT-PWD"]), enc=True)
        }
        
        if self.args.config.get_value("LOCALMODE", "mode", dtype="boolean"): self.pipeline_session = LocalPipelineSession()
        else: self.pipeline_session = PipelineSession()
        
        self.pm.put_params(key=self.strPrefix + "PIPELINE-NAME", value=self.strPipelineName, overwrite=True)
        
        print (f" == Envrionment parameters == ")
        print (f"   SAGEMAKER-ROLE-ARN: {self.strExcutionRole}")
        print (f"   PREFIX: {self.strPrefix}")
        print (f"   BUCKET: {self.strBucketName}")
        print (f"   IMAGE-URI: {self.strImageUri}")
        
    def _step_preprocessing(self, ):
        
        if self.args.config.get_value("LOCALMODE", "mode", dtype="boolean"):
            pipeline_session = LocalPipelineSession()
        else:
            pipeline_session = PipelineSession()
            
        strPrefixPrep = "/opt/ml/processing/"
        strDataPath = self.args.config.get_value("PREPROCESSING", "data_path")
        
        # network_config로 받으면 된다
        prep_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            image_uri=self.strPrepImageUri,
            framework_version=self.args.config.get_value("PREPROCESSING", "framework_version"),
            role=self.strExcutionRole,
            instance_type=self.args.config.get_value("PREPROCESSING", "instance_type"),
            instance_count=self.args.config.get_value("PREPROCESSING", "instance_count", dtype="int"),
            base_job_name="preprocessing", # bucket에 보이는 이름 (pipeline으로 묶으면 pipeline에서 정의한 이름으로 bucket에 보임)
            sagemaker_session=pipeline_session
        )
            
        step_args = prep_processor.run(
            code='./preprocessing.py', #소스 디렉토리 안에서 파일 path
            source_dir="./sources/preprocessing/", #현재 파일에서 소스 디렉토리 상대경로 # add processing.py and requirements.txt here
            git_config=self.git_config,
            inputs=[
                ProcessingInput(
                    input_name="input",
                    source=strDataPath,
                    destination=os.path.join(strPrefixPrep, "input")
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train-data",
                    source=os.path.join(strPrefixPrep, "output", "train"),
                    destination=os.path.join(
                        "s3://{}".format(self.strBucketName),
                        self.strPipelineName,
                        "preprocessing",
                        "output",
                        "train-data"
                    ),
                ),
                ProcessingOutput(
                    output_name="validation-data",
                    source=os.path.join(strPrefixPrep, "output", "validation"),
                    destination=os.path.join(
                        "s3://{}".format(self.strBucketName),
                        self.strPipelineName,
                        "preprocessing",
                        "output",
                        "validation-data",
                    ),
                ),
                ProcessingOutput(
                    output_name="test-data",
                    source=os.path.join(strPrefixPrep, "output", "test"),
                    destination=os.path.join(
                        "s3://{}".format(self.strBucketName),
                        self.strPipelineName,
                        "preprocessing",
                        "output",
                        "test-data",
                    ),
                )
            ],
            arguments=["--prefix_prep", strPrefixPrep, "--region", self.strRegionName],
            job_name="preprocessing",
        )
        
        self.preprocessing_process = ProcessingStep(
            name="PreprocessingProcess", ## Processing job이름
            step_args=step_args,
            cache_config=self.cache_config,
        )
        
        print ("  \n== Preprocessing Step ==")
        print ("   \nArgs: ")
        for key, value in self.preprocessing_process.arguments.items():
            print ("===========================")
            print (f'key: {key}')
            pprint (value)
            
        print (type(self.preprocessing_process.properties))
            

    def _step_training(self, ):
        
        if self.args.config.get_value("LOCALMODE", "mode", dtype="boolean"):
            pipeline_session = LocalPipelineSession()
            pipeline_session.config = {'local': {'local_code': True}}
        else:
            pipeline_session = PipelineSession()
                
        dicHyperparameters = {  
            "max_depth": "10",
            "eta": "0.3",
            "objective": "reg:squarederror",
            "num_round": "100",
        }
        
        self.estimator = XGBoost(
            entry_point="xgboost_regression.py",
            source_dir="./sources/train/", #현재 파일에서 소스 디렉토리 상대경로 # add processing.py and requirements.txt here
            git_config=self.git_config,
            hyperparameters=dicHyperparameters, ## Contatiner내 env. variable로 들어 감
            role=self.strExcutionRole,
            instance_count=self.args.config.get_value("TRAINING", "instance_count", dtype="int"),
            instance_type=self.args.config.get_value("TRAINING", "instance_type"),
            framework_version=self.args.config.get_value("TRAINING", "framework_version"),
            image_uri = self.strImageUri,
            enable_sagemaker_metrics=True,
            volume_size=64, ## GB
            output_path=os.path.join(
                "s3://{}".format(self.strBucketName),
                self.strPipelineName,
                "training",
                "output"
            ),
            base_job_name="xgboost-train",
            sagemaker_session=pipeline_session,
            #metric_definitions=listMetricDefinitions
        )
        
        step_training_args = self.estimator.fit(
            job_name="training",
            inputs={
                "TR": self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["train-data"].S3Output.S3Uri,
                "VAL": self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["validation-data"].S3Output.S3Uri,
                "TE": self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["test-data"].S3Output.S3Uri,
            },
            logs="All",
        )
          
        self.training_process = TrainingStep(
            name="TrainingProcess",
            step_args=step_training_args,
            cache_config=self.cache_config,
            #depends_on=[self.preprocessing_process],
            retry_policies=self.retry_policies
        )
        
        print ("  \n== Training Step ==")
        print ("   \nArgs: ")
        for key, value in self.training_process.arguments.items():
            print ("===========================")
            print (f'key: {key}')
            pprint (value)
        
    def _step_evaluation(self, ):
        
        #https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job-frameworks-pytorch.html
        
        if self.args.config.get_value("LOCALMODE", "mode", dtype="boolean"):
            pipeline_session = LocalPipelineSession()
        else:
            pipeline_session = PipelineSession()
            
        strPrefixPrep = "/opt/ml/processing/"
        
        #Initialize the XGBoostProcessor
        eval_processor = XGBoostProcessor(
            image_uri=self.strImageUri, 
            framework_version=self.args.config.get_value("EVALUATION", "framework_version"),
            role=self.strExcutionRole,
            instance_type=self.args.config.get_value("EVALUATION", "instance_type"),
            instance_count=self.args.config.get_value("EVALUATION", "instance_count", dtype="int"),
            base_job_name='evaluation',
            sagemaker_session=pipeline_session,
        )
                
        self.evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation-metrics",
            path="evaluation-" + self.strModelName +  ".json",
        )
        
        step_args = eval_processor.run(
            job_name="evaluation", # Evaluation job name. If not specified, the processor generates a default job name, based on the base job name and current timestamp.
                                   # 이걸 넣어야 캐시가 작동함, 안그러면 프로세서의 base_job_name 이름뒤에 날짜 시간이 붙어서 캐시 동작 안함
            code='evaluation.py', #소스 디렉토리 안에서 파일 path
            source_dir="./sources/evaluation/", #현재 파일에서 소스 디렉토리 상대경로 # add processing.py and requirements.txt here
            git_config=self.git_config,
            
            inputs=[
                ProcessingInput(
                    source=self.training_process.properties.ModelArtifacts.S3ModelArtifacts,
                    destination=os.path.join(strPrefixPrep, "model") #"/opt/ml/processing/model"
                ),
                ProcessingInput(
                    source=self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["test-data"].S3Output.S3Uri,
                    destination=os.path.join(strPrefixPrep, "test") #"/opt/ml/processing/test"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation-metrics",
                    source=os.path.join(strPrefixPrep, "evaluation"), #"/opt/ml/processing/evaluation",
                    destination=os.path.join(
                        "s3://{}".format(self.strBucketName),
                        self.strPipelineName,
                        "evaluation",
                        "evaluation-metrics",
                    ),
                )
            ],
            arguments=["--s3_model_path", self.training_process.properties.ModelArtifacts.S3ModelArtifacts, \
                       "--region", self.strRegionName, "--model_name", self.strModelName, \
                       "--prefix_eval", strPrefixPrep]
        )
        
        self.evaluation_process = ProcessingStep(
            name="EvaluationProcess", ## Processing job이름들
            step_args=step_args,
            #depends_on=[self.preprocessing_process, self.training_process],
            property_files=[self.evaluation_report],
            cache_config=self.cache_config,
            retry_policies=self.retry_policies
        )
        
        print ("  \n== Evaluation Step ==")
        print ("   \nArgs: ")
        for key, value in self.evaluation_process.arguments.items():
            print ("===========================")
            print (f'key: {key}')
            pprint (value)
        
    def _step_model_registration(self, ):
        
        self.strModelPackageGroupName = "-".join(["MPG", self.strPrefix, self.strModelName])
        self.pm.put_params(key="-".join([self.strPrefix, "MODEL-GROUP-NAME"]), value=self.strModelPackageGroupName, overwrite=True)
                                                                              
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=Join(
                    on="/",
                    values=[
                        self.evaluation_process.properties.ProcessingOutputConfig.Outputs["evaluation-metrics"].S3Output.S3Uri,
                        #print (self.evaluation_process.arguments.items())로 확인가능
                        f"evaluation-{self.strModelName}.json"
                    ],
                ),
                content_type="application/json")
        )
        
        model = XGBoostModel(
            entry_point="inference.py",
            source_dir="./sources/inference/",
            git_config=self.git_config,
            framework_version=self.args.config.get_value("MODEL_REGISTER", "framework_version"),
            code_location=os.path.join(
                "s3://",
                self.strBucketName,
                self.strPipelineName,
                "inference",
                "model"
            ),
            model_data=self.training_process.properties.ModelArtifacts.S3ModelArtifacts,
            role=self.strExcutionRole,
            image_uri=self.strImageUri,
            sagemaker_session=self.pipeline_session,
        )
        
        step_args = model.register(
            content_types=["file-path/raw-bytes", "text/csv"],
            response_types=["application/json"],
            inference_instances=self.args.config.get_value("MODEL_REGISTER", "inference_instances", dtype="list"),
            transform_instances=self.args.config.get_value("MODEL_REGISTER", "transform_instances", dtype="list"),
            model_package_group_name=self.strModelPackageGroupName,
            approval_status=self.args.config.get_value("MODEL_REGISTER", "model_approval_status_default"),
            ## “Approved”, “Rejected”, or “PendingManualApproval” (default: “PendingManualApproval”).
            model_metrics=model_metrics,
            
        )
        self.register_process = ModelStep(
            name="ModelRegisterProcess",
            step_args=step_args,
            depends_on=[self.evaluation_process]
        )
              
    def _step_fail(self, ):
            
        self.fail_process = FailStep(
            name="ConditionFail",
            error_message=Join(
                on=" ",
                values=["Execution failed due to performance threshold"]
            ),
        )
        
    def _step_condition(self, ):
        
        if self.args.config.get_value("LOCALMODE", "mode", dtype="boolean"):
            self.pipeline_session = LocalPipelineSession()
        else:
            self.pipeline_session = PipelineSession()
        
        # https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/build-and-manage-steps.html#step-type-condition
        # 조건문 종류: https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_model_building_pipeline.html#conditions
        
        self.condition_acc = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=self.evaluation_process.name,
                property_file=self.evaluation_report,
                json_path="performance_metrics.mse.value" ## evaluation.py에서 json으로 performance를 기록한 대로 한다. 
                                                               ## 즉, S3에 저장된 evaluation-<model_name>.json 파일안에 있는 값을 적어줘야 한다. 
            ),
            right=self.args.config.get_value("CONDITION", "thesh_mse", dtype="float"),
        )
        
        self.condition_prec = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=self.evaluation_process.name,
                property_file=self.evaluation_report,
                json_path="performance_metrics.rmse.value" ## evaluation.py에서 json으로 performance를 기록한 대로 한다. 
                                                           ## 즉, S3에 저장된 evaluation-<model_name>.json 파일안에 있는 값을 적어줘야 한다. 
            ),
            right=self.args.config.get_value("CONDITION", "thesh_rmse", dtype="float"),
        )
        
        self.condition_process = ConditionStep(
            name="CheckCondition",
            display_name="CheckCondition",
            conditions=[self.condition_acc, self.condition_prec], ## 여러 조건 함께 사용할 수 있음
            if_steps=[self.register_process],
            else_steps=[self.fail_process],
            #depends_on=[self.evaluation_process]
        )
        
        print ("  \n== Condition Step ==")
        print ("   \nArgs: ")
        for key, value in self.condition_process.arguments.items():
            print ("===========================")
            print (f'key: {key}')
            pprint (value)
        
    def _get_pipeline(self, ):
        
        pipeline = Pipeline(
            name=self.strPipelineName,
            #steps=[self.preprocessing_process, self.training_process, self.evaluation_process],
            steps=[self.preprocessing_process, self.training_process, self.evaluation_process, self.condition_process],
            sagemaker_session=self.pipeline_session
        )

        return pipeline
                            
    def execution(self, ):
         
        self._step_preprocessing()
        self._step_training()
        self._step_evaluation()
        self._step_model_registration()
        self._step_fail()
        self._step_condition()
        
        pipeline = self._get_pipeline()
        pipeline.upsert(role_arn=self.strExcutionRole) ## Submit the pipeline definition to the SageMaker Pipelines service 
        execution = pipeline.start()
        desc = execution.describe()
        
        self.pm.put_params(
            key="-".join([self.strPrefix, "PIPELINE-ARN"]),
            value=desc["PipelineArn"],
            overwrite=True
        )
        #print (execution.describe())

if __name__ == "__main__":
    
    strBasePath, strCurrentDir = path.dirname(path.abspath(__file__)), os.getcwd()
    os.chdir(strBasePath)
    print ("==================")
    print (f"  Working Dir: {os.getcwd()}")
    print (f"  You should execute 'mlops_pipeline.py' in 'pipeline' directory'") 
    print ("==================")
    
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    args.config = config_handler()
    
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.config.get_value("COMMON", "region")
    
    pipe_tr = pipeline_tr(args)
    pipe_tr.execution()