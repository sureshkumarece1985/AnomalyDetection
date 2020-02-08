```python
import boto3
import botocore
import sagemaker
import sys


bucket = 'sureshtest2067'   # <--- specify a bucket you have access to
prefix = 'sagemaker/rcf-benchmarks'
execution_role = sagemaker.get_execution_role()


# check if the bucket exists
try:
    boto3.Session().client('s3').head_bucket(Bucket=bucket)
except botocore.exceptions.ParamValidationError as e:
    print('Hey! You either forgot to specify your S3 bucket'
          ' or you gave your bucket an invalid name!')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == '403':
        print("Hey! You don't have permission to access the bucket, {}.".format(bucket))
    elif e.response['Error']['Code'] == '404':
        print("Hey! Your bucket, {}, doesn't exist!".format(bucket))
    else:
        raise
else:
    print('Training input/output will be stored in: s3://{}/{}'.format(bucket, prefix))
```

    Training input/output will be stored in: s3://sureshtest2067/sagemaker/rcf-benchmarks



```python
%%time

import pandas as pd
import urllib.request

#data_filename = 'nyc_taxi.csv'
#data_source = 'https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv'

data_filename = 'elb_request_count_8c0756.csv'
data_source = 'https://github.com/sureshkumarece1985/AnomalyDetection/raw/master/elb_request_count_8c0756.csv'

urllib.request.urlretrieve(data_source, data_filename)
taxi_data = pd.read_csv(data_filename, delimiter=',')
```

    CPU times: user 27 ms, sys: 0 ns, total: 27 ms
    Wall time: 320 ms



```python
taxi_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-04-10 00:04:00</td>
      <td>94.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-04-10 00:09:00</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-04-10 00:14:00</td>
      <td>187.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-04-10 00:19:00</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-04-10 00:24:00</td>
      <td>51.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 100

taxi_data.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f56a5359438>




![png](output_3_1.png)



```python
taxi_data[3600:3800].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f56a5ceee48>




![png](output_4_1.png)



```python
taxi_data[3675:4675]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3675</th>
      <td>2014-04-22 18:59:00</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>3676</th>
      <td>2014-04-22 19:04:00</td>
      <td>136.0</td>
    </tr>
    <tr>
      <th>3677</th>
      <td>2014-04-22 19:09:00</td>
      <td>116.0</td>
    </tr>
    <tr>
      <th>3678</th>
      <td>2014-04-22 19:14:00</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>3679</th>
      <td>2014-04-22 19:19:00</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>3680</th>
      <td>2014-04-22 19:24:00</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>3681</th>
      <td>2014-04-22 19:29:00</td>
      <td>175.0</td>
    </tr>
    <tr>
      <th>3682</th>
      <td>2014-04-22 19:34:00</td>
      <td>656.0</td>
    </tr>
    <tr>
      <th>3683</th>
      <td>2014-04-22 19:39:00</td>
      <td>256.0</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>2014-04-22 19:44:00</td>
      <td>195.0</td>
    </tr>
    <tr>
      <th>3685</th>
      <td>2014-04-22 19:49:00</td>
      <td>338.0</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>2014-04-22 19:54:00</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>3687</th>
      <td>2014-04-22 19:59:00</td>
      <td>145.0</td>
    </tr>
    <tr>
      <th>3688</th>
      <td>2014-04-22 20:04:00</td>
      <td>173.0</td>
    </tr>
    <tr>
      <th>3689</th>
      <td>2014-04-22 20:09:00</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>3690</th>
      <td>2014-04-22 20:14:00</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>3691</th>
      <td>2014-04-22 20:19:00</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>3692</th>
      <td>2014-04-22 20:24:00</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>3693</th>
      <td>2014-04-22 20:29:00</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>3694</th>
      <td>2014-04-22 20:34:00</td>
      <td>131.0</td>
    </tr>
    <tr>
      <th>3695</th>
      <td>2014-04-22 20:39:00</td>
      <td>235.0</td>
    </tr>
    <tr>
      <th>3696</th>
      <td>2014-04-22 20:44:00</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>3697</th>
      <td>2014-04-22 20:49:00</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>3698</th>
      <td>2014-04-22 20:54:00</td>
      <td>136.0</td>
    </tr>
    <tr>
      <th>3699</th>
      <td>2014-04-22 20:59:00</td>
      <td>239.0</td>
    </tr>
    <tr>
      <th>3700</th>
      <td>2014-04-22 21:04:00</td>
      <td>61.0</td>
    </tr>
    <tr>
      <th>3701</th>
      <td>2014-04-22 21:09:00</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>3702</th>
      <td>2014-04-22 21:14:00</td>
      <td>135.0</td>
    </tr>
    <tr>
      <th>3703</th>
      <td>2014-04-22 21:19:00</td>
      <td>102.0</td>
    </tr>
    <tr>
      <th>3704</th>
      <td>2014-04-22 21:24:00</td>
      <td>112.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4002</th>
      <td>2014-04-23 22:14:00</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>4003</th>
      <td>2014-04-23 22:19:00</td>
      <td>164.0</td>
    </tr>
    <tr>
      <th>4004</th>
      <td>2014-04-23 22:24:00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>4005</th>
      <td>2014-04-23 22:29:00</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>4006</th>
      <td>2014-04-23 22:34:00</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4007</th>
      <td>2014-04-23 22:39:00</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>4008</th>
      <td>2014-04-23 22:44:00</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4009</th>
      <td>2014-04-23 22:49:00</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>4010</th>
      <td>2014-04-23 22:54:00</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>4011</th>
      <td>2014-04-23 22:59:00</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>4012</th>
      <td>2014-04-23 23:04:00</td>
      <td>93.0</td>
    </tr>
    <tr>
      <th>4013</th>
      <td>2014-04-23 23:09:00</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>4014</th>
      <td>2014-04-23 23:14:00</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4015</th>
      <td>2014-04-23 23:19:00</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>4016</th>
      <td>2014-04-23 23:24:00</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>4017</th>
      <td>2014-04-23 23:29:00</td>
      <td>115.0</td>
    </tr>
    <tr>
      <th>4018</th>
      <td>2014-04-23 23:34:00</td>
      <td>261.0</td>
    </tr>
    <tr>
      <th>4019</th>
      <td>2014-04-23 23:39:00</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>4020</th>
      <td>2014-04-23 23:44:00</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4021</th>
      <td>2014-04-23 23:49:00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>4022</th>
      <td>2014-04-23 23:54:00</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>4023</th>
      <td>2014-04-23 23:59:00</td>
      <td>182.0</td>
    </tr>
    <tr>
      <th>4024</th>
      <td>2014-04-24 00:04:00</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>4025</th>
      <td>2014-04-24 00:09:00</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>4026</th>
      <td>2014-04-24 00:14:00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4027</th>
      <td>2014-04-24 00:19:00</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>4028</th>
      <td>2014-04-24 00:24:00</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>4029</th>
      <td>2014-04-24 00:29:00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>4030</th>
      <td>2014-04-24 00:34:00</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>4031</th>
      <td>2014-04-24 00:39:00</td>
      <td>60.0</td>
    </tr>
  </tbody>
</table>
<p>357 rows × 2 columns</p>
</div>




```python
from sagemaker import RandomCutForest

session = sagemaker.Session()

# specify general training job information
rcf = RandomCutForest(role=execution_role,
                      train_instance_count=1,
                      train_instance_type='ml.m4.xlarge',
                      data_location='s3://{}/{}/'.format(bucket, prefix),
                      output_path='s3://{}/{}/output'.format(bucket, prefix),
                      num_samples_per_tree=512,
                      num_trees=50)

# automatically upload the training data to S3 and run the training job
rcf.fit(rcf.record_set(taxi_data.value.as_matrix().reshape(-1,1)))
```

    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/ipykernel/__main__.py:15: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.


    2020-02-08 06:52:06 Starting - Starting the training job...
    2020-02-08 06:52:07 Starting - Launching requested ML instances...
    2020-02-08 06:53:04 Starting - Preparing the instances for training......
    2020-02-08 06:53:58 Downloading - Downloading input data...
    2020-02-08 06:54:22 Training - Downloading the training image...
    2020-02-08 06:55:02 Uploading - Uploading generated training model[34mDocker entrypoint called with argument(s): train[0m
    [34m/opt/amazon/lib/python2.7/site-packages/scipy/_lib/_numpy_compat.py:10: DeprecationWarning: Importing from numpy.testing.nosetester is deprecated, import from numpy.testing instead.
      from numpy.testing.nosetester import import_nose[0m
    [34m/opt/amazon/lib/python2.7/site-packages/scipy/stats/morestats.py:12: DeprecationWarning: Importing from numpy.testing.decorators is deprecated, import from numpy.testing instead.
      from numpy.testing.decorators import setastest[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/resources/default-conf.json: {u'_ftp_port': 8999, u'num_samples_per_tree': 256, u'_tuning_objective_metric': u'', u'_num_gpus': u'auto', u'_log_level': u'info', u'_kvstore': u'dist_async', u'force_dense': u'true', u'epochs': 1, u'num_trees': 100, u'eval_metrics': [u'accuracy', u'precision_recall_fscore'], u'_num_kv_servers': u'auto', u'mini_batch_size': 1000}[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'mini_batch_size': u'1000', u'feature_dim': u'1', u'num_samples_per_tree': u'512', u'num_trees': u'50'}[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Final configuration: {u'_ftp_port': 8999, u'num_samples_per_tree': u'512', u'_tuning_objective_metric': u'', u'_num_gpus': u'auto', u'_log_level': u'info', u'_kvstore': u'dist_async', u'force_dense': u'true', u'epochs': 1, u'feature_dim': u'1', u'num_trees': u'50', u'eval_metrics': [u'accuracy', u'precision_recall_fscore'], u'_num_kv_servers': u'auto', u'mini_batch_size': u'1000'}[0m
    [34m[02/08/2020 06:54:59 WARNING 140502373369664] Loggers have already been setup.[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Launching parameter server for role scheduler[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] {'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/e21e10b1-d400-4fc1-a79d-9813d23bd6e3', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'MXNET_KVSTORE_BIGARRAY_BOUND': '400000000', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'AWS_REGION': 'us-east-2', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'randomcutforest-2020-02-08-06-52-06-063', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'SAGEMAKER_METRICS_DIRECTORY': '/opt/ml/output/metrics/sagemaker', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-107-107.us-east-2.compute.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/0e444141-07f7-4732-9c49-fab8489db1d6', 'PWD': '/', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-2:234268085836:training-job/randomcutforest-2020-02-08-06-52-06-063', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] envs={'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/e21e10b1-d400-4fc1-a79d-9813d23bd6e3', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'DMLC_NUM_WORKER': '1', 'DMLC_PS_ROOT_PORT': '9000', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'MXNET_KVSTORE_BIGARRAY_BOUND': '400000000', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'DMLC_PS_ROOT_URI': '10.0.107.107', 'AWS_REGION': 'us-east-2', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'randomcutforest-2020-02-08-06-52-06-063', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'SAGEMAKER_METRICS_DIRECTORY': '/opt/ml/output/metrics/sagemaker', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-107-107.us-east-2.compute.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/0e444141-07f7-4732-9c49-fab8489db1d6', 'DMLC_ROLE': 'scheduler', 'PWD': '/', 'DMLC_NUM_SERVER': '1', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-2:234268085836:training-job/randomcutforest-2020-02-08-06-52-06-063', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Launching parameter server for role server[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] {'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/e21e10b1-d400-4fc1-a79d-9813d23bd6e3', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'MXNET_KVSTORE_BIGARRAY_BOUND': '400000000', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'AWS_REGION': 'us-east-2', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'randomcutforest-2020-02-08-06-52-06-063', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'SAGEMAKER_METRICS_DIRECTORY': '/opt/ml/output/metrics/sagemaker', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-107-107.us-east-2.compute.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/0e444141-07f7-4732-9c49-fab8489db1d6', 'PWD': '/', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-2:234268085836:training-job/randomcutforest-2020-02-08-06-52-06-063', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] envs={'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/e21e10b1-d400-4fc1-a79d-9813d23bd6e3', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'DMLC_NUM_WORKER': '1', 'DMLC_PS_ROOT_PORT': '9000', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'SAGEMAKER_HTTP_PORT': '8080', 'HOME': '/root', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'MXNET_KVSTORE_BIGARRAY_BOUND': '400000000', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'DMLC_PS_ROOT_URI': '10.0.107.107', 'AWS_REGION': 'us-east-2', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'randomcutforest-2020-02-08-06-52-06-063', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'SAGEMAKER_METRICS_DIRECTORY': '/opt/ml/output/metrics/sagemaker', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-107-107.us-east-2.compute.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/0e444141-07f7-4732-9c49-fab8489db1d6', 'DMLC_ROLE': 'server', 'PWD': '/', 'DMLC_NUM_SERVER': '1', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-2:234268085836:training-job/randomcutforest-2020-02-08-06-52-06-063', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Environment: {'ECS_CONTAINER_METADATA_URI': 'http://169.254.170.2/v3/e21e10b1-d400-4fc1-a79d-9813d23bd6e3', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION': '2', 'DMLC_PS_ROOT_PORT': '9000', 'DMLC_NUM_WORKER': '1', 'SAGEMAKER_HTTP_PORT': '8080', 'PATH': '/opt/amazon/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/bin:/opt/amazon/bin', 'PYTHONUNBUFFERED': 'TRUE', 'CANONICAL_ENVROOT': '/opt/amazon', 'LD_LIBRARY_PATH': '/opt/amazon/lib/python2.7/site-packages/cv2/../../../../lib:/usr/local/nvidia/lib64:/opt/amazon/lib', 'MXNET_KVSTORE_BIGARRAY_BOUND': '400000000', 'LANG': 'en_US.utf8', 'DMLC_INTERFACE': 'eth0', 'SHLVL': '1', 'DMLC_PS_ROOT_URI': '10.0.107.107', 'AWS_REGION': 'us-east-2', 'SAGEMAKER_METRICS_DIRECTORY': '/opt/ml/output/metrics/sagemaker', 'NVIDIA_VISIBLE_DEVICES': 'void', 'TRAINING_JOB_NAME': 'randomcutforest-2020-02-08-06-52-06-063', 'HOME': '/root', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp', 'ENVROOT': '/opt/amazon', 'SAGEMAKER_DATA_PATH': '/opt/ml', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'NVIDIA_REQUIRE_CUDA': 'cuda>=9.0', 'OMP_NUM_THREADS': '2', 'HOSTNAME': 'ip-10-0-107-107.us-east-2.compute.internal', 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI': '/v2/credentials/0e444141-07f7-4732-9c49-fab8489db1d6', 'DMLC_ROLE': 'worker', 'PWD': '/', 'DMLC_NUM_SERVER': '1', 'TRAINING_JOB_ARN': 'arn:aws:sagemaker:us-east-2:234268085836:training-job/randomcutforest-2020-02-08-06-52-06-063', 'AWS_EXECUTION_ENV': 'AWS_ECS_EC2'}[0m
    [34mProcess 32 is a shell:scheduler.[0m
    [34mProcess 33 is a shell:server.[0m
    [34mProcess 1 is a worker.[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Using default worker.[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Loaded iterator creator application/x-recordio-protobuf for content type ('application/x-recordio-protobuf', '1.0')[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Verifying hyperparamemters...[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Hyperparameters are correct.[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Validating that feature_dim agrees with dimensions in training data...[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] feature_dim is correct.[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Validating memory limits...[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Available memory in bytes: 15323021312[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Estimated sample size in bytes: 409600[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Estimated memory needed to build the forest in bytes: 1024000[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Memory limits validated.[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Starting cluster sharing facilities...[0m
    [34m[02/08/2020 06:54:59 INFO 140502373369664] Create Store: dist_async[0m
    [34m[I 20-02-08 06:54:59] >>> starting FTP server on 0.0.0.0:8999, pid=1 <<<[0m
    [34m[02/08/2020 06:54:59 INFO 140500896036608] >>> starting FTP server on 0.0.0.0:8999, pid=1 <<<[0m
    [34m[I 20-02-08 06:54:59] poller: <class 'pyftpdlib.ioloop.Epoll'>[0m
    [34m[02/08/2020 06:54:59 INFO 140500896036608] poller: <class 'pyftpdlib.ioloop.Epoll'>[0m
    [34m[I 20-02-08 06:54:59] masquerade (NAT) address: None[0m
    [34m[02/08/2020 06:54:59 INFO 140500896036608] masquerade (NAT) address: None[0m
    [34m[I 20-02-08 06:54:59] passive ports: None[0m
    [34m[02/08/2020 06:54:59 INFO 140500896036608] passive ports: None[0m
    [34m[I 20-02-08 06:54:59] use sendfile(2): False[0m
    [34m[02/08/2020 06:54:59 INFO 140500896036608] use sendfile(2): False[0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] Cluster sharing facilities started.[0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] Verifying all workers are accessible...[0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] All workers accessible.[0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] Initializing Sampler...[0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] Sampler correctly initialized.[0m
    [34m#metrics {"Metrics": {"initialize.time": {"count": 1, "max": 662.7261638641357, "sum": 662.7261638641357, "min": 662.7261638641357}}, "EndTime": 1581144900.034806, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "RandomCutForest"}, "StartTime": 1581144899.367679}
    [0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Batches Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Records Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Batches Seen": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Records Seen": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Max Records Seen Between Resets": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Reset Count": {"count": 1, "max": 0, "sum": 0.0, "min": 0}}, "EndTime": 1581144900.035065, "Dimensions": {"Host": "algo-1", "Meta": "init_train_data_iter", "Operation": "training", "Algorithm": "RandomCutForest"}, "StartTime": 1581144900.034997}
    [0m
    [34m[2020-02-08 06:55:00.035] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 0, "duration": 666, "num_examples": 1, "num_bytes": 32000}[0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] Sampling training data...[0m
    [34m[2020-02-08 06:55:00.052] [tensorio] [info] epoch_stats={"data_pipeline": "/opt/ml/input/data/train", "epoch": 1, "duration": 16, "num_examples": 5, "num_bytes": 129024}[0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] Sampling training data completed.[0m
    [34m#metrics {"Metrics": {"epochs": {"count": 1, "max": 1, "sum": 1.0, "min": 1}, "update.time": {"count": 1, "max": 19.932985305786133, "sum": 19.932985305786133, "min": 19.932985305786133}}, "EndTime": 1581144900.055644, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "RandomCutForest"}, "StartTime": 1581144900.034928}
    [0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] Early stop condition met. Stopping training.[0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] #progress_metric: host=algo-1, completed 100 % epochs[0m
    [34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 5, "sum": 5.0, "min": 5}, "Number of Batches Since Last Reset": {"count": 1, "max": 5, "sum": 5.0, "min": 5}, "Number of Records Since Last Reset": {"count": 1, "max": 4032, "sum": 4032.0, "min": 4032}, "Total Batches Seen": {"count": 1, "max": 5, "sum": 5.0, "min": 5}, "Total Records Seen": {"count": 1, "max": 4032, "sum": 4032.0, "min": 4032}, "Max Records Seen Between Resets": {"count": 1, "max": 4032, "sum": 4032.0, "min": 4032}, "Reset Count": {"count": 1, "max": 1, "sum": 1.0, "min": 1}}, "EndTime": 1581144900.056141, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "RandomCutForest", "epoch": 0}, "StartTime": 1581144900.03558}
    [0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] #throughput_metric: host=algo-1, train throughput=194856.879651 records/second[0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] Master node: building Random Cut Forest...[0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] Gathering samples...[0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] 4032 samples gathered[0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] Building Random Cut Forest...[0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] Random Cut Forest built: 
    [0m
    [34mForestInfo{num_trees: 50, num_samples_in_forest: 4000, num_samples_per_tree: 80, sample_dim: 1, shingle_size: 1, trees_num_nodes: [107, 129, 125, 119, 109, 117, 135, 123, 103, 127, 119, 89, 93, 123, 105, 101, 137, 125, 119, 129, 123, 123, 113, 123, 139, 127, 119, 121, 105, 113, 105, 121, 95, 93, 107, 97, 87, 93, 111, 107, 103, 121, 119, 121, 101, 121, 123, 113, 121, 123, ], trees_depth: [11, 13, 11, 17, 14, 12, 13, 16, 11, 17, 11, 10, 12, 15, 12, 12, 10, 11, 12, 13, 12, 13, 14, 11, 18, 13, 14, 12, 11, 12, 12, 10, 17, 14, 12, 11, 14, 12, 11, 11, 15, 12, 13, 15, 12, 13, 12, 13, 12, 17, ], max_num_nodes: 139, min_num_nodes: 87, avg_num_nodes: 114, max_tree_depth: 18, min_tree_depth: 10, avg_tree_depth: 12, mem_size: 595536}[0m
    [34m#metrics {"Metrics": {"finalize.time": {"count": 1, "max": 8.117914199829102, "sum": 8.117914199829102, "min": 8.117914199829102}, "model.bytes": {"count": 1, "max": 595536, "sum": 595536.0, "min": 595536}, "fit_model.time": {"count": 1, "max": 3.2820701599121094, "sum": 3.2820701599121094, "min": 3.2820701599121094}}, "EndTime": 1581144900.064541, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "RandomCutForest"}, "StartTime": 1581144900.055732}
    [0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] Master node: Serializing the RandomCutForest model[0m
    [34m#metrics {"Metrics": {"serialize_model.time": {"count": 1, "max": 11.003971099853516, "sum": 11.003971099853516, "min": 11.003971099853516}}, "EndTime": 1581144900.075679, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "RandomCutForest"}, "StartTime": 1581144900.064629}
    [0m
    [34m[02/08/2020 06:55:00 INFO 140502373369664] Test data is not provided.[0m
    [34m[I 20-02-08 06:55:00] >>> shutting down FTP server (0 active fds) <<<[0m
    [34m[02/08/2020 06:55:00 INFO 140500896036608] >>> shutting down FTP server (0 active fds) <<<[0m
    [34m#metrics {"Metrics": {"totaltime": {"count": 1, "max": 914.3290519714355, "sum": 914.3290519714355, "min": 914.3290519714355}, "setuptime": {"count": 1, "max": 195.8479881286621, "sum": 195.8479881286621, "min": 195.8479881286621}}, "EndTime": 1581144900.079008, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "RandomCutForest"}, "StartTime": 1581144900.075737}
    [0m
    
    2020-02-08 06:55:08 Completed - Training job completed
    Training seconds: 70
    Billable seconds: 70



```python
print('Training job name: {}'.format(rcf.latest_training_job.job_name))
```

    Training job name: randomcutforest-2020-02-08-06-52-06-063



```python

rcf_inference = rcf.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge',
)
```

    -------------!


```python
print('Endpoint name: {}'.format(rcf_inference.endpoint))
```

    Endpoint name: randomcutforest-2020-02-08-06-52-06-063



```python
from sagemaker.predictor import csv_serializer, json_deserializer

rcf_inference.content_type = 'text/csv'
rcf_inference.serializer = csv_serializer
rcf_inference.accept = 'application/json'
rcf_inference.deserializer = json_deserializer
```


```python
taxi_data_numpy = taxi_data.value.as_matrix().reshape(-1,1)
print(taxi_data_numpy[:6])
results = rcf_inference.predict(taxi_data_numpy[:6])
```

    [[ 94.]
     [ 56.]
     [187.]
     [ 95.]
     [ 51.]
     [ 10.]]


    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/ipykernel/__main__.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      if __name__ == '__main__':



```python

results = rcf_inference.predict(taxi_data_numpy)
scores = [datum['score'] for datum in results['scores']]

# add scores to taxi data frame and print first few values
taxi_data['score'] = pd.Series(scores, index=taxi_data.index)
taxi_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>value</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-04-10 00:04:00</td>
      <td>94.0</td>
      <td>0.865573</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-04-10 00:09:00</td>
      <td>56.0</td>
      <td>0.786471</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-04-10 00:14:00</td>
      <td>187.0</td>
      <td>1.613852</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-04-10 00:19:00</td>
      <td>95.0</td>
      <td>0.866955</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-04-10 00:24:00</td>
      <td>51.0</td>
      <td>0.770862</td>
    </tr>
  </tbody>
</table>
</div>




```python

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

#
# *Try this out* - change `start` and `end` to zoom in on the 
# anomaly found earlier in this notebook
#
start, end = 0, len(taxi_data)
#start, end = 5500, 6500
taxi_data_subset = taxi_data[start:end]

ax1.plot(taxi_data_subset['value'], color='C0', alpha=0.8)
ax2.plot(taxi_data_subset['score'], color='C1')

ax1.grid(which='major', axis='both')

ax1.set_ylabel('Taxi Ridership', color='C0')
ax2.set_ylabel('Anomaly Score', color='C1')

ax1.tick_params('y', colors='C0')
ax2.tick_params('y', colors='C1')

ax1.set_ylim(0, 40000)
ax2.set_ylim(min(scores), 1.4*max(scores))
fig.set_figwidth(10)
```


![png](output_13_0.png)



```python
score_mean = taxi_data['score'].mean()
score_std = taxi_data['score'].std()
score_cutoff = score_mean + 3*score_std

anomalies = taxi_data_subset[taxi_data_subset['score'] > score_cutoff]
anomalies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>value</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>84</th>
      <td>2014-04-10 07:04:00</td>
      <td>222.0</td>
      <td>2.159113</td>
    </tr>
    <tr>
      <th>142</th>
      <td>2014-04-10 11:59:00</td>
      <td>255.0</td>
      <td>2.594596</td>
    </tr>
    <tr>
      <th>176</th>
      <td>2014-04-10 14:49:00</td>
      <td>232.0</td>
      <td>2.267812</td>
    </tr>
    <tr>
      <th>193</th>
      <td>2014-04-10 16:14:00</td>
      <td>335.0</td>
      <td>3.586372</td>
    </tr>
    <tr>
      <th>197</th>
      <td>2014-04-10 16:34:00</td>
      <td>264.0</td>
      <td>2.710799</td>
    </tr>
    <tr>
      <th>219</th>
      <td>2014-04-10 18:24:00</td>
      <td>303.0</td>
      <td>3.197637</td>
    </tr>
    <tr>
      <th>270</th>
      <td>2014-04-10 22:39:00</td>
      <td>209.0</td>
      <td>2.017933</td>
    </tr>
    <tr>
      <th>304</th>
      <td>2014-04-11 01:29:00</td>
      <td>237.0</td>
      <td>2.381542</td>
    </tr>
    <tr>
      <th>392</th>
      <td>2014-04-11 08:49:00</td>
      <td>203.0</td>
      <td>1.910654</td>
    </tr>
    <tr>
      <th>398</th>
      <td>2014-04-11 09:19:00</td>
      <td>252.0</td>
      <td>2.557541</td>
    </tr>
    <tr>
      <th>438</th>
      <td>2014-04-11 12:39:00</td>
      <td>266.0</td>
      <td>2.729446</td>
    </tr>
    <tr>
      <th>453</th>
      <td>2014-04-11 13:54:00</td>
      <td>220.0</td>
      <td>2.150669</td>
    </tr>
    <tr>
      <th>487</th>
      <td>2014-04-11 16:44:00</td>
      <td>280.0</td>
      <td>2.898184</td>
    </tr>
    <tr>
      <th>509</th>
      <td>2014-04-11 18:34:00</td>
      <td>226.0</td>
      <td>2.193046</td>
    </tr>
    <tr>
      <th>521</th>
      <td>2014-04-11 19:34:00</td>
      <td>237.0</td>
      <td>2.381542</td>
    </tr>
    <tr>
      <th>531</th>
      <td>2014-04-11 20:24:00</td>
      <td>259.0</td>
      <td>2.653464</td>
    </tr>
    <tr>
      <th>532</th>
      <td>2014-04-11 20:29:00</td>
      <td>272.0</td>
      <td>2.830559</td>
    </tr>
    <tr>
      <th>564</th>
      <td>2014-04-11 23:09:00</td>
      <td>335.0</td>
      <td>3.586372</td>
    </tr>
    <tr>
      <th>572</th>
      <td>2014-04-11 23:49:00</td>
      <td>231.0</td>
      <td>2.254257</td>
    </tr>
    <tr>
      <th>580</th>
      <td>2014-04-12 00:29:00</td>
      <td>272.0</td>
      <td>2.830559</td>
    </tr>
    <tr>
      <th>740</th>
      <td>2014-04-12 13:49:00</td>
      <td>294.0</td>
      <td>3.102726</td>
    </tr>
    <tr>
      <th>782</th>
      <td>2014-04-12 17:19:00</td>
      <td>288.0</td>
      <td>3.019603</td>
    </tr>
    <tr>
      <th>785</th>
      <td>2014-04-12 17:34:00</td>
      <td>381.0</td>
      <td>3.951076</td>
    </tr>
    <tr>
      <th>789</th>
      <td>2014-04-12 17:54:00</td>
      <td>283.0</td>
      <td>2.933365</td>
    </tr>
    <tr>
      <th>790</th>
      <td>2014-04-12 17:59:00</td>
      <td>381.0</td>
      <td>3.951076</td>
    </tr>
    <tr>
      <th>924</th>
      <td>2014-04-13 05:14:00</td>
      <td>261.0</td>
      <td>2.702431</td>
    </tr>
    <tr>
      <th>1036</th>
      <td>2014-04-13 14:34:00</td>
      <td>212.0</td>
      <td>2.105080</td>
    </tr>
    <tr>
      <th>1040</th>
      <td>2014-04-13 14:54:00</td>
      <td>210.0</td>
      <td>2.011637</td>
    </tr>
    <tr>
      <th>1068</th>
      <td>2014-04-13 17:14:00</td>
      <td>203.0</td>
      <td>1.910654</td>
    </tr>
    <tr>
      <th>1138</th>
      <td>2014-04-13 23:04:00</td>
      <td>220.0</td>
      <td>2.150669</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3061</th>
      <td>2014-04-20 15:49:00</td>
      <td>284.0</td>
      <td>2.951369</td>
    </tr>
    <tr>
      <th>3079</th>
      <td>2014-04-20 17:19:00</td>
      <td>217.0</td>
      <td>2.098308</td>
    </tr>
    <tr>
      <th>3108</th>
      <td>2014-04-20 19:44:00</td>
      <td>220.0</td>
      <td>2.150669</td>
    </tr>
    <tr>
      <th>3319</th>
      <td>2014-04-21 13:19:00</td>
      <td>234.0</td>
      <td>2.343383</td>
    </tr>
    <tr>
      <th>3370</th>
      <td>2014-04-21 17:34:00</td>
      <td>234.0</td>
      <td>2.343383</td>
    </tr>
    <tr>
      <th>3387</th>
      <td>2014-04-21 18:59:00</td>
      <td>222.0</td>
      <td>2.159113</td>
    </tr>
    <tr>
      <th>3411</th>
      <td>2014-04-21 20:59:00</td>
      <td>219.0</td>
      <td>2.122204</td>
    </tr>
    <tr>
      <th>3419</th>
      <td>2014-04-21 21:39:00</td>
      <td>330.0</td>
      <td>3.560200</td>
    </tr>
    <tr>
      <th>3420</th>
      <td>2014-04-21 21:44:00</td>
      <td>204.0</td>
      <td>1.916887</td>
    </tr>
    <tr>
      <th>3447</th>
      <td>2014-04-21 23:59:00</td>
      <td>245.0</td>
      <td>2.424295</td>
    </tr>
    <tr>
      <th>3457</th>
      <td>2014-04-22 00:49:00</td>
      <td>208.0</td>
      <td>1.986095</td>
    </tr>
    <tr>
      <th>3491</th>
      <td>2014-04-22 03:39:00</td>
      <td>223.0</td>
      <td>2.163553</td>
    </tr>
    <tr>
      <th>3521</th>
      <td>2014-04-22 06:09:00</td>
      <td>207.0</td>
      <td>2.019778</td>
    </tr>
    <tr>
      <th>3650</th>
      <td>2014-04-22 16:54:00</td>
      <td>308.0</td>
      <td>3.237144</td>
    </tr>
    <tr>
      <th>3656</th>
      <td>2014-04-22 17:24:00</td>
      <td>230.0</td>
      <td>2.242455</td>
    </tr>
    <tr>
      <th>3669</th>
      <td>2014-04-22 18:29:00</td>
      <td>244.0</td>
      <td>2.404212</td>
    </tr>
    <tr>
      <th>3682</th>
      <td>2014-04-22 19:34:00</td>
      <td>656.0</td>
      <td>5.102451</td>
    </tr>
    <tr>
      <th>3683</th>
      <td>2014-04-22 19:39:00</td>
      <td>256.0</td>
      <td>2.589191</td>
    </tr>
    <tr>
      <th>3685</th>
      <td>2014-04-22 19:49:00</td>
      <td>338.0</td>
      <td>3.578690</td>
    </tr>
    <tr>
      <th>3695</th>
      <td>2014-04-22 20:39:00</td>
      <td>235.0</td>
      <td>2.286132</td>
    </tr>
    <tr>
      <th>3699</th>
      <td>2014-04-22 20:59:00</td>
      <td>239.0</td>
      <td>2.343428</td>
    </tr>
    <tr>
      <th>3711</th>
      <td>2014-04-22 21:59:00</td>
      <td>229.0</td>
      <td>2.237990</td>
    </tr>
    <tr>
      <th>3718</th>
      <td>2014-04-22 22:34:00</td>
      <td>227.0</td>
      <td>2.187403</td>
    </tr>
    <tr>
      <th>3759</th>
      <td>2014-04-23 01:59:00</td>
      <td>301.0</td>
      <td>3.165757</td>
    </tr>
    <tr>
      <th>3883</th>
      <td>2014-04-23 12:19:00</td>
      <td>222.0</td>
      <td>2.159113</td>
    </tr>
    <tr>
      <th>3910</th>
      <td>2014-04-23 14:34:00</td>
      <td>313.0</td>
      <td>3.342713</td>
    </tr>
    <tr>
      <th>3949</th>
      <td>2014-04-23 17:49:00</td>
      <td>299.0</td>
      <td>3.205282</td>
    </tr>
    <tr>
      <th>3973</th>
      <td>2014-04-23 19:49:00</td>
      <td>261.0</td>
      <td>2.702431</td>
    </tr>
    <tr>
      <th>4001</th>
      <td>2014-04-23 22:09:00</td>
      <td>209.0</td>
      <td>2.017933</td>
    </tr>
    <tr>
      <th>4018</th>
      <td>2014-04-23 23:34:00</td>
      <td>261.0</td>
      <td>2.702431</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 3 columns</p>
</div>




```python
ax2.plot(anomalies.index, anomalies.score, 'ko')
fig
```




![png](output_15_0.png)




```python

sagemaker.Session().delete_endpoint(rcf_inference.endpoint)
```


    ---------------------------------------------------------------------------

    ClientError                               Traceback (most recent call last)

    <ipython-input-34-11c53021fd05> in <module>()
          1 
    ----> 2 sagemaker.Session().delete_endpoint(rcf_inference.endpoint)
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sagemaker/session.py in delete_endpoint(self, endpoint_name)
       2406         """
       2407         LOGGER.info("Deleting endpoint with name: %s", endpoint_name)
    -> 2408         self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
       2409 
       2410     def delete_endpoint_config(self, endpoint_config_name):


    ~/anaconda3/envs/python3/lib/python3.6/site-packages/botocore/client.py in _api_call(self, *args, **kwargs)
        274                     "%s() only accepts keyword arguments." % py_operation_name)
        275             # The "self" in this scope is referring to the BaseClient.
    --> 276             return self._make_api_call(operation_name, kwargs)
        277 
        278         _api_call.__name__ = str(py_operation_name)


    ~/anaconda3/envs/python3/lib/python3.6/site-packages/botocore/client.py in _make_api_call(self, operation_name, api_params)
        584             error_code = parsed_response.get("Error", {}).get("Code")
        585             error_class = self.exceptions.from_code(error_code)
    --> 586             raise error_class(parsed_response, operation_name)
        587         else:
        588             return parsed_response


    ClientError: An error occurred (ValidationException) when calling the DeleteEndpoint operation: Could not find endpoint "arn:aws:sagemaker:us-east-2:234268085836:endpoint/randomcutforest-2020-02-08-06-52-06-063".



```python

```
