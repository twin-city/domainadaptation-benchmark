import boto3
from botocore.config import Config
import os
from pathlib import Path

#%%



s3 = boto3.resource('s3',
                    use_ssl=True,
                    endpoint_url='https://s3.fr-par.scw.cloud',
                    region_name='fr-par',
                    aws_access_key_id="SCWP7DPK2XG19YKFH9QR",
                    aws_secret_access_key="53868785-4df9-42d4-a99a-c26a786782bb")

bucket = s3.Bucket('dataset-full')

key = ''
objs = list(bucket.objects.filter(Prefix=key))
for obj in objs:
    print(obj.key)

    # remove the file name from the object key
    obj_path = os.path.dirname(obj.key)

    # create nested directory structure
    (Path('twincity-dataset') / obj_path).mkdir(parents=True, exist_ok=True)

    # save file with full path locally
    bucket.download_file(obj.key, f'twincity-dataset/{str(obj.key)}')
