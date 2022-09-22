# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:20:16 2022

@author: xurui
"""
# download files from s3

from boto3.session import Session

AWSAccessKeyId=''
AWSSecretKey=''
session = Session(aws_access_key_id=AWSAccessKeyId, aws_secret_access_key=AWSSecretKey, region_name='us-west-2')
s3 = session.client("s3")
res=s3.list_objects_v2(Bucket='',Prefix='')
for i in range(len(res['Contents'])):
    key = res['Contents'][i]['Key']
    print('Start ', key)
    s3.download_file(Filename=key.split('''/''')[1], Key=key, Bucket='')
