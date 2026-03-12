import cv2
import numpy as np
import pandas as pd
import os
import logging
import pandas as pd
from pytube import YouTube

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_video(yt,save_path,video_id):
    my_streams = yt.streams.filter(file_extension='mp4',res='1080p',only_video=True)
    for streams in my_streams:
        print(f"Video itag : {streams.itag} Resolution : {streams.resolution} VCodec : {streams.codecs[0]}")
        video = yt.streams.get_by_itag(streams.itag)
        video.download(output_path=save_path,filename=str(video_id)+".mp4")
        print("Video is Downloading as",str(video_id)+".mp4")

'''
download all videos by iMiGue/Video.csv
return e.g., 'dataset_name'/1.mp4, where 1 denote the index of the video
'''

def download_video_by_dataset_subjectID(file,dataset_name,training_index,testing_index):
    """ Download videl from given csv file
    return 'dataset_name'/videos """
    # create folder for saving videos
    if os.path.exists(dataset_name):
        logging.info(f"The folder of dataset {dataset_name} is exist.")
    else:
        os.mkdir(dataset_name)
        os.mkdir(dataset_name + '/training_videos')
        os.mkdir(dataset_name + '/testing_videos')
        logging.info(f"The folder of dataset {dataset_name} has created successfully.")        
    # load csv file
    df = pd.read_csv(file)
    # download videos to created folder'
    for index, row in df.iterrows():
        try:
            yt = YouTube(row['Link']) 
            file_name = row['Video_id']
            if row['Sub_id'] in training_index:
                download_video(yt,dataset_name+'/training_videos/',file_name)
            elif row['Sub_id'] in testing_index:
                download_video(yt,dataset_name+'/testing_videos/',file_name)
        except:
            logging.warning(f"This link {row['Link']} is not avaialble now.")

def download_video_by_dataset_videoID(file,dataset_name,training_index,testing_index):
    """ Download videl from given csv file
    return 'dataset_name'/videos """
    # create folder for saving videos
    if os.path.exists(dataset_name):
        logging.info(f"The folder of dataset {dataset_name} is exist.")
    else:
        os.mkdir(dataset_name)
        os.mkdir(dataset_name + '/training_videos')
        os.mkdir(dataset_name + '/testing_videos')
        logging.info(f"The folder of dataset {dataset_name} has created successfully.")        
    # load csv file
    df = pd.read_csv(file)
    # download videos to created folder'
    for index, row in df.iterrows():
        try:
            yt = YouTube(row['Link']) 
            file_name = row['Video_id']
            if row['Video_id'] in training_index:
                download_video(yt,dataset_name+'/training_videos/',file_name)
            elif row['Video_id'] in testing_index:
                download_video(yt,dataset_name+'/testing_videos/',file_name)
        except:
            logging.warning(f"This link {row['Link']} is not avaialble now.")

# Example usage
'''
dataset_dir = '../dataset/'
dataset_name = 'iMiGUE'
download_video_by_dataset(dataset_dir + dataset_name + '/Video.csv',dataset_name)
'''