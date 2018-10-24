# Deep Video Deblurring for Hand-held Cameras
This is the demo code for [Deep Video Deblurring for Hand-held Cameras](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/). Given a stack of pre-aligned input frames, our network predicts a sharper central image. 

## Test

### Prepare data
- Prepare video frames:
	- Download and unzip test videos to `dataset`, from this [link](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset.zip)
	- or place your own test video frames under `dataset/qualitative_datasets/[video_file_name]/input`, 
	- or place your own test '.mp4' video files under `dataset/qualitative_datasets` and extract frames by running ```extractFrames.m```

- Align frames in Matlab, by running one of the following script: 
	- original implementation: ```preprocess/launcher.m```
	- my improved parallel implementation with breakpoint record and process bar: ```preprocess/generateAllAlignments.m``` with parameters of ('..\dataset\qualitative_datasets', 'testing')

- Outputs should be stored at `data/testing_real_all_nostab_[alignment]` under structure
```Shell
	/image_-2
	/image_-1
	/image_0
	/image_1
	/image_2
```
- Alternatively, you can download the pre-aligned qualitative videos from [here](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/testing_real_all_nostab_OF.zip), [here](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/testing_real_all_nostab_homography.zip), and [here](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/testing_real_all_nostab_nowarp.zip).

### Download pretrained weights
- Download and unzip pretrained weights into `logs`, from [here](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/pretrained.zip).

### Run prediction script
- Run script: `sh run_pred.sh`
- Results will be saved to `outImg`. 

## Train

### Prepare data
- Download and unzip train videos to `dataset`, from this [link](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset.zip), or place your own train video frames under:
	- input: `dataset/quantitative_datasets/[video_file_name]/input` 
	- ground truth: `dataset/quantitative_datasets/[video_file_name]/GT`

- Align frames in Matlab, by running ```preprocess/generateAllAlignments.m``` with parameters of ('..\dataset\quantitative_datasets','training')
- Outputs should be stored at `data/training_real_all_nostab_[alignment]` under same structure
- Argument the data in Matlab, by running ```preprocess/dataArgumentAndCrop.m```
- Argumented and Croped training data should be stored at ```../data/training_augumented_all_nostab_[alignment]` under same structure
- TODO: Read Argumented data images with torch, split them into 61 training videos and 10 testing videos, and convert images to torch7 files.
- TODO: Train the net with converted data.


### Citation
If you find this code useful for your research, please cite:
```
@inproceedings{su2017deep,
  title={Deep Video Deblurring for Hand-held Cameras},
  author={Su, Shuochen and Delbracio, Mauricio and Wang, Jue and Sapiro, Guillermo and Heidrich, Wolfgang and Wang, Oliver},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1279--1288},
  year={2017}
}
```
